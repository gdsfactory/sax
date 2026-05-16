"""Neural fitting module."""

from __future__ import annotations

import io
import sys
import warnings
from collections.abc import Callable
from functools import partial
from typing import IO, Annotated, Any, TypeAlias, overload

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import sympy
from jaxtyping import Array
from typing_extensions import TypedDict

import sax

__all__ = [
    "NeuralFitResult",
    "PRNGKey",
    "Params",
    "get_neural_fit_functions_str",
    "neural_fit",
    "neural_fit_equations",
    "write_neural_fit_functions",
]


def neural_fit(
    data: pd.DataFrame,
    targets: list[str],
    features: list[str] | None = None,
    hidden_dims: tuple[int, ...] = (10,),
    learning_rate: float = 0.003,
    num_epochs: int = 1000,
    random_seed: int = 42,
    *,
    activation_fn: Callable = jnp.tanh,
    loss_fn: Callable = sax.mse,
    progress_bar: bool = True,
) -> NeuralFitResult:
    """Neural fitting function with JMP-like capabilities.

    Args:
        data: Input data with features and target.
        targets: Names of the target columns.
        features: Names of the feature columns.
            If None, uses all numeric columns except target columns.
        hidden_dims: Hidden layer dimensions, e.g., (10, 5) for two hidden layers.
        learning_rate: Learning rate for optimization.
        num_epochs: Number of training epochs per tour.
        random_seed: Random seed for reproducibility.
        activation_fn: The activation function to use in the network.
        loss_fn: The loss function to use for training.
        progress_bar: Whether to show a progress bar during training.

    Returns:
        Dictionary containing trained model data

    Raises:
        RuntimeError: If all training tours fail.
    """
    df_work = data.copy()

    if features is None:
        features = [
            col
            for col in data.select_dtypes(include=[np.number]).columns
            if col not in targets
        ]

    X = jnp.array(df_work[features].values, dtype=jnp.float32)
    Y = jnp.array(df_work[targets].values, dtype=jnp.float32)

    X_mean = jnp.mean(X, axis=0, keepdims=True)
    X_std = jnp.std(X, axis=0, keepdims=True)

    Y_mean = jnp.mean(Y, axis=0, keepdims=True)
    Y_std = jnp.std(Y, axis=0, keepdims=True)

    X_norm = (X - X_mean) / X_std
    Y_norm = (Y - Y_mean) / Y_std

    input_dim = X.shape[1]
    output_dim = Y.shape[1]

    init_fn, forward_fn = create_network(
        input_dim, hidden_dims, output_dim, activation_fn
    )

    key = jax.random.PRNGKey(random_seed)
    params, _ = train_network(
        X_norm,
        Y_norm,
        init_fn,
        forward_fn,
        key,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        progress_bar=progress_bar,
        loss_fn=loss_fn,
    )

    @overload
    def predict_fn(X: Array) -> Array: ...

    @overload
    def predict_fn(X: pd.DataFrame) -> pd.DataFrame: ...

    def predict_fn(X: Array | pd.DataFrame) -> Array | pd.DataFrame:
        """Make predictions on new data.

        Args:
            X: New input features.

        Returns:
            Predictions.
        """
        df = None
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            X = jnp.array(X[features].values)

        X_norm = (X - X_mean) / X_std
        Y_norm = forward_fn(params, X_norm)
        Y = Y_norm * Y_std + Y_mean

        if df is not None:
            pred_cols = np.array([f"{c}_pred" for c in targets])
            df_pred = pd.DataFrame(np.asarray(Y), columns=pred_cols)
            return pd.concat([df, df_pred], axis=1)

        return Y

    return NeuralFitResult(
        params=params,
        features=features,
        targets=targets,
        hidden_dims=hidden_dims,
        forward_fn=forward_fn,
        predict_fn=predict_fn,
        activation_fn=activation_fn,
        X_norm=sax.Normalization(X_mean, X_std),
        Y_norm=sax.Normalization(Y_mean, Y_std),
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        final_loss=loss_fn(Y_norm, forward_fn(params, X_norm)),
    )


def neural_fit_equations(result: NeuralFitResult) -> dict[str, Equation]:
    """Convert neural fit result to symbolic equations.

    Args:
        result: Result from neural_fit function.

    Returns:
        Dictionary mapping target names to symbolic equations.
    """
    X_list = [sympy.Symbol(f) for f in result["features"]]
    activation_fn = getattr(sympy, result["activation_fn"].__name__, None)
    if activation_fn is None:
        msg = f"Activation function {result['activation_fn']} is not supported."
        raise ValueError(msg)
    x = np.asarray(
        [
            (x - n) / s
            for (x, n, s) in zip(
                X_list,
                np.atleast_1d(np.asarray(result["X_norm"].mean).squeeze()),
                np.atleast_1d(np.asarray(result["X_norm"].std).squeeze()),
                strict=True,
            )
        ]
    )
    for layer_params in result["params"][:-1]:
        w = np.asarray(layer_params["w"])
        b = np.asarray(layer_params["b"])
        x = np.asarray([activation_fn(xx) for xx in (x @ w + b)])

    w = np.asarray(result["params"][-1]["w"])
    b = np.asarray(result["params"][-1]["b"])
    Y_list = np.asarray(
        [
            y * s + n
            for (y, n, s) in zip(
                x @ w + b,
                np.atleast_1d(np.asarray(result["Y_norm"].mean).squeeze()),
                np.atleast_1d(np.asarray(result["Y_norm"].std).squeeze()),
                strict=True,
            )
        ]
    )
    equations = dict(zip(result["targets"], Y_list, strict=True))
    return equations


def write_neural_fit_functions(
    result: NeuralFitResult,
    io: IO = sys.stdout,
    *,
    with_imports: bool = True,
) -> None:
    """Write neural fit as a python function.

    Args:
        result: Result from neural_fit function.
        io: where to write the functions to
        with_imports: Whether to include import statements in the output.
    """
    act_fn = result["activation_fn"]
    eqs = neural_fit_equations(result)
    for target, eq in eqs.items():
        if with_imports:
            io.write("import sax\n")
            io.write("import jax.numpy as jnp\n")
        io.write(
            _render_function_template(
                target=target, eq=eq, act=act_fn, args=result["features"]
            )
        )


def get_neural_fit_functions_str(
    result: NeuralFitResult,
    *,
    with_imports: bool = True,
) -> str:
    """Get the neural fit function string.

    Args:
        result: Result from neural_fit function.
        with_imports: Whether to include import statements in the output.
    """
    buf = io.StringIO()
    write_neural_fit_functions(result, buf, with_imports=with_imports)
    return buf.getvalue()


def eval_neural_fit(
    result: NeuralFitResult,
    func: str,
    **kwargs: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Evaluate a neural fit.

    Args:
        result: Result from neural_fit function.
        func: which output function to use.
        kwargs: Parameters to evaluate the function at
            (should match the columns in the dataframe the neural fit was trained on).
    """
    namespace = {}
    s = get_neural_fit_functions_str(result, with_imports=True)
    exec(s, namespace)  # noqa: S102
    return namespace[func](**kwargs)


Equation: TypeAlias = Annotated[Any, "Equation"]
"""A sumpy-equation."""

PRNGKey: TypeAlias = Annotated[Any, "PRNGKey"]
"""The jax.PRNGKey used to generate random weights and biases."""


Params = TypedDict(
    "Params",
    {
        "w": Array,
        "b": Array,
    },
)
"""Parameters for a single layer in the neural network.

Attributes:
    w: Weights for the layer.
    b: Biases for the layer.
"""

NeuralFitResult = TypedDict(
    "NeuralFitResult",
    {
        "params": list[Params],
        "features": list[str],
        "targets": list[str],
        "hidden_dims": tuple[int, ...],
        "forward_fn": Callable,
        "predict_fn": Callable,
        "activation_fn": Callable,
        "X_norm": sax.Normalization,
        "Y_norm": sax.Normalization,
        "learning_rate": float,
        "num_epochs": int,
        "final_loss": float,
    },
)
"""Complete result from neural_fit function.

Attributes:
    params: Trained model parameters.
"""


def create_network(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    activation_fn: Callable,
) -> tuple[Callable, Callable]:
    """Create a neural network function.

    Args:
        input_dim: Number of input features.
        hidden_dims: Tuple of hidden layer dimensions.
        output_dim: Number of output dimensions.
        activation_fn: Activation function to use.

    Returns:
        Tuple of (parameter initialization function, forward function).
    """

    def init_fn(key: PRNGKey) -> list[Params]:
        """Initialize network parameters.

        Args:
            key: JAX random key.

        Returns:
            List of parameter dictionaries for each layer.
        """
        keys = jax.random.split(key, len(hidden_dims) + 1)
        params = []

        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            w = jax.random.normal(keys[i], (prev_dim, hidden_dim)) * jnp.sqrt(
                2.0 / prev_dim
            )
            b = jnp.zeros(hidden_dim)
            params.append({"w": w, "b": b})
            prev_dim = hidden_dim

        # Output layer
        w_out = jax.random.normal(keys[-1], (prev_dim, output_dim)) * jnp.sqrt(
            2.0 / prev_dim
        )
        b_out = jnp.zeros(output_dim)
        params.append({"w": w_out, "b": b_out})

        return params

    def forward_fn(params: list[Params], x: Array) -> Array:
        """Forward pass through the network.

        Args:
            params: Network parameters.
            x: Input array.

        Returns:
            Network output.
        """
        for layer_params in params[:-1]:
            x = jnp.dot(x, layer_params["w"]) + layer_params["b"]
            x = activation_fn(x)

        x = jnp.dot(x, params[-1]["w"]) + params[-1]["b"]
        return x

    return init_fn, forward_fn


def _compute_loss(
    params: list[Params],
    forward_fn: Callable,
    X: Array,
    Y: Array,
    loss_fn: Callable,
) -> float:
    """Compute loss with optional robustness and penalty.

    Args:
        params: Network parameters.
        forward_fn: Forward function.
        X: Input features.
        Y: Target values.
        loss_fn: Loss function to use (e.g., sax.mse, sax.huber_loss, ...).

    Returns:
        Computed loss value.
    """
    Y_pred = forward_fn(params, X)
    return loss_fn(Y, Y_pred)


def train_network(
    X_norm: Array,
    Y_norm: Array,
    init_fn: Callable[..., list[Params]],
    forward_fn: Callable,
    key: PRNGKey,
    learning_rate: float = 0.01,
    num_epochs: int = 1000,
    *,
    progress_bar: bool = True,
    loss_fn: Callable = sax.mse,
) -> tuple[list, list[float]]:
    """Train the neural network with validation.

    Args:
        X_norm: Normalized input features.
        Y_norm: Normalized target values.
        init_fn: Parameter initialization function.
        forward_fn: Forward pass function.
        key: JAX random key.
        learning_rate: Learning rate for optimizer.
        num_epochs: Number of training epochs.
        progress_bar: Whether to show a progress bar during training.
        loss_fn: Loss function to use for training.

    Returns:
        Tuple of (best parameters, training history).
    """
    params = init_fn(key)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)  # type: ignore[reportAgumentType]

    # Loss function with penalty
    loss_fn = jax.jit(
        partial(
            _compute_loss,
            forward_fn=forward_fn,
            X=X_norm,
            Y=Y_norm,
            loss_fn=loss_fn,
        )
    )

    # Training loop
    losses = []
    best_loss = jnp.inf
    best_params = params

    grad_fn = jax.grad(loss_fn)

    _tqdm = _noop if not progress_bar else partial(_get_tqdm(), total=num_epochs)
    for _ in (pb := _tqdm(range(num_epochs))):
        grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params: list[Params] = optax.apply_updates(params, updates)  # type: ignore[reportArgumentType, reportAssignmentType]
        loss = loss_fn(params)
        losses.append(float(loss))
        if progress_bar:
            pb.set_postfix(
                loss=f"{loss:.4f}",
            )
        if loss < best_loss:
            best_loss = loss
            best_params = params

    return best_params, losses


def _get_tqdm() -> Callable:
    """Get the tqdm function, handling imports."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        from tqdm.autonotebook import tqdm
    return tqdm


def _noop(x: Any) -> Any:  # noqa: ANN401
    return x


argument_template = "    {arg}: sax.FloatArrayLike,"
_function_template = """
def {target}(
{args}
) -> sax.FloatArray:
    return jnp.asarray({eq})

"""


def _render_function_template(
    *, target: str, eq: Equation, act: Callable, args: list[str]
) -> str:
    rendered_args = "\n".join([argument_template.format(arg=arg) for arg in args])
    return _function_template.format(
        target=target,
        eq=str(eq).replace(act.__name__, f"jnp.{act.__name__}"),
        args=rendered_args,
    )
