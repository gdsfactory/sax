"""SAX Loss Functions."""

from __future__ import annotations

from typing import cast

import jax.numpy as jnp

from .saxtypes import ComplexArray

__all__ = [
    "huber_loss",
    "l2_reg",
    "mse",
]


def mse(x: ComplexArray, y: ComplexArray) -> float:
    """Compute mean squared error between two complex arrays.

    Args:
        x: First complex array for comparison.
        y: Second complex array for comparison.

    Returns:
        Mean squared error as a float value.

    Example:
        ```python
        import jax.numpy as jnp

        x = jnp.array([1 + 2j, 3 + 4j])
        y = jnp.array([1 + 1j, 3 + 3j])
        loss = mse(x, y)
        ```
    """
    return cast(float, (abs(x - y) ** 2).mean())


def huber_loss(x: ComplexArray, y: ComplexArray, delta: float = 0.5) -> float:
    """Compute Huber loss between two complex arrays.

    The Huber loss combines the best properties of MSE and MAE losses. It is
    quadratic for small errors and linear for large errors, making it robust
    to outliers.

    Args:
        x: First complex array for comparison.
        y: Second complex array for comparison.
        delta: Threshold parameter controlling the transition between quadratic
            and linear regions. Defaults to 0.5.

    Returns:
        Huber loss as a float value.

    Example:
        ```python
        import jax.numpy as jnp

        x = jnp.array([1 + 2j, 3 + 4j])
        y = jnp.array([1 + 1j, 3 + 3j])
        loss = huber_loss(x, y, delta=1.0)
        ```
    """
    loss = ((delta**2) * ((1.0 + (abs(x - y) / delta) ** 2) ** 0.5 - 1.0)).mean()
    return cast(float, loss)


def l2_reg(weights: dict[str, ComplexArray]) -> float:
    """Compute L2 regularization loss for model weights.

    L2 regularization adds a penalty term proportional to the sum of squares
    of the model parameters, helping to prevent overfitting.

    Args:
        weights: Dictionary of weight arrays where keys starting with 'w' or 'b'
            are considered for regularization (typically weights and biases).

    Returns:
        L2 regularization loss normalized by total number of elements.

    Example:
        ```python
        import jax.numpy as jnp

        weights = {
            "w1": jnp.array([1 + 1j, 2 + 2j]),
            "b1": jnp.array([0.1 + 0.2j]),
            "other": jnp.array([5 + 5j]),  # ignored (doesn't start with w/b)
        }
        reg_loss = l2_reg(weights)
        ```
    """
    numel = 0
    loss = 0.0
    for w in (v for k, v in weights.items() if k[0] in ("w", "b")):
        numel = numel + w.size
        loss = loss + (jnp.abs(w) ** 2).sum()
    return cast(float, loss / numel)
