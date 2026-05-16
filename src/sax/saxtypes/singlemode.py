"""SAX SingleMode Types and type coercions.

This module defines types and validators specifically for single-mode optical
circuits, where each port represents a single optical mode. It includes
S-matrix representations and model validation functions.
"""

from __future__ import annotations

__all__ = [
    "InstanceName",
    "InstancePort",
    "ModelFactorySM",
    "ModelSM",
    "ModelsSM",
    "Port",
    "PortCombinationSM",
    "PortMapSM",
    "SCooModelFactorySM",
    "SCooModelSM",
    "SCooSM",
    "SDenseModelFactorySM",
    "SDenseModelSM",
    "SDenseSM",
    "SDictModelFactorySM",
    "SDictModelSM",
    "SDictSM",
    "STypeSM",
]

import inspect
from collections.abc import Callable
from typing import (
    Annotated,
    Any,
    TypeAlias,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from .core import ComplexArray, IntArray1D, Name, cast_string, val, val_name

T = TypeVar("T")


def val_instance_name(obj: Any) -> Port:
    """Validate an instance name allowing dots and angle brackets.

    Args:
        obj: The object to validate as an instance name.

    Returns:
        The validated instance name string.

    Raises:
        TypeError: If the string is not a valid instance name.

    Examples:
        Validate a string as an instance name:

        ```python
        import sax.saxtypes.singlemode as sm

        # Valid instance names
        result = sm.val_instance_name("coupler1")  # "coupler1"
        result = sm.val_instance_name("mzi.left_arm")  # "mzi.left_arm"
        result = sm.val_instance_name("array<0,1>")  # "array<0,1>"
        ```
    """
    return val_name(obj, type_name="InstanceName", extra_allowed_chars=(".", "<", ">"))


InstanceName: TypeAlias = Annotated[str, val(val_instance_name)]
"""An instance name allowing allowing an array index suffix '<x.y>'."""


def val_port(obj: Any) -> Port:
    """Validate a port name as a valid Python identifier.

    Args:
        obj: The object to validate as a port name.

    Returns:
        The validated port name string.

    Raises:
        TypeError: If the string is not a valid port name.

    Examples:
        Validate a string as a port name:

        ```python
        import sax.saxtypes.singlemode as sm

        # Valid port names
        result = sm.val_port("in0")  # "in0"
        result = sm.val_port("out1")  # "out1"
        result = sm.val_port("port")  # "port"
        ```
    """
    return val_name(obj, type_name="Port")


Port: TypeAlias = Annotated[str, val(val_port)]
"""A single-mode port name - must be a valid Python identifier."""


def val_instance_port(obj: Any) -> InstancePort:
    """Validate an instance port reference in 'instance,port' format.

    Args:
        obj: The object to validate as an instance port reference.

    Returns:
        The validated instance port string.

    Raises:
        TypeError: If the string is not a valid instance port reference.

    Examples:
        Validate a string as an instance port name:

        ```python
        import sax.saxtypes.singlemode as sm

        # Valid instance port references
        result = sm.val_instance_port("coupler1,in0")  # "coupler1,in0"
        result = sm.val_instance_port("mzi.left,out")  # "mzi.left,out"
        ```
    """
    s = cast_string(obj)
    parts = s.split(",")
    if len(parts) != 2:
        msg = f"an InstancePort should have exactly one ','-separator. Got: {obj!r}"
        raise TypeError(msg)
    inst, port = parts
    inst = val_instance_name(inst)
    port = val_port(port)
    return f"{inst},{port}"


InstancePort: TypeAlias = Annotated[str, val(val_instance_port)]
"""An instance port reference in the format 'instance_name,port_name'."""


PortMapSM: TypeAlias = dict[Port, int]
"""A mapping from single-mode port names to their matrix indices."""


PortCombinationSM: TypeAlias = tuple[Port, Port]
"""A pair of single-mode port names representing an S-parameter."""


SDictSM: TypeAlias = dict[PortCombinationSM, ComplexArray]
"""A sparse dictionary-based S-matrix representation.

A mapping from a port combination to an S-parameter or an array of S-parameters.

Examples:
    Creating an `SDictSM`:

    ```python
    sdict: sax.SDict = {
        ("in0", "out0"): 3.0,
    }
    ```

"""

SDenseSM: TypeAlias = tuple[ComplexArray, PortMapSM]
"""A dense S-matrix representation.

S-matrix (2D array) or multidimensional batched S-matrix (N+2)-D array with a port map.
If (N+2)-D array then the S-matrix dimensions are the last two.

Examples:
    Creating an `SDenseSM`:

    ```python
    Sd = jnp.arange(9, dtype=float).reshape(3, 3)
    port_map = {"in0": 0, "in1": 2, "out0": 1}
    sdense = Sd, port_map
    ```

"""

SCooSM: TypeAlias = tuple[IntArray1D, IntArray1D, ComplexArray, PortMapSM]
"""A sparse S-matrix in COO format (recommended for internal library use only).

An `SCoo` is a sparse matrix based representation of an S-matrix consisting
of three arrays and a port map. The three arrays represent the output port
indices [`int`] (rows), input port indices [`int`] (columns) and the
S-matrix values [`ComplexFloat`] of the sparse matrix. The port map maps a
port name [`str`] to a port index [`int`].

Only these four arrays **together** and in this specific **order** are considered a
valid `SCoo` representation!

Examples:
    Creating an `SCooSM`:

    ```python
    Si = jnp.arange(3, dtype=int)
    Sj = jnp.array([0, 1, 0], dtype=int)
    Sx = jnp.array([3.0, 4.0, 1.0])
    port_map = {"in0": 0, "in1": 2, "out0": 1}
    scoo: sax.SCoo = (Si, Sj, Sx, port_map)
    ```

Note:
    This representation is only recommended for internal library use. Please don't
    write user-facing code using this representation.

"""

STypeSM: TypeAlias = SDictSM | SCooSM | SDenseSM
"""Any S-Matrix type [SDict, SDense, SCOO]."""


def val_sax_callable(model: Any) -> Callable:
    """Validate that a function can be used as a SAX model.

    A valid SAX model must be callable, have a signature, and follow
    specific parameter conventions (no positional-only, no *args, no **kwargs,
    all parameters must have defaults).

    Args:
        model: The function to validate.

    Returns:
        The validated callable model.

    Raises:
        TypeError: If the function violates SAX model conventions.

    Examples:
        Validate a callable SAX model:

        ```python
        def my_model(wl=1.55, coupling=0.5):
            return {"in0,out0": 1.0}


        validated = sm.val_sax_callable(my_model)
        ```
    """
    if not callable(model):
        msg = f"NOT_CALLABLE: A SAX model should be callable. Got: {model!r}."
        raise TypeError(msg)

    model_name = getattr(model, "__name__", str(model))
    try:
        sig = inspect.signature(model)
    except Exception as e:
        msg = (
            f"NO_SIGNATURE: Function '{model_name}' cannot be used as a SAX model. "
            "It has no function signature."
        )
        raise TypeError(msg) from e
    for name, param in sig.parameters.items():
        if param.kind == param.POSITIONAL_ONLY:
            msg = (
                f"NO_POSITIONAL_ONLY: A SAX model should not have positional-only"
                f"arguments. Got: '{model_name}' with param '{name!r}'"
            )
            raise TypeError(msg)
        if param.kind == param.VAR_POSITIONAL:
            msg = (
                f"NO_VAR_POSITIONAL: A SAX model should not have var-positional "
                f"arguments. Got: '{model_name}' with var-param '*{name}'."
            )
            raise TypeError(msg)
        if param.kind == param.VAR_KEYWORD:
            msg = (
                f"NO_VAR_KEYWORD: A SAX model should not have var-keyword "
                f"arguments. Got: '{model_name}' with var-keyword '**{name}'."
            )
            raise TypeError(msg)
        if param.default is inspect.Parameter.empty:
            msg = (
                "NO_DEFAULT: A SAX model should not have arguments without defaults. "
                f"Got: '{model_name}' with param '{name}'."
            )
            raise TypeError(msg)
    return model


def has_callable_return_annotation(model: Callable) -> bool:
    """Check if a model has a callable return type annotation.

    Args:
        model: The model function to check.

    Returns:
        True if the return annotation indicates a callable type.
    """
    return is_callable_return_annotation(inspect.signature(model).return_annotation)


def is_callable_return_annotation(annot: Any) -> bool:
    """Check if a type annotation represents a callable type.

    Args:
        annot: The type annotation to check.

    Returns:
        True if the annotation represents a callable type.
    """
    if isinstance(annot, str) and (
        "model" in annot.lower() or "callable" in annot.lower()
    ):
        return True
    origin = get_origin(annot)
    if origin is Union or origin is Annotated:
        return is_callable_return_annotation(get_args(annot)[0])
    return origin is Callable


def val_not_callable_annotated(model: Callable) -> Callable:
    """Validate that a model does not have a callable return annotation.

    Models should return S-matrix data, not other callables. If a function
    returns a callable, it's likely a model factory instead.

    Args:
        model: The model function to validate.

    Returns:
        The validated model function.

    Raises:
        TypeError: If the function appears to be a model factory.
    """
    annot = inspect.signature(model).return_annotation
    if has_callable_return_annotation(model):
        model_name = getattr(model, "__name__", str(model))
        msg = (
            "IS_MODEL_FACTORY: A SAX model should return an SDict, "
            f"SDense, SCoo or SType. Got '{model_name}' returning {annot}. "
            "This indicates that this is in fact a ModelFactory."
        )
        raise TypeError(msg)
    return model


def val_callable_annotated(model: Callable) -> Callable:
    """Validate that a model factory has a callable return annotation.

    Model factories should be annotated to indicate they return callable models.

    Args:
        model: The model factory function to validate.

    Returns:
        The validated model factory function.

    Raises:
        TypeError: If the function lacks proper annotation.
    """
    annot = inspect.signature(model).return_annotation
    if not has_callable_return_annotation(model):
        model_name = getattr(model, "__name__", str(model))
        msg = (
            "NOT_ANNOTATED: A SAX ModelFactory should be annotated with a Callable "
            "return annotation to make sure it's not mistaken as a Model. "
            f"Got: '{model_name}' returning {annot}."
        )
        raise TypeError(msg)
    return model


def val_model(model: Any) -> ModelSM:
    """Validate a single-mode SAX model function.

    Args:
        model: The model function to validate.

    Returns:
        The validated single-mode model.

    Raises:
        TypeError: If validation fails.
    """
    return val_not_callable_annotated(val_sax_callable(model))


SDictModelSM: TypeAlias = Annotated[Callable[..., SDictSM], val(val_model)]
"""A keyword-only function that produces a single-mode SDict S-matrix."""

SDenseModelSM: TypeAlias = Annotated[Callable[..., SDenseSM], val(val_model)]
"""A keyword-only function that produces a single-mode SDense S-matrix."""


SCooModelSM: TypeAlias = Annotated[Callable[..., SCooSM], val(val_model)]
"""A keyword-only function that produces a single-mode SCoo S-matrix."""


ModelSM: TypeAlias = Annotated[
    SDictModelSM | SDenseModelSM | SCooModelSM, val(val_model)
]
"""A keyword-only function that produces any single-mode S-matrix type."""


def val_model_factory(model: Any) -> ModelFactorySM:
    """Validate a single-mode SAX model factory function.

    Args:
        model: The model factory function to validate.

    Returns:
        The validated single-mode model factory.

    Raises:
        TypeError: If validation fails.
    """
    return val_callable_annotated(val_sax_callable(model))


SDictModelFactorySM: TypeAlias = Annotated[
    Callable[..., SDictModelSM], val(val_model_factory)
]
"""A keyword-only function that produces a single-mode SDict model."""


SDenseModelFactorySM: TypeAlias = Annotated[
    Callable[..., SDenseModelSM], val(val_model_factory)
]
"""A keyword-only function that produces a single-mode SDense model."""

SCooModelFactorySM: TypeAlias = Annotated[
    Callable[..., SCooModelSM], val(val_model_factory)
]
"""A keyword-only function that produces a single-mode SCoo model."""


ModelFactorySM: TypeAlias = Annotated[
    SDictModelFactorySM | SDenseModelFactorySM | SCooModelFactorySM,
    val(val_model_factory),
]
"""A keyword-only function that produces any single-mode model."""

ModelsSM: TypeAlias = dict[Name, ModelSM]
"""A mapping from model names to single-mode model functions."""
