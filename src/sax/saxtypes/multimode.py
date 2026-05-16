"""SAX Multimode Types and type coercions.

This module defines types and validators specifically for multi-mode optical
circuits, where ports can support multiple optical modes (e.g., TE0, TE1, TM0).
Ports are specified using the 'port@mode' notation.

References:
    Numpy type reference: https://numpy.org/doc/stable/reference/arrays.scalars.html
"""

from __future__ import annotations

__all__ = [
    "ModelFactoryMM",
    "ModelMM",
    "ModelsMM",
    "PortCombinationMM",
    "PortMapMM",
    "PortMode",
    "SCooMM",
    "SCooModelFactoryMM",
    "SCooModelMM",
    "SDenseMM",
    "SDenseModelFactoryMM",
    "SDenseModelMM",
    "SDictMM",
    "SDictModelFactoryMM",
    "SDictModelMM",
    "STypeMM",
]

from collections.abc import Callable
from typing import (
    Annotated,
    Any,
    TypeAlias,
)

from .core import ComplexArray, IntArray1D, Name, val
from .singlemode import (
    cast_string,
    val_callable_annotated,
    val_not_callable_annotated,
    val_port,
    val_sax_callable,
)


def val_mode(obj: Any) -> Mode:
    """Validate a mode specification string.

    Modes can be any string identifier, including numerical modes like '@0', '@1'
    or named modes like '@TE0', '@TM0'.

    Args:
        obj: The object to validate as a mode.

    Returns:
        The validated mode string.

    Examples:
        Validating a string as a mode:

        ```python
        import sax.saxtypes.multimode as mm

        # Valid mode specifications
        result = mm.val_mode("0")  # "0" (numerical mode)
        result = mm.val_mode("TE0")  # "TE0" (named mode)
        result = mm.val_mode("TM1")  # "TM1" (named mode)
        ```
    """
    return str(obj)  # just a string to allow '@0' etc.


Mode: TypeAlias = Annotated[str, val(val_mode)]
"""A mode identifier string (e.g., '0', 'TE0', 'TM1')."""


def val_port_mode(obj: Any) -> PortMode:
    """Validate a port-mode specification in 'port@mode' format.

    Args:
        obj: The object to validate as a port-mode specification.

    Returns:
        The validated port-mode string.

    Raises:
        TypeError: If the string doesn't follow 'port@mode' format.

    Examples:
        Validating a port-mode string:

        ```python
        import sax.saxtypes.multimode as mm

        # Valid port-mode specifications
        result = mm.val_port_mode("in0@0")  # "in0@0"
        result = mm.val_port_mode("out1@TE0")  # "out1@TE0"
        result = mm.val_port_mode("port@TM1")  # "port@TM1"
        ```
    """
    s = cast_string(obj)
    parts = s.split("@")
    if len(parts) != 2:
        msg = f"a PortMode should have exactly one '@'-separator. Got: {obj!r}"
        raise TypeError(msg)
    port, mode = parts
    port = val_port(port)
    mode = val_mode(mode)
    return f"{port}@{mode}"


PortMode: TypeAlias = Annotated[str, val(val_port_mode)]
"""A port-mode specification in the format 'port_name@mode_name'."""


PortMapMM: TypeAlias = dict[PortMode, int]
"""A mapping from multi-mode port-mode names to their matrix indices."""


PortCombinationMM: TypeAlias = tuple[PortMode, PortMode]
"""A pair of multi-mode port-mode names representing an S-parameter."""


SDictMM: TypeAlias = dict[PortCombinationMM, ComplexArray]
"""A sparse dictionary-based S-matrix representation.

A mapping from a port combination to an S-parameter or an array of S-parameters.

Examples:
    Creating an `SDictMM`:

    ```python
    sdict: sax.SDict = {
        ("in0", "out0"): 3.0,
    }
    ```

"""

SDenseMM: TypeAlias = tuple[ComplexArray, PortMapMM]
"""A dense S-matrix representation.

S-matrix (2D array) or multidimensional batched S-matrix (N+2)-D array with a port map.
If (N+2)-D array then the S-matrix dimensions are the last two.

Examples:
    Creating an `SDenseMM`:

    ```python
    Sd = jnp.arange(9, dtype=float).reshape(3, 3)
    port_map = {"in0": 0, "in1": 2, "out0": 1}
    sdense = Sd, port_map
    ```

"""

SCooMM: TypeAlias = tuple[IntArray1D, IntArray1D, ComplexArray, PortMapMM]
"""A sparse S-matrix in COO format (recommended for internal library use only).

An `SCoo` is a sparse matrix based representation of an S-matrix consisting of three
arrays and a port map. The three arrays represent the input port indices [`int`],
output port indices [`int`] and the S-matrix values [`ComplexFloat`] of the sparse
matrix. The port map maps a port name [`str`] to a port index [`int`].

Only these four arrays **together** and in this specific **order** are considered a
valid `SCoo` representation!

Examples:
    Creating an `SCooMM':

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

STypeMM: TypeAlias = SDictMM | SCooMM | SDenseMM
"""Any S-Matrix type [SDict, SDense, SCOO]."""


def val_model(model: Any) -> ModelMM:
    """Validate a multi-mode SAX model function.

    Args:
        model: The model function to validate.

    Returns:
        The validated multi-mode model.

    Raises:
        TypeError: If validation fails.
    """
    return val_not_callable_annotated(val_sax_callable(model))


SDictModelMM: TypeAlias = Annotated[Callable[..., SDictMM], val(val_model)]
"""A keyword-only function that produces a multi-mode SDict S-matrix."""

SDenseModelMM: TypeAlias = Annotated[Callable[..., SDenseMM], val(val_model)]
"""A keyword-only function that produces a multi-mode SDense S-matrix."""


SCooModelMM: TypeAlias = Annotated[Callable[..., SCooMM], val(val_model)]
"""A keyword-only function that produces a multi-mode SCoo S-matrix."""


ModelMM: TypeAlias = Annotated[
    SDenseModelMM | SCooModelMM | SDictModelMM, val(val_model)  # type: ignore[reportInvalidTypeForm]
]
"""A keyword-only function that produces any multi-mode S-matrix type."""


def val_model_factory(model: Any) -> ModelFactoryMM:
    """Validate a multi-mode SAX model factory function.

    Args:
        model: The model factory function to validate.

    Returns:
        The validated multi-mode model factory.

    Raises:
        TypeError: If validation fails.
    """
    return val_callable_annotated(val_sax_callable(model))


SDictModelFactoryMM: TypeAlias = Annotated[
    Callable[..., SDictModelMM], val(val_model_factory)
]
"""A keyword-only function that produces a multi-mode SDict model."""


SDenseModelFactoryMM: TypeAlias = Annotated[
    Callable[..., SDenseModelMM], val(val_model_factory)
]
"""A keyword-only function that produces a multi-mode SDense model."""

SCooModelFactoryMM: TypeAlias = Annotated[
    Callable[..., SCooModelMM], val(val_model_factory)
]
"""A keyword-only function that produces a multi-mode SCoo model."""


ModelFactoryMM: TypeAlias = Annotated[
    SDictModelFactoryMM | SDenseModelFactoryMM | SCooModelFactoryMM,
    val(val_model_factory),
]
"""A keyword-only function that produces any multi-mode model."""

ModelsMM: TypeAlias = dict[Name, ModelMM]
"""A mapping from model names to multi-mode model functions."""
