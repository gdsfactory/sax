"""SAX Types and type coercions.

This module provides the core type system for SAX, including type validation
functions and type aliases for scalars and arrays. It handles type coercion
between Python/NumPy types and JAX arrays with proper validation.

References:
    Numpy type reference: https://numpy.org/doc/stable/reference/arrays.scalars.html
"""

from __future__ import annotations

__all__ = [
    "ArrayLike",
    "Bool",
    "BoolArray",
    "BoolArrayLike",
    "BoolLike",
    "Complex",
    "ComplexArray",
    "ComplexArray1D",
    "ComplexArray1DLike",
    "ComplexArrayLike",
    "ComplexLike",
    "Float",
    "FloatArray",
    "FloatArray1D",
    "FloatArray1DLike",
    "FloatArray2D",
    "FloatArray2DLike",
    "FloatArrayLike",
    "FloatLike",
    "IOLike",
    "Int",
    "IntArray",
    "IntArray1D",
    "IntArray1DLike",
    "IntArrayLike",
    "IntLike",
]

from collections.abc import Callable
from contextlib import suppress
from functools import partial, wraps
from types import UnionType
from typing import (
    Annotated,
    Any,
    Literal,
    LiteralString,
    Protocol,
    TypeAlias,
    TypeVar,
    cast,
    get_args,
    overload,
)

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from pydantic import BeforeValidator, PlainValidator
from pydantic_core import PydanticCustomError

T = TypeVar("T")

ArrayLike: TypeAlias = Array | np.ndarray
"""Anything that can turn into an array with ndim>=1."""


def _val(fun: Callable, **val_kwargs: Any) -> Callable:
    @wraps(fun)
    def new(*args: Any, **kwargs: Any) -> Any:
        kwargs.update(val_kwargs)
        try:
            return fun(*args, **kwargs)
        except TypeError as e:
            msg = str(e)
            if ":" in msg:
                subtype, msg = msg.split(":", 1)
            else:
                subtype = ""
            raise PydanticCustomError(
                cast(LiteralString, subtype),
                cast(LiteralString, msg),
            ) from e

    return new


def val(fun: Callable, **val_kwargs: Any) -> PlainValidator:
    """Create a PlainValidator from a validation function.

    Args:
        fun: The validation function to wrap.
        **val_kwargs: Additional keyword arguments to pass to the validation function.

    Returns:
        A Pydantic PlainValidator wrapping the function.

    Raises:
        PydanticCustomError: If validation fails.
    """
    return PlainValidator(_val(fun, **val_kwargs))


def bval(fun: Callable, **val_kwargs: Any) -> BeforeValidator:
    """Create a BeforeValidator from a validation function.

    Args:
        fun: The validation function to wrap.
        **val_kwargs: Additional keyword arguments to pass to the validation function.

    Returns:
        A Pydantic PlainValidator wrapping the function.

    Raises:
        PydanticCustomError: If validation fails.
    """
    return BeforeValidator(_val(fun, **val_kwargs))


def _val_item_type(
    obj: Any,
    *,
    strict: bool,
    cast: bool,
    type_cast: Callable[..., T],
    type_def: Any,
    type_name: str,
) -> T:
    """Validate and optionally cast a scalar value to a specific type.

    Args:
        obj: The object to validate.
        strict: Whether to use strict validation.
        cast: Whether to cast the result to the target type.
        type_cast: Function to cast to the target type.
        type_def: Type definition for validation.
        type_name: Name of the type for error messages.

    Returns:
        The validated (and optionally cast) value.

    Raises:
        TypeError: If validation fails.
    """
    from ..utils import maybe

    item = _val_0d(obj, type_name=type_name).item()
    if not isinstance(item, _get_annotated_type(type_def)):
        arr = maybe(np.asarray)(item)
        if arr is None or not np.can_cast(arr, type_cast, casting="same_kind"):  # type: ignore[arg-type]
            msg = f"NOT_{type_name.upper()}: Cannot validate {obj!r} into {type_name}."
            raise TypeError(msg)
        if strict:
            msg = (
                f"NOT_{type_name.upper()}: Strict validation does not allow casting "
                f"{obj!r} [dtype={arr.dtype}] into {type_name}. "
                f"Note: use {type_name}Like type for less strict type checking."
            )
            raise TypeError(msg)
    if cast:
        return type_cast(item)
    return item


@overload
def val_bool(obj: Any, *, strict: bool = ..., cast: Literal[True] = True) -> Bool: ...


@overload
def val_bool(
    obj: Any, *, strict: bool = ..., cast: Literal[False] = False
) -> BoolLike: ...


def val_bool(obj: Any, *, strict: bool = False, cast: bool = True) -> BoolLike:
    """Validate and optionally cast an object to a boolean.

    Args:
        obj: The object to validate.
        strict: Whether to use strict validation (no casting allowed).
        cast: Whether to cast the result to bool.

    Returns:
        The validated boolean value.

    Raises:
        TypeError: If validation fails.

    Examples:
        Validate and cast an object to a boolean:

        ```python
        import sax.saxtypes as sxt

        # Valid boolean values
        result = sxt.val_bool(True)  # True
        result = sxt.val_bool(1)  # True (cast from int)
        result = sxt.val_bool(0.0)  # False (cast from float)
        ```
    """
    return _val_item_type(
        obj,
        strict=strict,
        cast=cast,
        type_cast=bool,
        type_def=Bool,
        type_name="Bool",
    )


Bool: TypeAlias = Annotated[bool | np.bool_, val(val_bool, strict=False)]
"""Any boolean value (Python bool or NumPy boolean)."""


@overload
def val_int(obj: Any, *, strict: bool = ..., cast: Literal[True] = True) -> Int: ...


@overload
def val_int(
    obj: Any, *, strict: bool = ..., cast: Literal[False] = False
) -> IntLike: ...


def val_int(obj: Any, *, strict: bool = False, cast: bool = True) -> IntLike:
    """Validate and optionally cast an object to an integer.

    Args:
        obj: The object to validate.
        strict: Whether to use strict validation (no casting from bool).
        cast: Whether to cast the result to int.

    Returns:
        The validated integer value.

    Raises:
        TypeError: If validation fails.

    Examples:
        Validate and cast an object to an integer:

        ```python
        import sax.saxtypes as sxt

        # Valid integer values
        result = sxt.val_int(42)  # 42
        result = sxt.val_int(3.0)  # 3 (cast from float)
        result = sxt.val_int("5")  # 5 (cast from string)
        ```
    """
    from ..utils import maybe

    if strict and maybe(partial(val_bool, strict=False, cast=False))(obj) is not None:
        msg = (
            f"NOT_INT: Strict validation does not allow casting {obj} [bool] into Int. "
        )
        raise TypeError(msg)
    return _val_item_type(
        obj,
        strict=strict,
        cast=cast,
        type_cast=int,
        type_def=Int,
        type_name="Int" if strict else "IntLike",
    )


Int: TypeAlias = Annotated[int | np.signedinteger, val(val_int, strict=False)]
"""Any signed integer (Python int or NumPy signed integer)."""


@overload
def val_float(obj: Any, *, strict: bool = ..., cast: Literal[True] = True) -> Float: ...


@overload
def val_float(
    obj: Any, *, strict: bool = ..., cast: Literal[False] = False
) -> FloatLike: ...


def val_float(obj: Any, *, strict: bool = False, cast: bool = True) -> FloatLike:
    """Validate and optionally cast an object to a float.

    Args:
        obj: The object to validate.
        strict: Whether to use strict validation.
        cast: Whether to cast the result to float.

    Returns:
        The validated float value.

    Raises:
        TypeError: If validation fails.

    Examples:
        Validate and cast an object to a float:

        ```python
        import sax.saxtypes as sxt

        # Valid float values
        result = sxt.val_float(3.14)  # 3.14
        result = sxt.val_float(42)  # 42.0 (cast from int)
        result = sxt.val_float("2.5")  # 2.5 (cast from string)
        ```
    """
    return _val_item_type(
        obj,
        strict=strict,
        cast=cast,
        type_cast=float,
        type_def=Float,
        type_name="Float" if strict else "FloatLike",
    )


Float: TypeAlias = Annotated[float | np.floating, val(val_float, strict=False)]
"""Any floating-point number (Python float or NumPy floating)."""


@overload
def val_complex(
    obj: Any, *, strict: bool = ..., cast: Literal[True] = True
) -> Complex: ...


@overload
def val_complex(
    obj: Any, *, strict: bool = ..., cast: Literal[False] = False
) -> ComplexLike: ...


def val_complex(obj: Any, *, strict: bool = False, cast: bool = True) -> ComplexLike:
    """Validate and optionally cast an object to a complex number.

    Args:
        obj: The object to validate.
        strict: Whether to use strict validation.
        cast: Whether to cast the result to complex.

    Returns:
        The validated complex value.

    Raises:
        TypeError: If validation fails.

    Examples:
        Validate and cast an object to a complex number:

        ```python
        import sax.saxtypes as sxt

        # Valid complex values
        result = sxt.val_complex(1 + 2j)  # (1+2j)
        result = sxt.val_complex(3.14)  # (3.14+0j) (cast from float)
        result = sxt.val_complex(42)  # (42+0j) (cast from int)
        ```
    """
    return _val_item_type(
        obj,
        strict=strict,
        cast=cast,
        type_cast=complex,
        type_def=Complex,
        type_name="Complex" if strict else "ComplexLike",
    )


Complex: TypeAlias = Annotated[
    complex | np.complexfloating, val(val_complex, strict=False)
]
"""Any complex number (Python complex or NumPy complex floating)."""


def _val_array_type(  # noqa: C901
    obj: Any,
    *,
    strict: bool,
    cast: bool,
    type_def: Any,
    default_dtype: Any,
    type_name: str,
) -> Array:
    """Validate and optionally cast an object to a JAX array of specific type.

    This is the core array validation function used by all array validators.
    It handles dimensionality checking, dtype validation, and casting.

    Args:
        obj: The object to validate.
        strict: Whether to use strict validation (no numpy arrays, no broadcasting).
        cast: Whether to cast the result to the default dtype.
        type_def: Type definition for validation.
        default_dtype: Default dtype to cast to if casting is enabled.
        type_name: Name of the type for error messages.

    Returns:
        The validated JAX array.

    Raises:
        TypeError: If validation fails.
    """
    if strict:
        if isinstance(obj, np.ndarray):
            msg = (
                f"NOT_{type_name.upper()}: Strict validation does not allow casting "
                f"a numpy array {obj!r} {type(obj)} into a jax-array of "
                f"type {type_name}. Note: use {type_name}Like type for less "
                "strict type checking."
            )
            raise TypeError(msg)
        if not isinstance(obj, Array):
            msg = (
                f"NOT_{type_name.upper()}: Strict validation does not allow casting "
                f"the scalar {obj!r} {type(obj)} into an array type [{type_name}]. "
                f"Note: use {type_name}Like type for less strict type checking."
            )
            raise TypeError(msg)

    short_message = f"Cannot validate {obj!r} into a JAX array."
    arr = _val_array(obj, error_message=short_message)
    ndim = _get_annotated_ndim(type_def)
    if ndim is not None and arr.ndim != ndim:
        if strict:
            msg = (
                f"NOT_{type_name.upper()}: Strict validation does not allow "
                f"broadcasting the {int(arr.ndim)}D array {arr!r} into a {ndim}D "
                f"array. Note: use {type_name}Like type for less strict type checking."
            )
            raise TypeError(msg)
        _ndim = int(arr.ndim)
        while arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim > ndim:
            msg = (
                f"NOT_{type_name.upper()}: Unable to cast the {_ndim}D "
                f"array {obj!r} into a {ndim}D array. "
            )
            raise TypeError(msg)
        while arr.ndim < ndim:
            arr = arr[None]
    if not np.issubdtype(arr.dtype, _get_annotated_dtype(type_def)):
        if not np.can_cast(arr, default_dtype, casting="same_kind"):
            msg = f"NOT_{type_name.upper()}: Cannot validate {obj!r} into {type_name}."
            raise TypeError(msg)
        if strict:
            msg = (
                f"NOT_{type_name.upper()}: Strict validation does not allow casting "
                f"{obj!r} [dtype={arr.dtype}] into {type_name}. "
                f"Note: use {type_name}Like type for less strict type checking."
            )
            raise TypeError(msg)
    if cast:
        return jnp.asarray(arr, dtype=default_dtype)
    return arr


@overload
def val_bool_array(
    obj: Any, *, strict: bool = ..., cast: Literal[True] = True
) -> BoolArray: ...


@overload
def val_bool_array(
    obj: Any, *, strict: bool = ..., cast: Literal[False] = False
) -> BoolArrayLike: ...


def val_bool_array(
    obj: Any, *, strict: bool = False, cast: bool = True
) -> BoolArrayLike:
    """Validate and optionally cast an object to a boolean array.

    Args:
        obj: The object to validate (array-like or scalar).
        strict: Whether to use strict validation.
        cast: Whether to cast the result to the default boolean dtype.

    Returns:
        The validated boolean array.

    Raises:
        TypeError: If validation fails.

    Examples:
        Validate and cast an object to a boolean array:

        ```python
        import sax.saxtypes as sxt
        import jax.numpy as jnp

        # Valid boolean arrays
        result = sxt.val_bool_array([True, False, True])
        result = sxt.val_bool_array(jnp.array([1, 0, 1]))
        result = sxt.val_bool_array([[True, False], [False, True]])
        ```
    """
    return _val_array_type(
        obj,
        strict=strict,
        cast=cast,
        type_def=BoolArray if strict else BoolArrayLike,
        default_dtype=np.bool_,
        type_name="BoolArray" if strict else "BoolArrayLike",
    )


BoolArray: TypeAlias = Annotated[Array, np.bool_, val(val_bool_array, strict=False)]
"""N-dimensional boolean array (JAX Array with boolean dtype)."""


@overload
def val_int_array(
    obj: Any, *, strict: bool = ..., cast: Literal[True] = True
) -> IntArray: ...


@overload
def val_int_array(
    obj: Any, *, strict: bool = ..., cast: Literal[False] = False
) -> IntArrayLike: ...


def val_int_array(obj: Any, *, strict: bool = False, cast: bool = True) -> IntArrayLike:
    """Validate and optionally cast an object to an integer array.

    Args:
        obj: The object to validate (array-like or scalar).
        strict: Whether to use strict validation.
        cast: Whether to cast the result to the default integer dtype.

    Returns:
        The validated integer array.

    Raises:
        TypeError: If validation fails.

    Examples:
        Validate and cast an object to an integer array:

        ```python
        import sax.saxtypes as sxt
        import jax.numpy as jnp

        # Valid integer arrays
        result = sxt.val_int_array([1, 2, 3])
        result = sxt.val_int_array(jnp.array([1.0, 2.0, 3.0]))  # cast from float
        result = sxt.val_int_array([[1, 2], [3, 4]])
        ```
    """
    return _val_array_type(
        obj,
        strict=strict,
        cast=cast,
        type_def=IntArray if strict else IntArrayLike,
        default_dtype=np.int64 if _x64_enabled() else np.int32,
        type_name="IntArray" if strict else "IntArrayLike",
    )


IntArray: TypeAlias = Annotated[
    Array, np.signedinteger, val(val_int_array, strict=False)
]
"""N-dimensional signed integer array (JAX Array with integer dtype)."""


@overload
def val_float_array(
    obj: Any, *, strict: bool = ..., cast: Literal[True] = True
) -> FloatArray: ...


@overload
def val_float_array(
    obj: Any, *, strict: bool = ..., cast: Literal[False] = False
) -> FloatArrayLike: ...


def val_float_array(
    obj: Any, *, strict: bool = False, cast: bool = True
) -> FloatArrayLike:
    """Validate and optionally cast an object to a float array.

    Args:
        obj: The object to validate (array-like or scalar).
        strict: Whether to use strict validation.
        cast: Whether to cast the result to the default float dtype.

    Returns:
        The validated float array.

    Raises:
        TypeError: If validation fails.

    Examples:
        Validate and cast an object to a float array:

        ```python
        import sax.saxtypes as sxt
        import jax.numpy as jnp

        # Valid float arrays
        result = sxt.val_float_array([1.0, 2.5, 3.14])
        result = sxt.val_float_array([1, 2, 3])  # cast from int
        result = sxt.val_float_array(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
        ```
    """
    return _val_array_type(
        obj,
        strict=strict,
        cast=cast,
        type_def=FloatArray if strict else FloatArrayLike,
        default_dtype=np.float64 if _x64_enabled() else np.float32,
        type_name="FloatArray" if strict else "FloatArrayLike",
    )


FloatArray: TypeAlias = Annotated[
    Array, np.floating, val(val_float_array, strict=False)
]
"""N-dimensional floating-point array (JAX Array with float dtype)."""


@overload
def val_complex_array(
    obj: Any, *, strict: bool = ..., cast: Literal[True] = True
) -> ComplexArray: ...


@overload
def val_complex_array(
    obj: Any, *, strict: bool = ..., cast: Literal[False] = False
) -> ComplexArrayLike: ...


def val_complex_array(
    obj: Any, *, strict: bool = False, cast: bool = True
) -> ComplexArrayLike:
    """Validate and optionally cast an object to a complex array.

    Args:
        obj: The object to validate (array-like or scalar).
        strict: Whether to use strict validation.
        cast: Whether to cast the result to the default complex dtype.

    Returns:
        The validated complex array.

    Raises:
        TypeError: If validation fails.

    Examples:
        Validate and cast an object to a complex array:

        ```python
        import sax.saxtypes as sxt
        import jax.numpy as jnp

        # Valid complex arrays
        result = sxt.val_complex_array([1 + 2j, 3 + 4j])
        result = sxt.val_complex_array([1.0, 2.0, 3.0])  # cast from float
        result = sxt.val_complex_array(jnp.array([[1, 2], [3, 4]]))  # cast from int
        ```
    """
    return _val_array_type(
        obj,
        strict=strict,
        cast=cast,
        type_def=ComplexArray if strict else ComplexArrayLike,
        default_dtype=np.complex128 if _x64_enabled() else np.complex64,
        type_name="ComplexArray" if strict else "ComplexArrayLike",
    )


ComplexArray: TypeAlias = Annotated[
    Array, np.complexfloating, val(val_complex_array, strict=False)
]
"""N-dimensional complex array (JAX Array with complex dtype)."""


@overload
def val_int_array_1d(
    obj: Any, *, strict: bool = ..., cast: Literal[True] = True
) -> IntArray1D: ...


@overload
def val_int_array_1d(
    obj: Any, *, strict: bool = ..., cast: Literal[False] = False
) -> IntArray1DLike: ...


def val_int_array_1d(
    obj: Any, *, strict: bool = False, cast: bool = True
) -> IntArray1DLike:
    """Validate and optionally cast an object to a 1D integer array.

    Args:
        obj: The object to validate (1D array-like or scalar).
        strict: Whether to use strict validation.
        cast: Whether to cast the result to the default integer dtype.

    Returns:
        The validated 1D integer array.

    Raises:
        TypeError: If validation fails or array is not 1D.

    Examples:
        Validate and cast an object to a 1D integer array:

        ```python
        import sax.saxtypes as sxt

        # Valid 1D integer arrays
        result = sxt.val_int_array_1d([1, 2, 3, 4])
        result = sxt.val_int_array_1d([1.0, 2.0, 3.0])  # cast from float
        ```
    """
    return _val_array_type(
        obj,
        strict=strict,
        cast=cast,
        type_def=IntArray1D if strict else IntArray1DLike,
        default_dtype=np.int64 if _x64_enabled() else np.int32,
        type_name="IntArray1D" if strict else "Intarray1DLike",
    )


IntArray1D: TypeAlias = Annotated[
    ArrayLike, np.signedinteger, 1, val(val_int_array_1d, strict=False)
]
"""1-dimensional signed integer array (JAX Array with integer dtype)."""


@overload
def val_float_array_1d(
    obj: Any, *, strict: bool = ..., cast: Literal[True] = True
) -> FloatArray1D: ...


@overload
def val_float_array_1d(
    obj: Any, *, strict: bool = ..., cast: Literal[False] = False
) -> FloatArray1DLike: ...


def val_float_array_1d(
    obj: Any, *, strict: bool = False, cast: bool = True
) -> FloatArray1DLike:
    """Validate and optionally cast an object to a 1D float array.

    Args:
        obj: The object to validate (1D array-like or scalar).
        strict: Whether to use strict validation.
        cast: Whether to cast the result to the default float dtype.

    Returns:
        The validated 1D float array.

    Raises:
        TypeError: If validation fails or array is not 1D.

    Examples:
        Validate and cast an object to a 1D float array:

        ```python
        import sax.saxtypes as sxt

        # Valid 1D float arrays
        result = sxt.val_float_array_1d([1.0, 2.5, 3.14])
        result = sxt.val_float_array_1d([1, 2, 3])  # cast from int
        ```
    """
    return _val_array_type(
        obj,
        strict=strict,
        cast=cast,
        type_def=FloatArray1D if strict else FloatArray1DLike,
        default_dtype=np.float64 if _x64_enabled() else np.float32,
        type_name="FloatArray1D" if strict else "FloatArray1DLike",
    )


FloatArray1D: TypeAlias = Annotated[
    ArrayLike, np.floating, 1, val(val_float_array_1d, strict=False)
]
"""1-dimensional floating-point array (JAX Array with float dtype)."""


@overload
def val_float_array_2d(
    obj: Any, *, strict: bool = ..., cast: Literal[True] = True
) -> FloatArray2D: ...


@overload
def val_float_array_2d(
    obj: Any, *, strict: bool = ..., cast: Literal[False] = False
) -> FloatArray2DLike: ...


def val_float_array_2d(
    obj: Any, *, strict: bool = False, cast: bool = True
) -> FloatArray2DLike:
    """Validate and optionally cast an object to a 2D float array.

    Args:
        obj: The object to validate (2D array-like).
        strict: Whether to use strict validation.
        cast: Whether to cast the result to the default float dtype.

    Returns:
        The validated 2D float array.

    Raises:
        TypeError: If validation fails or array is not 2D.

    Examples:
        Validate and cast an object to a 2D float array:

        ```python
        import sax.saxtypes as sxt

        # Valid 2D float arrays
        result = sxt.val_float_array_2d([[1.0, 2.0], [3.0, 4.0]])
        result = sxt.val_float_array_2d([[1, 2], [3, 4]])  # cast from int
        ```
    """
    return _val_array_type(
        obj,
        strict=strict,
        cast=cast,
        type_def=FloatArray2D if strict else FloatArray2DLike,
        default_dtype=np.float64 if _x64_enabled() else np.float32,
        type_name="FloatArray2D" if strict else "FloatArray2DLike",
    )


FloatArray2D: TypeAlias = Annotated[
    ArrayLike, np.floating, 2, val(val_float_array_2d, strict=False)
]
"""2-dimensional floating-point array (JAX Array with float dtype)."""


@overload
def val_complex_array_1d(
    obj: Any, *, strict: bool = ..., cast: Literal[True] = True
) -> ComplexArray1D: ...


@overload
def val_complex_array_1d(
    obj: Any, *, strict: bool = ..., cast: Literal[False] = False
) -> ComplexArray1DLike: ...


def val_complex_array_1d(
    obj: Any, *, strict: bool = False, cast: bool = True
) -> ComplexArray1DLike:
    """Validate and optionally cast an object to a 1D complex array.

    Args:
        obj: The object to validate (1D array-like or scalar).
        strict: Whether to use strict validation.
        cast: Whether to cast the result to the default complex dtype.

    Returns:
        The validated 1D complex array.

    Raises:
        TypeError: If validation fails or array is not 1D.

    Examples:
        Validate and cast an object to a 1D complex array:

        ```python
        import sax.saxtypes as sxt

        # Valid 1D complex arrays
        result = sxt.val_complex_array_1d([1 + 2j, 3 + 4j])
        result = sxt.val_complex_array_1d([1.0, 2.0, 3.0])  # cast from float
        result = sxt.val_complex_array_1d([1, 2, 3])  # cast from int
        ```
    """
    return _val_array_type(
        obj,
        strict=strict,
        cast=cast,
        type_def=ComplexArray1D if strict else ComplexArray1DLike,
        default_dtype=np.complex128 if _x64_enabled() else np.complex64,
        type_name="ComplexArray1D" if strict else "ComplexArray1DLike",
    )


ComplexArray1D: TypeAlias = Annotated[
    ArrayLike, np.complexfloating, 1, val(val_complex_array_1d, strict=False)
]
"""1-dimensional complex array (JAX Array with complex dtype)."""

BoolLike: TypeAlias = Annotated[bool | np.bool_, val(val_bool, cast=False)]
"""Anything that can be cast into a Bool without loss of data."""

IntLike: TypeAlias = Annotated[int | np.integer, val(val_int, cast=False)]
"""Anything that can be cast into an Int without loss of data."""

FloatLike: TypeAlias = Annotated[
    IntLike | float | np.floating, val(val_float, cast=False)
]
"""Anything that can be cast into a Float without loss of data."""

ComplexLike: TypeAlias = Annotated[
    FloatLike | complex | np.inexact, val(val_complex, cast=False)
]
"""Anything that can be cast into a Complex without loss of data."""

BoolArrayLike: TypeAlias = Annotated[
    ArrayLike | BoolLike, np.bool_, val(val_bool_array, cast=False)
]
"""Anything that can be cast into an N-dimensional Bool array without loss of data."""

IntArrayLike: TypeAlias = Annotated[
    ArrayLike | IntLike, np.integer, val(val_int_array, cast=False)
]
"""Anything that can be cast into an N-dimensional Int array without loss of data."""

FloatArrayLike: TypeAlias = Annotated[
    ArrayLike | FloatLike, np.floating, val(val_float_array, cast=False)
]
"""Anything that can be cast into an N-dimensional Float array without loss of data."""

ComplexArrayLike: TypeAlias = Annotated[
    ArrayLike | ComplexLike, np.inexact, val(val_complex_array, cast=False)
]
"""Anything that can be cast into an N-dim Complex array without loss of data."""

IntArray1DLike: TypeAlias = Annotated[
    IntArrayLike, np.integer, 1, val(val_int_array_1d, cast=False)
]
"""Anything that can be cast into a 1-dimensional integer array without loss of data."""

FloatArray1DLike: TypeAlias = Annotated[
    FloatArrayLike, np.floating, 1, val(val_float_array_1d, cast=False)
]
"""Anything that can be cast into a 1-dimensional float array without loss of data."""

FloatArray2DLike: TypeAlias = Annotated[
    FloatArrayLike, np.floating, 2, val(val_float_array_2d, cast=False)
]
"""Anything that can be cast into a 2-dimensional float array without loss of data."""

ComplexArray1DLike: TypeAlias = Annotated[
    ComplexArrayLike, np.inexact, 1, val(val_complex_array_1d, cast=False)
]
"""Anything that can be cast into a 1-dimensional complex array without loss of data."""


def cast_string(obj: Any) -> str:
    """Cast an object to a string, decoding bytes if necessary.

    Args:
        obj: The object to cast to string.

    Returns:
        The string representation of the object.

    Examples:
        Validate and cast various types to string:

        ```python
        import sax.saxtypes.core as core

        result = core.cast_string("hello")  # "hello"
        result = core.cast_string(b"hello")  # "hello" (decoded)
        result = core.cast_string(42)  # "42"
        ```
    """
    if isinstance(obj, bytes):
        obj = obj.decode()
    return str(obj)


def val_name(
    obj: Any, *, type_name: str = "Name", extra_allowed_chars: tuple[str, ...] = ()
) -> str:
    """Validate that a string is a valid Python identifier.

    Args:
        obj: The object to validate as a name.
        type_name: The type name for error messages.
        extra_allowed_chars: Additional characters to allow beyond standard identifiers.

    Returns:
        The validated name string.

    Raises:
        TypeError: If the string is not a valid identifier.

    Examples:
        Validate a string as a valid Python identifier:

        ```python
        import sax.saxtypes.core as core

        # Valid names
        result = core.val_name("my_var")  # "my_var"
        result = core.val_name("Component1")  # "Component1"

        # With extra allowed characters
        result = core.val_name("inst.port", extra_allowed_chars=(".",))  # "inst.port"
        ```
    """
    s = cast_string(obj)
    _s = s
    for c in extra_allowed_chars:
        _s = _s.replace(c, "_")
    if not _s.isidentifier():
        msg = (
            f"A {type_name!r} string should be a valid python identifier. Got: {s!r}. "
            "note: python identifiers should only contain letters, numbers or "
            "underscores. The first character should not be a number."
        )
        raise TypeError(msg)
    return s


Name: TypeAlias = Annotated[str, val(val_name)]
"""A valid Python identifier - contains only letters, numbers, and underscores."""


class IOLike(Protocol):
    """Protocol for file-like objects that can be read from.

    This protocol defines the interface for objects that can be used
    for reading text data, such as file objects or StringIO objects.
    """

    def read(self, size: int | None = -1, /) -> str:
        """Read and return up to size characters.

        Args:
            size: Maximum number of characters to read. If None or -1, read all.

        Returns:
            The read characters as a string.
        """
        ...


def _get_annotated_type(annotated: Any) -> type | UnionType:
    """Extract the base type from an Annotated type alias."""
    return get_args(annotated)[0]


def _get_annotated_dtype(annotated: Any) -> Any:
    """Extract the dtype from an Annotated array type alias."""
    return get_args(annotated)[1]


def _get_annotated_ndim(annotated: Any) -> Any:
    """Extract the number of dimensions from an Annotated array type alias."""
    with suppress(Exception):
        return int(get_args(annotated)[2])
    return None


def _val_array(obj: Any, error_message: str = "") -> Array:
    """Convert an object to a JAX array with error handling.

    Args:
        obj: The object to convert to an array.
        error_message: Additional error message context.

    Returns:
        The JAX array representation of the object.

    Raises:
        TypeError: If the object cannot be converted to an array.
    """
    try:
        return jnp.asarray(obj)
    except TypeError as e:
        msg = f"NOT_ARRAYLIKE: {error_message}."
        raise TypeError(msg) from e


def _val_0d(obj: Any, *, type_name: str = "0D") -> Array:
    """Validate that an object can be converted to a 0D (scalar) array.

    Args:
        obj: The object to validate.
        type_name: The type name for error messages.

    Returns:
        A 0D JAX array containing the scalar value.

    Raises:
        TypeError: If the object has more than 0 dimensions.
    """
    short_message = f"Cannot cast {obj!r} to {type_name}"
    arr = _val_array(obj)
    if arr.ndim > 0:
        msg = f"NOT_{type_name.upper()}: {short_message}. Got {arr.ndim}D."
        raise TypeError(msg)
    return arr


def _x64_enabled() -> bool:
    """Check if JAX is configured to use 64-bit precision.

    Returns:
        True if 64-bit precision is enabled, False otherwise.
    """
    return jax.config.jax_enable_x64  # type: ignore[reportAttributeAccessIssue]
