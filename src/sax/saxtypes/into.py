"""Type conversion utilities for SAX.

This module provides utilities for safely converting between different SAX types
using Pydantic validation. It offers both strict conversion (raises errors) and
optional conversion (returns None on failure).

Examples:
    ```python
    import sax.saxtypes as sxt

    # Strict conversion
    x: int = 5
    y: sxt.Complex = sxt.into[sxt.Complex](x)
    print(y)  # (5+0j)

    # Optional conversion
    if (z := sxt.try_into[sxt.Float](x)) is not None:
        print(z)  # 5.0
    ```
"""

from __future__ import annotations

__all__ = ["into", "try_into"]

from collections.abc import Callable
from typing import Any, TypeVar, cast, get_args, overload

from pydantic import PlainValidator, TypeAdapter, ValidationError
from typing_extensions import _AnnotatedAlias

T = TypeVar("T")


class Into(type):
    """Metaclass for the `into` type converter.

    This metaclass enables the `into[Type]` syntax for type conversion.
    It creates type-safe converters that use Pydantic validation.
    """

    @overload
    def __getitem__(cls, key: type[T]) -> Callable[..., T]: ...

    @overload
    def __getitem__(cls, key: str) -> Callable[..., Any]: ...

    @overload
    def __getitem__(cls, key: Any) -> Callable[..., Any]: ...

    def __getitem__(
        cls,
        key: type[T] | str | Any,
    ) -> Callable[..., T | Any]:
        """Get a type converter for the specified key.

        Args:
            key: The target type, string name, or type specification.

        Returns:
            A converter function that validates and converts objects.
        """
        if isinstance(key, str) and hasattr(cls, key):
            return getattr(cls, key)

        if isinstance(key, _AnnotatedAlias):
            validator = get_args(key)[-1]
            if isinstance(validator, PlainValidator):
                return getattr(validator.func, "__wrapped__", validator.func)
        return _wrapped_validator(cast(type[T], key))


def _wrapped_validator(key: type[T]) -> Callable[..., T]:
    """Create a wrapped validator function for a given type.

    Args:
        key: The target type to validate against.

    Returns:
        A validator function that converts objects to the target type.

    Raises:
        TypeError: If the object cannot be converted to the target type.
    """

    def into_type(obj: Any) -> T:
        """Convert an object to the target type.

        Args:
            obj: The object to convert.

        Returns:
            The converted object.

        Raises:
            TypeError: If conversion fails.
        """
        try:
            return TypeAdapter(key).validate_python(obj, strict=False)
        except ValidationError as e:
            raise TypeError(str(e)) from e

    return into_type


class into(metaclass=Into):  # noqa: N801
    """Type converter utility for strict type conversion.

    Use this class with bracket notation to convert objects to specific SAX types.
    Raises TypeError if conversion fails.

    Examples:
        Validate and cast objects to SAX or python types:

        ```python
        import sax.saxtypes as sxt

        # Convert to different numeric types
        x = sxt.into[sxt.Complex](5)  # (5+0j)
        y = sxt.into[sxt.Float](42)  # 42.0
        z = sxt.into[sxt.ComplexArray]([1, 2, 3])  # Array([1.+0.j, 2.+0.j, 3.+0.j])

        # Convert to SAX-specific types
        port = sxt.into[sxt.Port]("in0")  # "in0"
        name = sxt.into[sxt.Name]("component1")  # "component1"
        ```
    """


class TryInto(type):
    """Metaclass for the `try_into` optional type converter.

    This metaclass enables the `try_into[Type]` syntax for optional type conversion.
    Returns None instead of raising errors when conversion fails.
    """

    @overload
    def __getitem__(cls, key: type[T]) -> Callable[..., T | None]: ...

    @overload
    def __getitem__(cls, key: str) -> Callable[..., Any | None]: ...

    @overload
    def __getitem__(cls, key: Any) -> Callable[..., Any | None]: ...

    def __getitem__(cls, key: type[T] | str | Any) -> Callable[..., T | Any | None]:
        """Get an optional type converter for the specified key.

        Args:
            key: The target type, string name, or type specification.

        Returns:
            A converter function that returns the converted object or None on failure.
        """
        from ..utils import maybe

        return maybe(into[key])


class try_into(metaclass=TryInto):  # noqa: N801
    """Optional type converter utility that returns None on failure.

    Use this class with bracket notation to safely attempt type conversion.
    Returns None instead of raising errors when conversion fails.

    Examples:
        Validate and cast objects to SAX or python types safely:

        ```python
        import sax.saxtypes as sxt

        # Safe conversion attempts
        result = sxt.try_into[sxt.Float]("3.14")  # 3.14
        result = sxt.try_into[sxt.Float]("invalid")  # None

        # Use in conditional expressions
        if (value := sxt.try_into[sxt.Complex]("1+2j")) is not None:
            print(f"Converted: {value}")  # (1+2j)

        # Handle arrays safely
        arr = sxt.try_into[sxt.FloatArray]([[1, 2], [3, 4]])
        if arr is not None:
            print(f"Array shape: {arr.shape}")  # (2, 2)
        ```
    """
