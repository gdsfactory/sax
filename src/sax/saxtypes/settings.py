"""SAX Settings type definitions.

This module defines the type system for configuration settings used throughout
SAX. Settings can be nested dictionaries containing arrays, strings, or other
nested settings, providing a flexible configuration system.
"""

from __future__ import annotations

__all__ = [
    "Settings",
    "SettingsValue",
]

from typing import Annotated, Any, TypeAlias

from .core import val, val_complex_array


def val_settings_value(value: Any) -> SettingsValue:
    """Validate a value that can be used in a settings dictionary.

    Settings values can be strings, None, nested dictionaries, or complex arrays.
    This validator ensures type safety while maintaining flexibility.

    Args:
        value: The value to validate.

    Returns:
        The validated settings value.

    Examples:
        Validate an object into a settings-value:

        ```python
        import sax.saxtypes.settings as settings

        # Valid settings values
        result = settings.val_settings_value("text")  # String
        result = settings.val_settings_value(None)  # None
        result = settings.val_settings_value([1.0, 2.0, 3.0])  # Array
        result = settings.val_settings_value({"nested": "dict"})  # Dict
        ```
    """
    if isinstance(value, str) or value is None:
        return value
    if isinstance(value, dict):
        return {k: val_settings_value(v) for k, v in value.items()}
    return val_complex_array(value, strict=False, cast=False)


SettingsValue: TypeAlias = Annotated[Any, val(val_settings_value)]
"""Any value that can be stored in a settings dictionary"""
# SettingsValue: TypeAlias = Annotated[
#     Settings | ComplexArrayLike | str | None, val(val_settings_value)
# ]


def val_settings(settings: dict) -> Settings:
    """Validate a settings dictionary.

    Ensures the input is a dictionary and validates all nested values.

    Args:
        settings: The dictionary to validate as settings.

    Returns:
        The validated settings dictionary.

    Raises:
        TypeError: If the input is not a dictionary.

    Examples:
        Validate a dictionary into a settings mapping:

        ```python
        import sax.saxtypes.settings as settings

        # Valid settings dictionary
        result = settings.val_settings(
            {
                "wavelength": 1.55,
                "models": {"coupler": {"coupling": 0.5}, "waveguide": {"neff": 2.4}},
            }
        )
        ```
    """
    if not isinstance(settings, dict):
        msg = "Settings should be a dictionary. Got: {settings!r}."
        raise TypeError(msg)
    return {k: val_settings_value(v) for k, v in settings.items()}


Settings: TypeAlias = Annotated[dict[str, SettingsValue], val(val_settings)]
"""A (possibly nested) settings mapping for configuring models and circuits.

Settings provide a hierarchical way to configure SAX models and circuits.
Top-level keys often correspond to global parameters or instance names,
while nested dictionaries contain model-specific parameters.

Examples:
    Define settings for a model or circuit:

    ```python
    mzi_settings: sax.Settings = {
        "wl": 1.5,  # global wavelength setting
        "lft": {"coupling": 0.5},  # settings for the left coupler
        "top": {"neff": 3.4},      # settings for the top waveguide
        "rgt": {"coupling": 0.3},  # settings for the right coupler
    }

    array_settings: sax.Settings = {
        "wavelengths": [1.50, 1.51, 1.52, 1.53, 1.54, 1.55],
        "temperature": 25.0,
        "component_settings": {
            "loss_db_per_cm": 0.1,
            "group_index": 4.2
        }
    }
    ```
"""
