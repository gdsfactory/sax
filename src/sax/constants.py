"""Constants and magic numbers.

This module defines physical constants and optical communication band parameters
according to ITU standards. The optical bands (O, E, S, C, L) represent different
wavelength ranges used in fiber optic communications:

- O-band (Original): 1260-1360 nm
- E-band (Extended): 1360-1460 nm
- S-band (Short): 1460-1530 nm
- C-band (Conventional): 1530-1565 nm
- L-band (Long): 1565-1625 nm

Reference: https://www.fiberlabs.com/glossary/optical-communication-band/
"""

from __future__ import annotations

import jax.numpy as jnp

from sax.saxtypes.core import FloatArray1D

__all__ = [
    "C_M_S",
    "C_UM_S",
    "DEFAULT_FREQUENCY",
    "DEFAULT_MODE",
    "DEFAULT_MODES",
    "DEFAULT_WL_STEP",
    "EPS",
    "WL_C",
    "WL_C_MAX",
    "WL_C_MIN",
    "WL_E",
    "WL_E_MAX",
    "WL_E_MIN",
    "WL_L",
    "WL_L_MAX",
    "WL_L_MIN",
    "WL_O",
    "WL_O_MAX",
    "WL_O_MIN",
    "WL_S",
    "WL_S_MAX",
    "WL_S_MIN",
    "wl_c",
    "wl_e",
    "wl_l",
    "wl_o",
    "wl_s",
]

# Physical constants
EPS: float = 1e-12
"""Small numerical epsilon for determining which s-values can be considered zero. """

C_M_S: float = 299792458.0
"""Speed of light in vacuum (m/s)."""

C_UM_S: float = 1e6 * C_M_S
"""Speed of light in vacuum (μm/s)."""

DEFAULT_FREQUENCY: float = 5e9
"""Default radio frequency (Hz)."""

DEFAULT_MODE: str = "TE"
"""Default optical mode."""

DEFAULT_MODES: tuple[str, ...] = ("TE", "TM")
"""Default multimode configuration."""

DEFAULT_WL_STEP: float = 0.0001
""" Default wavelength step for array generation (μm)."""

WL_O_MIN: float = 1.260
"""O-band minimum wavelength (μm)."""

WL_O_MAX: float = 1.360
"""O-band maximum wavelength (μm)."""

WL_O: float = 1.31
"""O-band center wavelength (μm)."""

WL_E_MIN: float = 1.360
"""E-band minimum wavelength (μm)."""

WL_E_MAX: float = 1.460
"""E-band maximum wavelength (μm)."""

WL_E: float = 1.41
"""E-band center wavelength (μm)."""

WL_S_MIN: float = 1.460
"""S-band minimum wavelength (μm)."""

WL_S_MAX: float = 1.530
"""S-band maximum wavelength (μm)."""

WL_S: float = 1.5
"""S-band center wavelength (μm)."""

WL_C_MIN: float = 1.530
"""C-band minimum wavelength (μm)."""

WL_C_MAX: float = 1.565
"""C-band maximum wavelength (μm)."""

WL_C: float = 1.55
"""C-band center wavelength (μm)."""

WL_L_MIN: float = 1.565
"""L-band minimum wavelength (μm)."""

WL_L_MAX: float = 1.625
"""L-band maximum wavelength (μm)."""

WL_L: float = 1.6
"""L-band center wavelength (μm)."""


def wl_o(
    *,
    step: float = DEFAULT_WL_STEP,
    num: int | None = None,
    wl_min: float = WL_O_MIN,
    wl_max: float = WL_O_MAX,
) -> FloatArray1D:
    """Generate wavelength array in the O-band (Original band: 1260-1360 nm).

    Args:
        step: Wavelength step size in μm. Used when num is None.
        num: Number of wavelength points. If provided, step is ignored.
        wl_min: Minimum wavelength in μm. Defaults to O-band minimum.
        wl_max: Maximum wavelength in μm. Defaults to O-band maximum.

    Returns:
        1D array of wavelengths in μm.

    Example:
        ```python
        # Generate O-band wavelengths with default step
        wl = wl_o()
        # Generate 100 equally spaced points in O-band
        wl = wl_o(num=100)
        # Custom range with specific step
        wl = wl_o(step=0.001, wl_min=1.27, wl_max=1.35)
        ```
    """
    return _wl(step=step, num=num, wl_min=wl_min, wl_max=wl_max)


def wl_e(
    *,
    step: float = DEFAULT_WL_STEP,
    num: int | None = None,
    wl_min: float = WL_E_MIN,
    wl_max: float = WL_E_MAX,
) -> FloatArray1D:
    """Generate wavelength array in the E-band (Extended band: 1360-1460 nm).

    Args:
        step: Wavelength step size in μm. Used when num is None.
        num: Number of wavelength points. If provided, step is ignored.
        wl_min: Minimum wavelength in μm. Defaults to E-band minimum.
        wl_max: Maximum wavelength in μm. Defaults to E-band maximum.

    Returns:
        1D array of wavelengths in μm.

    Example:
        ```python
        # Generate E-band wavelengths with default step
        wl = wl_e()
        # Generate 50 equally spaced points in E-band
        wl = wl_e(num=50)
        ```
    """
    return _wl(step=step, num=num, wl_min=wl_min, wl_max=wl_max)


def wl_s(
    *,
    step: float = DEFAULT_WL_STEP,
    num: int | None = None,
    wl_min: float = WL_S_MIN,
    wl_max: float = WL_S_MAX,
) -> FloatArray1D:
    """Generate wavelength array in the S-band (Short band: 1460-1530 nm).

    Args:
        step: Wavelength step size in μm. Used when num is None.
        num: Number of wavelength points. If provided, step is ignored.
        wl_min: Minimum wavelength in μm. Defaults to S-band minimum.
        wl_max: Maximum wavelength in μm. Defaults to S-band maximum.

    Returns:
        1D array of wavelengths in μm.

    Example:
        ```python
        # Generate S-band wavelengths with default step
        wl = wl_s()
        # Generate 100 equally spaced points in S-band
        wl = wl_s(num=100)
        ```
    """
    return _wl(step=step, num=num, wl_min=wl_min, wl_max=wl_max)


def wl_c(
    *,
    step: float = DEFAULT_WL_STEP,
    num: int | None = None,
    wl_min: float = WL_C_MIN,
    wl_max: float = WL_C_MAX,
) -> FloatArray1D:
    """Generate wavelength array in the C-band (Conventional band: 1530-1565 nm).

    The C-band is the most commonly used wavelength range in optical communications
    due to the low attenuation characteristics of standard single-mode fiber.

    Args:
        step: Wavelength step size in μm. Used when num is None.
        num: Number of wavelength points. If provided, step is ignored.
        wl_min: Minimum wavelength in μm. Defaults to C-band minimum.
        wl_max: Maximum wavelength in μm. Defaults to C-band maximum.

    Returns:
        1D array of wavelengths in μm.

    Example:
        ```python
        # Generate C-band wavelengths with default step
        wl = wl_c()
        # Generate 100 equally spaced points in C-band
        wl = wl_c(num=100)
        # High resolution C-band scan
        wl = wl_c(step=0.00001)
        ```
    """
    return _wl(step=step, num=num, wl_min=wl_min, wl_max=wl_max)


def wl_l(
    *,
    step: float = DEFAULT_WL_STEP,
    num: int | None = None,
    wl_min: float = WL_L_MIN,
    wl_max: float = WL_L_MAX,
) -> FloatArray1D:
    """Generate wavelength array in the L-band (Long band: 1565-1625 nm).

    Args:
        step: Wavelength step size in μm. Used when num is None.
        num: Number of wavelength points. If provided, step is ignored.
        wl_min: Minimum wavelength in μm. Defaults to L-band minimum.
        wl_max: Maximum wavelength in μm. Defaults to L-band maximum.

    Returns:
        1D array of wavelengths in μm.

    Example:
        ```python
        # Generate L-band wavelengths with default step
        wl = wl_l()
        # Generate 100 equally spaced points in L-band
        wl = wl_l(num=100)
        ```
    """
    return _wl(step=step, num=num, wl_min=wl_min, wl_max=wl_max)


def _wl(
    *,
    step: float,
    num: int | None,
    wl_min: float,
    wl_max: float,
) -> FloatArray1D:
    if num is not None:
        return jnp.linspace(wl_min, wl_max, num)
    return jnp.arange(wl_min, wl_max + step, step)
