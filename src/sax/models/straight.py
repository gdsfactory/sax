"""SAX Default Models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from pydantic import validate_call

import sax


@jax.jit
@validate_call
def straight(
    *,
    wl: sax.FloatArrayLike = sax.WL_C,
    wl0: sax.FloatArrayLike = sax.WL_C,
    neff: sax.FloatArrayLike = 2.34,
    ng: sax.FloatArrayLike = 3.4,
    length: sax.FloatArrayLike = 10.0,
    loss_dB_cm: sax.FloatArrayLike = 0.0,
) -> sax.SDict:
    """Dispersive straight waveguide model.

    ```{svgbob}
    in0             out0
     o1 =========== o2
    ```

    Args:
        wl: The wavelength in micrometers.
        wl0: The center wavelength used for dispersion calculation.
        neff: The Effective refractive index at the center wavelength.
        ng: The Group refractive index at the center wavelength.
        length: The length of the waveguide in micrometers.
        loss_dB_cm: The Propagation loss in dB/cm.

    Returns:
        S-matrix dictionary containing the complex transmission coefficient.

    Examples:
        Lossless waveguide:

        ```python
        # mkdocs: render
        import matplotlib.pyplot as plt
        import numpy as np
        import sax

        sax.set_port_naming_strategy("optical")

        wl = sax.wl_c()
        s = sax.models.straight(wl=wl, coupling=0.3)
        thru = np.abs(s[("o1", "o2")]) ** 2
        plt.figure()
        plt.plot(wl, thru, label="thru")
        plt.xlabel("Wavelength [μm]")
        plt.ylabel("Power")
        plt.legend()
        ```
    """
    dwl: sax.FloatArray = sax.into[sax.FloatArray](wl) - wl0
    dneff_dwl = (ng - neff) / wl0
    _neff = neff - dwl * dneff_dwl
    phase = 2 * jnp.pi * _neff * length / wl
    amplitude = jnp.asarray(10 ** (-1e-4 * loss_dB_cm * length / 20), dtype=complex)
    transmission = amplitude * jnp.exp(1j * phase)
    p = sax.PortNamer(1, 1)
    return sax.reciprocal(
        {
            (p.in0, p.out0): transmission,
        },
    )


@jax.jit
@validate_call
def attenuator(
    *,
    wl: sax.FloatArrayLike = sax.WL_C,
    loss: sax.FloatArrayLike = 0.0,
) -> sax.SDict:
    """Simple optical attenuator model.

    ```{svgbob}
    in0             out0
     o1 =========== o2
    ```

    Args:
        wl: The wavelength in micrometers.
        loss: Attenuation in decibels (dB).

    Returns:
        S-matrix dictionary containing the complex transmission coefficient.

    Examples:
        Attenuator:

        ```python
        # mkdocs: render
        import matplotlib.pyplot as plt
        import numpy as np
        import sax

        sax.set_port_naming_strategy("optical")

        wl = sax.wl_c()
        s = sax.models.attenuator(wl=wl, loss=3.0)
        thru = np.abs(s[("o1", "o2")]) ** 2
        plt.figure()
        plt.plot(wl, thru, label="thru")
        plt.xlabel("Wavelength [μm]")
        plt.ylabel("Power")
        plt.legend()
        ```
    """
    one = jnp.ones_like(jnp.asarray(wl))
    transmission = jnp.asarray(10 ** (-loss / 20), dtype=complex) * one
    p = sax.PortNamer(1, 1)
    return sax.reciprocal(
        {
            (p.in0, p.out0): transmission,
        }
    )


@jax.jit
@validate_call
def phase_shifter(
    wl: sax.FloatArrayLike = sax.WL_C,
    neff: sax.FloatArrayLike = 2.34,
    voltage: sax.FloatArrayLike = 0,
    length: sax.FloatArrayLike = 10,
    loss: sax.FloatArrayLike = 0.0,
) -> sax.SDict:
    """Simple voltage-controlled phase shifter model.

    ```{svgbob}
    in0             out0
     o1 =========== o2
    ```

    Args:
        wl: The wavelength in micrometers.
        neff: The Effective index of the unperturbed waveguide mode.
        voltage: The Applied voltage in volts. The phase shift is assumed to be
            linearly proportional to voltage with a coefficient of π rad/V.
            Positive voltage increases the phase. Defaults to 0 V.
        length: The length of the phase shifter in micrometers.
        loss: Additional loss in dB introduced by the active region.

    Returns:
        S-matrix dictionary containing the complex-valued transmission coefficient.

    Examples:
        Phase shifter:

        ```python
        # mkdocs: render
        import matplotlib.pyplot as plt
        import numpy as np
        import sax

        sax.set_port_naming_strategy("optical")

        wl = sax.wl_c()
        s = sax.models.phase_shifter(wl=wl, loss=3.0)
        thru = np.abs(s[("o1", "o2")]) ** 2
        plt.figure()
        plt.plot(wl, thru, label="thru")
        plt.xlabel("Wavelength [μm]")
        plt.ylabel("Power")
        plt.legend()
        ```
    """
    deltaphi = voltage * jnp.pi
    phase = 2 * jnp.pi * neff * length / wl + deltaphi
    amplitude = jnp.asarray(10 ** (-loss * length / 20), dtype=complex)
    transmission = amplitude * jnp.exp(1j * phase)
    p = sax.PortNamer(1, 1)
    return sax.reciprocal(
        {
            (p.o1, p.o2): transmission,
        }
    )
