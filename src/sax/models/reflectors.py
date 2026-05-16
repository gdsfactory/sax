"""SAX Reflector Models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from pydantic import validate_call

import sax


@jax.jit
@validate_call
def reflector(
    *,
    wl: sax.FloatArrayLike = sax.WL_C,
    reflection: sax.FloatArrayLike = 0.5,
    loss_dB: sax.FloatArrayLike = 0.0,
) -> sax.SDict:
    r"""Partial reflector / mirror model.

    A 2-port component that reflects a fraction of the input power and
    transmits the rest (minus any loss).

    ```{svgbob}
    in0          |   out0
     o1 ========= | ========= o2
                 |
              mirror
    ```

    Args:
        wl: Wavelength in micrometers.
        reflection: Power reflection coefficient between 0 and 1.
            0 means full transmission, 1 means full reflection. Defaults to 0.5.
        loss_dB: Insertion loss in dB applied to both reflected and
            transmitted amplitudes. Defaults to 0.0 dB.

    Returns:
        S-matrix dictionary for the reflector.

    Examples:
        50% reflector:

        ```python
        # mkdocs: render
        import matplotlib.pyplot as plt
        import numpy as np
        import sax

        sax.set_port_naming_strategy("optical")

        wl = sax.wl_c()
        s = sax.models.reflector(wl=wl, reflection=0.5)
        thru = np.abs(s[("o1", "o2")]) ** 2
        refl = np.abs(s[("o1", "o1")]) ** 2
        plt.figure()
        plt.plot(wl, thru, label="transmission")
        plt.plot(wl, refl, label="reflection")
        plt.xlabel("Wavelength [μm]")
        plt.ylabel("Power")
        plt.legend()
        ```
    """
    one = jnp.ones_like(jnp.asarray(wl))
    loss_amp = jnp.asarray(10 ** (-loss_dB / 20), dtype=complex) * one
    r = jnp.asarray(reflection**0.5, dtype=complex) * loss_amp
    t = jnp.asarray((1 - reflection) ** 0.5, dtype=complex) * loss_amp
    p = sax.PortNamer(1, 1)
    return sax.reciprocal(
        {
            (p.in0, p.in0): r,
            (p.in0, p.out0): t,
            (p.out0, p.out0): r,
        }
    )


@jax.jit
@validate_call
def mirror(
    *,
    wl: sax.FloatArrayLike = sax.WL_C,
    reflection: sax.FloatArrayLike = 1.0,
    loss_dB: sax.FloatArrayLike = 0.0,
) -> sax.SDict:
    r"""Ideal mirror model (reflector with default 100% reflection).

    ```{svgbob}
    in0          ||
     o1 ========= ||
                 ||
              mirror
    ```

    Args:
        wl: Wavelength in micrometers.
        reflection: Power reflection coefficient between 0 and 1.
            Defaults to 1.0 (perfect mirror).
        loss_dB: Insertion loss in dB. Defaults to 0.0 dB.

    Returns:
        S-matrix dictionary for the mirror.

    Examples:
        Perfect mirror:

        ```python
        # mkdocs: render
        import matplotlib.pyplot as plt
        import numpy as np
        import sax

        sax.set_port_naming_strategy("optical")

        wl = sax.wl_c()
        s = sax.models.mirror(wl=wl)
        refl = np.abs(s[("o1", "o1")]) ** 2
        plt.figure()
        plt.plot(wl, refl, label="reflection")
        plt.xlabel("Wavelength [μm]")
        plt.ylabel("Power")
        plt.legend()
        ```
    """
    return reflector(wl=wl, reflection=reflection, loss_dB=loss_dB)
