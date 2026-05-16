"""SAX Terminator Models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from pydantic import validate_call

import sax


@jax.jit
@validate_call
def terminator(
    *,
    wl: sax.FloatArrayLike = sax.WL_C,
    reflection: sax.FloatArrayLike = 0.0,
) -> sax.SDict:
    """Optical terminator / absorber model.

    A 1-port device that absorbs all incoming light with minimal reflection.
    Used to terminate unused ports and prevent back-reflections.

    ```{svgbob}
    in0
     o1 =========X
    ```

    Args:
        wl: Wavelength in micrometers.
        reflection: Residual power reflection coefficient between 0 and 1.
            Defaults to 0.0 (perfect absorber).

    Returns:
        S-matrix dictionary for the terminator.

    Examples:
        Ideal terminator:

        ```python
        # mkdocs: render
        import matplotlib.pyplot as plt
        import numpy as np
        import sax

        sax.set_port_naming_strategy("optical")

        wl = sax.wl_c()
        s = sax.models.terminator(wl=wl, reflection=0.01)
        refl = np.abs(s[("o1", "o1")]) ** 2
        plt.figure()
        plt.plot(wl, refl, label="reflection")
        plt.xlabel("Wavelength [μm]")
        plt.ylabel("Power")
        plt.legend()
        ```
    """
    one = jnp.ones_like(jnp.asarray(wl))
    r = jnp.asarray(reflection**0.5, dtype=complex) * one
    return {
        ("o1", "o1"): r,
    }
