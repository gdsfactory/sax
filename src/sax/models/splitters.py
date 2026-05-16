"""SAX Default Models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from pydantic import validate_call

import sax


@jax.jit
@validate_call
def splitter_ideal(
    *,
    wl: sax.FloatArrayLike = sax.WL_C,
    coupling: sax.FloatArrayLike = 0.5,
) -> sax.SDict:
    r"""Ideal 1x2 power splitter model.

    ```{svgbob}
                    .--------- out0
                   /           o2
                  /   .-------
    in0 ---------'   /
     o1             (
        ---------.   \
                  \   '------- out1
                   \           o3
                    '---------
    ```

    Args:
        wl: Wavelength in micrometers.
        coupling: Power coupling ratio between 0 and 1.

    Returns:
        S-matrix dictionary containing the complex-valued cross/thru coefficients.

    Examples:
        Ideal 1x2 splitter:

        ```python
        # mkdocs: render
        import matplotlib.pyplot as plt
        import numpy as np
        import sax

        sax.set_port_naming_strategy("optical")

        wl = sax.wl_c()
        s = sax.models.splitter_ideal(wl=wl, coupling=0.3)
        thru = np.abs(s[("o1", "o3")]) ** 2
        cross = np.abs(s[("o1", "o2")]) ** 2
        plt.figure()
        plt.plot(wl, thru, label="thru")
        plt.plot(wl, cross, label="cross")
        plt.xlabel("Wavelength [μm]")
        plt.ylabel("Power")
        plt.legend()
        ```
    """
    one = jnp.ones_like(jnp.asarray(wl))
    kappa = jnp.asarray(coupling**0.5) * one
    tau = jnp.asarray((1 - coupling) ** 0.5) * one

    p = sax.PortNamer(num_inputs=1, num_outputs=2)
    return sax.reciprocal(
        {
            (p.in0, p.out0): tau,
            (p.in0, p.out1): kappa,
        },
    )
