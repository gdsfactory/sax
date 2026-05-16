"""SAX Isolator and Circulator Models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from pydantic import validate_call

import sax


@jax.jit
@validate_call
def isolator(
    *,
    wl: sax.FloatArrayLike = sax.WL_C,
    insertion_loss_dB: sax.FloatArrayLike = 0.0,
    isolation_dB: sax.FloatArrayLike = 40.0,
) -> sax.SDict:
    """Optical isolator model (non-reciprocal).

    Transmits light in the forward direction (in0 -> out0) with low loss
    while blocking the reverse direction (out0 -> in0).

    ```{svgbob}
    in0             out0
     o1 ====>====== o2
    ```

    Args:
        wl: Wavelength in micrometers.
        insertion_loss_dB: Forward insertion loss in dB. Defaults to 0.0 dB.
        isolation_dB: Reverse isolation in dB. Higher values mean better
            blocking of backward-propagating light. Defaults to 40.0 dB.

    Returns:
        S-matrix dictionary for the isolator.

    Examples:
        Isolator with 1 dB insertion loss and 30 dB isolation:

        ```python
        # mkdocs: render
        import matplotlib.pyplot as plt
        import numpy as np
        import sax

        sax.set_port_naming_strategy("optical")

        wl = sax.wl_c()
        s = sax.models.isolator(
            wl=wl,
            insertion_loss_dB=1.0,
            isolation_dB=30.0,
        )
        fwd = np.abs(s[("o1", "o2")]) ** 2
        bwd = np.abs(s[("o2", "o1")]) ** 2
        plt.figure()
        plt.plot(wl, 10 * np.log10(fwd), label="forward")
        plt.plot(wl, 10 * np.log10(bwd + 1e-10), label="backward")
        plt.xlabel("Wavelength [μm]")
        plt.ylabel("Transmission [dB]")
        plt.legend()
        ```
    """
    one = jnp.ones_like(jnp.asarray(wl))
    forward = jnp.asarray(10 ** (-insertion_loss_dB / 20), dtype=complex) * one
    backward = jnp.asarray(10 ** (-isolation_dB / 20), dtype=complex) * one
    return {
        ("o1", "o2"): forward,
        ("o2", "o1"): backward,
    }


@jax.jit
@validate_call
def circulator(
    *,
    wl: sax.FloatArrayLike = sax.WL_C,
    insertion_loss_dB: sax.FloatArrayLike = 0.0,
    isolation_dB: sax.FloatArrayLike = 40.0,
) -> sax.SDict:
    r"""Optical circulator model (non-reciprocal 3-port device).

    Routes light in a circular fashion: port 1 -> port 2 -> port 3 -> port 1.
    Light traveling in the reverse direction is strongly attenuated.

    ```{svgbob}
          o2
          *
         / \\
        /   \\
       / --> \\
      *-------*
     o1       o3
    ```

    Args:
        wl: Wavelength in micrometers.
        insertion_loss_dB: Insertion loss in dB for the forward circulation
            path. Defaults to 0.0 dB.
        isolation_dB: Isolation in dB between non-adjacent ports (reverse
            direction). Defaults to 40.0 dB.

    Returns:
        S-matrix dictionary for the circulator.

    Examples:
        3-port circulator:

        ```python
        # mkdocs: render
        import matplotlib.pyplot as plt
        import numpy as np
        import sax

        sax.set_port_naming_strategy("optical")

        wl = sax.wl_c()
        s = sax.models.circulator(
            wl=wl,
            insertion_loss_dB=0.5,
            isolation_dB=30.0,
        )
        fwd_12 = np.abs(s[("o1", "o2")]) ** 2
        fwd_23 = np.abs(s[("o2", "o3")]) ** 2
        iso_21 = np.abs(s[("o2", "o1")]) ** 2
        plt.figure()
        plt.plot(wl, 10 * np.log10(fwd_12), label="o1→o2")
        plt.plot(wl, 10 * np.log10(fwd_23), label="o2→o3")
        plt.plot(wl, 10 * np.log10(iso_21 + 1e-10), label="o2→o1 (isolated)")
        plt.xlabel("Wavelength [μm]")
        plt.ylabel("Transmission [dB]")
        plt.legend()
        ```
    """
    one = jnp.ones_like(jnp.asarray(wl))
    forward = jnp.asarray(10 ** (-insertion_loss_dB / 20), dtype=complex) * one
    isolated = jnp.asarray(10 ** (-isolation_dB / 20), dtype=complex) * one
    return {
        # Forward circulation: o1 -> o2 -> o3 -> o1
        ("o1", "o2"): forward,
        ("o2", "o3"): forward,
        ("o3", "o1"): forward,
        # Reverse (isolated): o2 -> o1, o3 -> o2, o1 -> o3
        ("o2", "o1"): isolated,
        ("o3", "o2"): isolated,
        ("o1", "o3"): isolated,
    }
