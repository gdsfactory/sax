"""SAX Probe Models."""

from __future__ import annotations

import jax.numpy as jnp
from pydantic import validate_call

import sax

__all__ = ["ideal_probe"]


@validate_call
def ideal_probe(wl: sax.FloatArrayLike = sax.WL_C) -> sax.SDict:
    """Ideal 4-port measurement probe with 100% transmission and 100% tap coupling.

    This is an unphysical component designed for debugging and measurement purposes.
    It intercepts a connection and provides access to both forward and backward
    traveling waves without affecting the signal propagation.

    ```
              in ─────────────────── out
               │                     │
               │     (ideal tap)     │
               │                     │
             tap_bwd               tap_fwd
    ```

    The probe has the following behavior:
    - Full transmission from `in` to `out` (and vice versa)
    - Forward tap (`tap_fwd`) copies the signal traveling from `in` toward `out`
    - Backward tap (`tap_bwd`) copies the signal traveling from `out` toward `in`
    - No reflections at any port
    - No cross-coupling between tap ports

    Note:
        This S-matrix is NOT unitary (violates energy conservation). This is
        intentional—it's a measurement tool, not a physical device.

    Args:
        wl: Wavelength in micrometers. Defaults to 1.55 μm.

    Returns:
        S-parameter dictionary for the ideal probe.

    Example:
        Probes are typically not used directly, but via the `probes` argument
        to `sax.circuit()`:

        ```python
        circuit_fn, info = sax.circuit(
            netlist,
            models=models,
            probes={"mid": "wg1,out"},
        )
        # Circuit now has additional ports: mid_fwd, mid_bwd
        ```
    """
    one = jnp.ones_like(jnp.asarray(wl))
    zero = jnp.zeros_like(jnp.asarray(wl))
    return {
        # Through path: full transmission
        ("in", "out"): one,
        ("out", "in"): one,
        # Forward tap: copies signal from in→out direction
        ("in", "tap_fwd"): one,
        ("tap_fwd", "in"): one,
        # Backward tap: copies signal from out→in direction
        ("out", "tap_bwd"): one,
        ("tap_bwd", "out"): one,
        # No cross-coupling between taps
        ("tap_fwd", "tap_bwd"): zero,
        ("tap_bwd", "tap_fwd"): zero,
        # No coupling from taps to opposite main port
        ("tap_fwd", "out"): zero,
        ("tap_bwd", "in"): zero,
        # No reflections
        ("in", "in"): zero,
        ("out", "out"): zero,
        ("tap_fwd", "tap_fwd"): zero,
        ("tap_bwd", "tap_bwd"): zero,
    }
