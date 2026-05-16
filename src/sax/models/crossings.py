"""SAX Crossing Models."""

import jax
import jax.numpy as jnp
from pydantic import validate_call

import sax


@jax.jit
@validate_call
def crossing_ideal(wl: sax.FloatArrayLike = sax.WL_C) -> sax.SDict:
    """Ideal waveguide crossing model.

    ```{svgbob}
            in1
             o2
              *
              |
              |
     o1 *-----+-----* o3
    in0       |     out0
              |
              *
             o4
           out1
    ```

    Args:
        wl: Wavelength in micrometers.

    Returns:
        The crossing s-matrix

    Examples:
        Ideal crossing:

    ```python
    # mkdocs: render
    import matplotlib.pyplot as plt
    import numpy as np
    import jax.numpy as jnp
    import sax

    sax.set_port_naming_strategy("optical")

    wl = sax.wl_c()
    s = sax.models.crossing_ideal(wl=wl)
    thru = np.abs(s[("o1", "o3")]) ** 2
    cross = np.abs(s.get(("o1", "o2"), jnp.zeros_like(wl))) ** 2
    plt.figure()
    plt.plot(wl, thru, label="thru")
    plt.plot(wl, cross, label="cross")
    plt.xlabel("Wavelength [μm]")
    plt.ylabel("Power")
    plt.legend()
    ```
    """
    one = jnp.ones_like(jnp.asarray(wl))
    p = sax.PortNamer(2, 2)
    return sax.reciprocal(
        {
            (p.o1, p.o3): one,
            (p.o2, p.o4): one,
        }
    )
