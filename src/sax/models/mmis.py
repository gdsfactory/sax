"""SAX MMI Models."""

import jax
import jax.numpy as jnp
from pydantic import validate_call

import sax

from .couplers import coupler_ideal
from .splitters import splitter_ideal


@jax.jit
@validate_call
def mmi1x2_ideal(wl: sax.FloatArrayLike = sax.WL_C) -> sax.SDict:
    """Ideal 1x2 multimode interference (MMI) splitter model.

    ```{svgbob}
            +-------------+     out1
            |             |---* o2
     o1 *---|             |
    in0     |             |---* o3
            +-------------+     out0
    ```

    Args:
        wl: The wavelength in micrometers.

    Returns:
        S-matrix dictionary representing the ideal MMI splitter behavior.

    Examples:
        Ideal 1x2 MMI:

        ```python
        # mkdocs: render
        import matplotlib.pyplot as plt
        import numpy as np
        import sax

        sax.set_port_naming_strategy("optical")

        wl = sax.wl_c()
        s = sax.models.mmi1x2_ideal(wl=wl)
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
    return splitter_ideal(wl=wl, coupling=0.5)


@jax.jit
@validate_call
def mmi2x2_ideal(*, wl: sax.FloatArrayLike = sax.WL_C) -> sax.SDict:
    """Ideal 2x2 multimode interference (MMI) coupler model.

    ```{svgbob}
    in1     +-------------+     out1
     o2 *---|             |---* o3
            |             |
     o1 *---|             |---* o4
    in0     +-------------+     out0
    ```

    Args:
        wl: The wavelength in micrometers.

    Returns:
        S-matrix dictionary representing the ideal MMI coupler behavior.

    Examples:
        Ideal 2x2 MMI:

        ```python
        # mkdocs: render
        import matplotlib.pyplot as plt
        import numpy as np
        import sax

        sax.set_port_naming_strategy("optical")

        wl = sax.wl_c()
        s = sax.models.mmi2x2_ideal(wl=wl)
        thru = np.abs(s[("o1", "o4")]) ** 2
        cross = np.abs(s[("o1", "o3")]) ** 2
        plt.figure()
        plt.plot(wl, thru, label="thru")
        plt.plot(wl, cross, label="cross")
        plt.xlabel("Wavelength [μm]")
        plt.ylabel("Power")
        plt.legend()
        ```
    """
    return coupler_ideal(wl=wl, coupling=0.5)


@jax.jit
@validate_call
def mmi1x2(
    wl: sax.FloatArrayLike = sax.WL_C,
    wl0: sax.FloatArrayLike = sax.WL_C,
    fwhm: sax.FloatArrayLike = 0.2,
    loss_dB: sax.FloatArrayLike = 0.3,
) -> sax.SDict:
    r"""Realistic 1x2 MMI splitter model with dispersion and loss.

    ```{svgbob}
            +-------------+     out1
            |             |---* o2
     o1 *---|             |
    in0     |             |---* o3
            +-------------+     out0
    ```

    Args:
        wl: The wavelength in micrometers.
        wl0: The Center wavelength of the MMI
        fwhm: The Full width at half maximum or the MMI.
        loss_dB: Insertion loss in dB at the center wavelength.

    Returns:
        S-matrix dictionary representing the dispersive MMI splitter behavior.

    Examples:
        Basic 1x2 MMI:

        ```python
        # mkdocs: render
        import matplotlib.pyplot as plt
        import numpy as np
        import sax

        sax.set_port_naming_strategy("optical")

        wl = sax.wl_c()
        s = sax.models.mmi1x2(wl=wl)
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
    thru = _mmi_amp(wl=wl, wl0=wl0, fwhm=fwhm, loss_dB=loss_dB) / 2**0.5

    p = sax.PortNamer(1, 2)
    return sax.reciprocal(
        {
            (p.o1, p.o2): thru,
            (p.o1, p.o3): thru,
        }
    )


@jax.jit
@validate_call
def mmi2x2(
    wl: sax.FloatArrayLike = sax.WL_C,
    wl0: sax.FloatArrayLike = sax.WL_C,
    fwhm: sax.FloatArrayLike = 0.2,
    loss_dB: sax.FloatArrayLike = 0.3,
    shift: sax.FloatArrayLike = 0.0,
    loss_dB_cross: sax.FloatArrayLike | None = None,
    loss_dB_thru: sax.FloatArrayLike | None = None,
    splitting_ratio_cross: sax.FloatArrayLike = 0.5,
    splitting_ratio_thru: sax.FloatArrayLike = 0.5,
) -> sax.SDict:
    r"""Realistic 2x2 MMI coupler model with dispersion and asymmetry.

    ```{svgbob}
    in1     +-------------+     out1
     o2 *---|             |---* o3
            |             |
     o1 *---|             |---* o4
    in0     +-------------+     out0
    ```

    Args:
        wl: wavelength in micrometers.
        wl0: Center wavelength of the MMI in micrometers.
        fwhm: Full width at half maximum bandwidth in micrometers.
        loss_dB: Insertion loss of the MMI at center wavelength.
        shift: The peak shift of the cross-transmission in micrometers.
        loss_dB_cross: Optional separate insertion loss in dB for cross ports.
            If None, uses loss_dB. Allows modeling of asymmetric loss.
        loss_dB_thru: Optional separate insertion loss in dB for bar (through)
            ports. If None, uses loss_dB. Allows modeling of asymmetric loss.
        splitting_ratio_cross: Power splitting ratio for cross ports (0 to 1).
            Allows modeling of imbalanced coupling. Defaults to 0.5.
        splitting_ratio_thru: Power splitting ratio for bar ports (0 to 1).
            Allows modeling of imbalanced transmission. Defaults to 0.5.

    Returns:
        S-matrix dictionary representing the realistic MMI coupler behavior.

    Examples:
        Basic 2x2 MMI:

        ```python
        # mkdocs: render
        import matplotlib.pyplot as plt
        import numpy as np
        import sax

        sax.set_port_naming_strategy("optical")

        wl = sax.wl_c()
        s = sax.models.mmi2x2(wl=wl, shift=0.001)
        thru = np.abs(s[("o1", "o4")]) ** 2
        cross = np.abs(s[("o1", "o3")]) ** 2
        plt.figure()
        plt.plot(wl, thru, label="thru")
        plt.plot(wl, cross, label="cross")
        plt.xlabel("Wavelength [μm]")
        plt.ylabel("Power")
        plt.legend()
        ```
    ```
    """
    loss_dB_cross = loss_dB_cross or loss_dB
    loss_dB_thru = loss_dB_thru or loss_dB

    # Convert splitting ratios from power to amplitude by taking the square root
    amplitude_ratio_thru = splitting_ratio_thru**0.5
    amplitude_ratio_cross = splitting_ratio_cross**0.5

    # _mmi_amp already includes the loss, so we don't need to apply it again
    thru = (
        _mmi_amp(wl=wl, wl0=wl0, fwhm=fwhm, loss_dB=loss_dB_thru) * amplitude_ratio_thru
    )
    cross = (
        1j
        * _mmi_amp(wl=wl, wl0=wl0 + shift, fwhm=fwhm, loss_dB=loss_dB_cross)
        * amplitude_ratio_cross
    )

    p = sax.PortNamer(2, 2)
    return sax.reciprocal(
        {
            (p.o1, p.o3): thru,
            (p.o1, p.o4): cross,
            (p.o2, p.o3): cross,
            (p.o2, p.o4): thru,
        }
    )


def _mmi_amp(
    wl: sax.FloatArrayLike = sax.WL_C,
    wl0: sax.FloatArrayLike = sax.WL_C,
    fwhm: sax.FloatArrayLike = 0.2,
    loss_dB: sax.FloatArrayLike = 0.3,
) -> sax.FloatArray:
    max_amplitude = 10 ** (-abs(loss_dB) / 20)
    f = 1 / wl
    f0 = 1 / wl0
    f1 = 1 / (wl0 + fwhm / 2)
    f2 = 1 / (wl0 - fwhm / 2)
    _fwhm = f2 - f1

    sigma = _fwhm / (2 * jnp.sqrt(2 * jnp.log(2)))
    # Gaussian response in frequency domain
    spectral_response = jnp.exp(-((f - f0) ** 2) / (2 * sigma**2))
    # Apply loss to amplitude, not power
    amplitude = max_amplitude * spectral_response / spectral_response.max()
    return jnp.asarray(amplitude)


def _mmi_nxn(
    n: int,
    wl: sax.FloatArrayLike = sax.WL_C,
    wl0: sax.FloatArrayLike = sax.WL_C,
    fwhm: sax.FloatArrayLike = 0.2,
    loss_dB: sax.FloatArrayLike | None = None,
    shift: sax.FloatArrayLike | None = None,
    splitting_matrix: sax.FloatArray2D | None = None,
) -> sax.SDict:
    _loss_dB = jnp.zeros(n) if loss_dB is None else jnp.asarray(loss_dB)
    _shift = jnp.zeros(n) if shift is None else jnp.asarray(shift)
    _splitting_matrix = (
        jnp.full((n, n), 1 / n)
        if splitting_matrix is None
        else jnp.asarray(splitting_matrix)
    )

    S = {}
    p = sax.PortNamer(n, n)
    for i in range(n):
        for j in range(n):
            amplitude = _mmi_amp(wl, wl0 + _shift[j], fwhm, _loss_dB[j])
            amplitude *= jnp.sqrt(_splitting_matrix[i][j])
            # _mmi_amp already includes the loss, so no additional loss factor needed
            S[(p[i], p[n + j])] = amplitude

    return sax.reciprocal(S)
