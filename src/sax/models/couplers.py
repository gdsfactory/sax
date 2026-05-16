"""SAX Coupler Models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from pydantic import validate_call

import sax


@jax.jit
@validate_call
def coupler_ideal(
    *,
    wl: sax.FloatArrayLike = sax.WL_C,
    coupling: sax.FloatArrayLike = 0.5,
) -> sax.SDict:
    r"""Ideal 2x2 directional coupler model.

    ```{svgbob}
     in1          out1
      o2          o3
       *          *
        \        /
         '------'
         coupling
         .------.
        /        \
       *          *
      o1          o4
     in0          out0
    ```

    Args:
        wl: the wavelength of the simulation in micrometers.
        coupling: Power coupling coefficient between 0 and 1. Defaults to 0.5.

    Returns:
        The coupler s-matrix

    Examples:
        Ideal coupler:

        ```python
        # mkdocs: render
        import matplotlib.pyplot as plt
        import numpy as np
        import sax

        sax.set_port_naming_strategy("optical")

        wl = sax.wl_c()
        s = sax.models.coupler_ideal(
            wl=wl,
            coupling=0.3,
        )
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
    one = jnp.ones_like(wl)
    kappa = jnp.asarray(coupling**0.5) * one
    tau = jnp.asarray((1 - coupling) ** 0.5) * one
    p = sax.PortNamer(2, 2)
    return sax.reciprocal(
        {
            (p.in0, p.out0): tau,
            (p.in0, p.out1): 1j * kappa,
            (p.in1, p.out0): 1j * kappa,
            (p.in1, p.out1): tau,
        },
    )


@jax.jit
@validate_call
def coupler(
    *,
    wl: sax.FloatArrayLike = sax.WL_C,
    wl0: sax.FloatArrayLike = sax.WL_C,
    length: sax.FloatArrayLike = 0.0,
    coupling0: sax.FloatArrayLike = 0.2,
    dk1: sax.FloatArrayLike = 1.2435,
    dk2: sax.FloatArrayLike = 5.3022,
    dn: sax.FloatArrayLike = 0.02,
    dn1: sax.FloatArrayLike = 0.1169,
    dn2: sax.FloatArrayLike = 0.4821,
) -> sax.SDict:
    r"""Dispersive directional coupler model.

    ```{svgbob}
            in1                out1
             o2                o3
              *                *
               \              /
                '------------'
    coupling0 /2   coupling   coupling0 /2
                .------------.
               /  <-length -> \
              *                *
             o1                o4
            in0                out0
    ```

    Args:
        wl: Operating wavelength in micrometers. Can be a scalar or array for
            multi-wavelength simulations. Defaults to 1.55 μm.
        wl0: Reference wavelength in micrometers for dispersion expansion.
            Typically the center design wavelength. Defaults to 1.55 μm.
        length: Coupling length in micrometers. This is the length over which
            the two waveguides are in close proximity. Defaults to 0.0 μm.
        coupling0: Base coupling coefficient at the reference wavelength from
            bend regions or other coupling mechanisms. Obtained from FDTD
            simulations or measurements. Defaults to 0.2.
        dk1: First-order derivative of coupling coefficient with respect to
            wavelength (∂κ/∂λ). Units: μm⁻¹. Defaults to 1.2435.
        dk2: Second-order derivative of coupling coefficient with respect to
            wavelength (∂²κ/∂λ²). Units: μm⁻². Defaults to 5.3022.
        dn: Effective index difference between even and odd supermodes at
            the reference wavelength. Determines beating length. Defaults to 0.02.
        dn1: First-order derivative of effective index difference with respect
            to wavelength (∂Δn/∂λ). Units: μm⁻¹. Defaults to 0.1169.
        dn2: Second-order derivative of effective index difference with respect
            to wavelength (∂²Δn/∂λ²). Units: μm⁻². Defaults to 0.4821.

    Returns:
        The coupler s-matrix

    Examples:
        Basic dispersive coupler:

    ```python
    # mkdocs: render
    import matplotlib.pyplot as plt
    import numpy as np
    import sax

    sax.set_port_naming_strategy("optical")

    wl = sax.wl_c()
    s = sax.models.coupler(
        wl=wl,
        length=15.7,
        coupling0=0.0,
        dn=0.02,
    )
    thru = np.abs(s[("o1", "o4")]) ** 2
    cross = np.abs(s[("o1", "o3")]) ** 2
    plt.figure()
    plt.plot(wl, thru, label="thru")
    plt.plot(wl, cross, label="cross")
    plt.xlabel("Wavelength [μm]")
    plt.ylabel("Power")
    plt.legend()
    ```

    Note:
        The coupling strength follows the formula:

        κ_total = κ₀ + κ₁ + κ_length

        Where:

        - κ₀ = coupling0 + dk1*(λ-λ₀) + 0.5*dk2*(λ-λ₀)²
        - κ₁ = π*Δn(λ)/λ
        - κ_length represents the distributed coupling over the interaction length

    """
    dwl = wl - wl0
    dn = dn + dn1 * dwl + 0.5 * dn2 * dwl**2
    kappa0 = coupling0 + dk1 * dwl + 0.5 * dk2 * dwl**2
    kappa1 = jnp.pi * dn / wl

    tau = jnp.cos(kappa0 + kappa1 * length)
    kappa = -jnp.sin(kappa0 + kappa1 * length)
    p = sax.PortNamer(2, 2)
    return sax.reciprocal(
        {
            (p.in0, p.out0): tau,
            (p.in0, p.out1): 1j * kappa,
            (p.in1, p.out0): 1j * kappa,
            (p.in1, p.out1): tau,
        }
    )


@jax.jit
@validate_call
def grating_coupler(
    *,
    wl: sax.FloatArrayLike = sax.WL_C,
    wl0: sax.FloatArrayLike = sax.WL_C,
    loss: sax.FloatArrayLike = 0.0,
    reflection: sax.FloatArrayLike = 0.0,
    reflection_fiber: sax.FloatArrayLike = 0.0,
    bandwidth: sax.FloatArrayLike = 40e-3,
) -> sax.SDict:
    """Grating coupler model for fiber-chip coupling.

    ```{svgbob}
                  out0
                  o2
             /  /  /  /
            /  /  /  /  fiber
           /  /  /  /
           _   _   _
     o1  _| |_| |_| |__
    in0
    ```

    Args:
        wl: Operating wavelength in micrometers. Can be a scalar or array for
            spectral analysis. Defaults to 1.55 μm.
        wl0: Center wavelength in micrometers where peak transmission occurs.
            This is the design wavelength of the grating. Defaults to 1.55 μm.
        loss: Insertion loss in dB at the center wavelength. Includes coupling
            efficiency losses, scattering, and mode mismatch. Defaults to 0.0 dB.
        reflection: Reflection coefficient from the waveguide side (chip interface).
            Represents reflections back into the waveguide from grating discontinuities.
            Range: 0 to 1. Defaults to 0.0.
        reflection_fiber: Reflection coefficient from the fiber side (top interface).
            Represents reflections back toward the fiber from the grating surface.
            Range: 0 to 1. Defaults to 0.0.
        bandwidth: 3dB bandwidth in micrometers. Determines the spectral width
            of the Gaussian transmission profile. Typical values: 20-50 nm.
            Defaults to 40e-3 μm (40 nm).

    Returns:
        The grating coupler s-matrix

    Examples:
        Basic grating coupler:

    ```python
    # mkdocs: render
    import matplotlib.pyplot as plt
    import numpy as np
    import sax

    sax.set_port_naming_strategy("optical")

    wl = np.linspace(1.5, 1.6, 101)
    s = sax.models.grating_coupler(
        wl=wl,
        loss=3.0,
        bandwidth=0.035,
    )
    plt.figure()
    plt.plot(wl, np.abs(s[("o1", "o2")]) ** 2, label="Transmission")
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Power")
    plt.legend()
    ```

    Note:
        The transmission power profile follows a Gaussian shape:
        P(λ) = P₀ * exp(-((λ-λ₀)/σ)²)

        The amplitude transmission is:
        A(λ) = A₀ * exp(-((λ-λ₀)/σ)²/2)

        Where σ = bandwidth / (2*√(2*ln(2))) converts FWHM to Gaussian width.

        This model assumes:

        - Gaussian spectral response (typical for uniform gratings)
        - Wavelength-independent loss coefficient
        - Linear polarization (TE or TM)
        - Single-mode fiber coupling
        - No higher-order diffraction effects

    """
    one = jnp.ones_like(wl)
    reflection = jnp.asarray(reflection) * one
    reflection_fiber = jnp.asarray(reflection_fiber) * one
    amplitude = jnp.asarray(10 ** (-loss / 20)) * one
    wl0_array = jnp.asarray(wl0) * one
    sigma = jnp.asarray(bandwidth / (2 * jnp.sqrt(2 * jnp.log(2)))) * one
    transmission = amplitude * jnp.exp(-((wl - wl0_array) ** 2) / (4 * sigma**2))
    p = sax.PortNamer(1, 1)
    return sax.reciprocal(
        {
            (p.in0, p.in0): reflection,
            (p.in0, p.out0): transmission,
            (p.out0, p.in0): transmission,
            (p.out0, p.out0): reflection_fiber,
        }
    )
