r"""Sax generic RF models.

This module provides generic RF components and JAX-jittable functions for
computing the characteristic impedance, effective permittivity, and
propagation constant of coplanar waveguides and microstrip lines.

Note:
    Some models in this module require the ``jaxellip`` package.
    Install the ``rf`` extra to include it: ``pip install sax[rf]``.
"""

import typing
from functools import partial

import jax
import jax.numpy as jnp

import sax
from sax.constants import C_M_S, DEFAULT_FREQUENCY

try:
    import jaxellip  # pyright: ignore[reportMissingImports]
except ImportError:
    jaxellip = None


@partial(jax.jit, inline=True, static_argnames=("n_ports"))
def gamma_0_load(
    *,
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    gamma_0: sax.Complex = 0,
    n_ports: int = 1,
) -> sax.SType:
    r"""Connection with given reflection coefficient.

    Args:
        f: Array of frequency points in Hz
        gamma_0: Reflection coefficient Γ₀ of connection
        n_ports: Number of ports in component. The diagonal ports of the matrix
            are set to Γ₀ and the off-diagonal ports to 0.

    Returns:
        sax.SType: S-parameters dictionary where $S = \Gamma_0I_\text{n_ports}$

    Examples:
        ```python
        # mkdocs: render
        import matplotlib.pyplot as plt
        import numpy as np
        import sax
        import jaxellip  # pyright: ignore[reportMissingImports]
        import scipy.constants


        sax.set_port_naming_strategy("optical")

        f = np.linspace(1e9, 10e9, 500)
        gamma_0 = 0.5 * np.exp(1j * np.pi / 4)
        s = sax.models.rf.gamma_0_load(f=f, gamma_0=gamma_0, n_ports=2)
        plt.figure()
        plt.plot(f / 1e9, np.abs(s[("o1", "o1")]), label="|S11|")
        plt.plot(f / 1e9, np.abs(s[("o2", "o2")]), label="|S22|")
        plt.plot(f / 1e9, np.abs(s[("o1", "o2")]), label="|S12|")
        plt.plot(f / 1e9, np.abs(s[("o2", "o1")]), label="|S21|")
        plt.xlabel("Frequency [GHz]")
        plt.ylabel("Magnitude")
        plt.legend()
        ```
    """
    f = jnp.asarray(f)
    f_flat = f.ravel()
    sdict = {
        (f"o{i}", f"o{i}"): jnp.full(f_flat.shape[0], gamma_0)
        for i in range(1, n_ports + 1)
    }
    sdict |= {
        (f"o{i}", f"o{j}"): jnp.zeros(f_flat.shape[0], dtype=complex)
        for i in range(1, n_ports + 1)
        for j in range(i + 1, n_ports + 1)
    }
    return sax.reciprocal({k: v.reshape(*f.shape) for k, v in sdict.items()})


@partial(jax.jit, inline=True)
def tee(*, f: sax.FloatArrayLike = DEFAULT_FREQUENCY) -> sax.SDict:
    """Ideal three-port RF power divider/combiner (T-junction).

    ```{svgbob}
            o2
            *
            |
            |
     o1 *---+---* o3
    ```

    Args:
        f: Array of frequency points in Hz

    Returns:
        S-dictionary representing ideal RF T-junction behavior

    Examples:
        ```python
        # mkdocs: render
        import matplotlib.pyplot as plt
        import numpy as np
        import sax
        import jaxellip  # pyright: ignore[reportMissingImports]
        import scipy.constants


        sax.set_port_naming_strategy("optical")

        f = np.linspace(1e9, 10e9, 500)
        s = sax.models.rf.tee(f=f)
        plt.figure()
        plt.plot(f / 1e9, np.abs(s[("o1", "o2")]) ** 2, label="|S12|^2")
        plt.plot(f / 1e9, np.abs(s[("o1", "o3")]) ** 2, label="|S13|^2")
        plt.plot(f / 1e9, np.abs(s[("o2", "o3")]) ** 2, label="|S23|^2")
        plt.xlabel("Frequency [GHz]")
        plt.ylabel("Power")
        plt.legend()
        ```
    """
    f = jnp.asarray(f)
    f_flat = f.ravel()
    sdict = {(f"o{i}", f"o{i}"): jnp.full(f_flat.shape[0], -1 / 3) for i in range(1, 4)}
    sdict |= {
        (f"o{i}", f"o{j}"): jnp.full(f_flat.shape[0], 2 / 3)
        for i in range(1, 4)
        for j in range(i + 1, 4)
    }
    return sax.reciprocal({k: v.reshape(*f.shape) for k, v in sdict.items()})


@partial(jax.jit, inline=True)
def impedance(
    *,
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    z: sax.ComplexLike = 50,
    z0: sax.ComplexLike = 50,
) -> sax.SDict:
    r"""Generalized two-port impedance element.

    Args:
        f: Frequency in Hz
        z: Impedance in Ω
        z0: Reference impedance in Ω.

    Returns:
        S-dictionary representing the impedance element

    References:
        Pozar

    Examples:
        ```python
        # mkdocs: render
        import matplotlib.pyplot as plt
        import numpy as np
        import sax
        import jaxellip  # pyright: ignore[reportMissingImports]
        import scipy.constants


        sax.set_port_naming_strategy("optical")

        f = np.linspace(1e9, 10e9, 500)
        s = sax.models.rf.impedance(f=f, z=75, z0=50)
        plt.figure()
        plt.plot(f / 1e9, np.abs(s[("o1", "o1")]), label="|S11|")
        plt.plot(f / 1e9, np.abs(s[("o1", "o2")]), label="|S12|")
        plt.plot(f / 1e9, np.abs(s[("o2", "o2")]), label="|S22|")
        plt.xlabel("Frequency [GHz]")
        plt.ylabel("Magnitude")
        plt.legend()
        ```
    """
    one = jnp.ones_like(jnp.asarray(f))
    sdict = {
        ("o1", "o1"): z / (z + 2 * z0) * one,
        ("o1", "o2"): 2 * z0 / (2 * z0 + z) * one,
        ("o2", "o2"): z / (z + 2 * z0) * one,
    }
    return sax.reciprocal(sdict)


@partial(jax.jit, inline=True)
def admittance(
    *, f: sax.FloatArrayLike = DEFAULT_FREQUENCY, y: sax.ComplexLike = 1 / 50
) -> sax.SDict:
    r"""Generalized two-port admittance element.

    Args:
        f: Frequency in Hz
        y: Admittance in siemens

    Returns:
        S-dictionary representing the admittance element

    References:
        Pozar

    Examples:
        ```python
        # mkdocs: render
        import matplotlib.pyplot as plt
        import numpy as np
        import sax
        import jaxellip  # pyright: ignore[reportMissingImports]
        import scipy.constants


        sax.set_port_naming_strategy("optical")

        f = np.linspace(1e9, 10e9, 500)
        s = sax.models.rf.admittance(f=f, y=1 / 75)
        plt.figure()
        plt.plot(f / 1e9, np.abs(s[("o1", "o1")]), label="|S11|")
        plt.plot(f / 1e9, np.abs(s[("o1", "o2")]), label="|S12|")
        plt.plot(f / 1e9, np.abs(s[("o2", "o2")]), label="|S22|")
        plt.xlabel("Frequency [GHz]")
        plt.ylabel("Magnitude")
        plt.legend()
        ```
    """
    one = jnp.ones_like(jnp.asarray(f))
    sdict = {
        ("o1", "o1"): 1 / (1 + y) * one,
        ("o1", "o2"): y / (1 + y) * one,
        ("o2", "o2"): 1 / (1 + y) * one,
    }
    return sax.reciprocal(sdict)


@partial(jax.jit, inline=True)
def resistor(
    *,
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    resistance: sax.FloatLike = 50,
    z0: sax.ComplexLike = 50,
) -> sax.SDict:
    r"""Ideal two-port resistor model.

    Args:
        f: Frequency in Hz
        resistance: Resistance in Ohms
        z0: Reference impedance in Ω.

    Returns:
        S-dictionary representing the resistor element

    References:
        [@pozar2012]

    Examples:
        ```python
        # mkdocs: render
        import matplotlib.pyplot as plt
        import numpy as np
        import sax

        sax.set_port_naming_strategy("optical")

        f = np.linspace(1e9, 10e9, 500)
        s = sax.models.rf.resistor(f=f, resistance=100, z0=50)
        plt.figure()
        plt.plot(f / 1e9, np.abs(s[("o1", "o1")]), label="|S11|")
        plt.plot(f / 1e9, np.abs(s[("o1", "o2")]), label="|S12|")
        plt.plot(f / 1e9, np.abs(s[("o2", "o2")]), label="|S22|")
        plt.xlabel("Frequency [GHz]")
        plt.ylabel("Magnitude")
        plt.legend()
        ```
    """
    return impedance(f=f, z=resistance, z0=z0)


@partial(jax.jit, inline=True)
def capacitor(
    *,
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    capacitance: sax.FloatLike = 1e-15,
    z0: sax.ComplexLike = 50,
) -> sax.SDict:
    r"""Ideal two-port capacitor model.

    Args:
        f: Frequency in Hz
        capacitance: Capacitance in Farads
        z0: Reference impedance in Ω.

    Returns:
        S-dictionary representing the capacitor element

    References:
        Pozar

    Examples:
        ```python
        # mkdocs: render
        import matplotlib.pyplot as plt
        import numpy as np
        import sax
        import jaxellip  # pyright: ignore[reportMissingImports]
        import scipy.constants


        sax.set_port_naming_strategy("optical")

        f = np.linspace(1e9, 10e9, 500)
        s = sax.models.rf.capacitor(f=f, capacitance=1e-12, z0=50)
        plt.figure()
        plt.plot(f / 1e9, np.abs(s[("o1", "o1")]), label="|S11|")
        plt.plot(f / 1e9, np.abs(s[("o1", "o2")]), label="|S12|")
        plt.plot(f / 1e9, np.abs(s[("o2", "o2")]), label="|S22|")
        plt.xlabel("Frequency [GHz]")
        plt.ylabel("Magnitude")
        plt.legend()
        ```
    """
    angular_frequency = 2 * jnp.pi * jnp.asarray(f)
    capacitor_impedance = 1 / (1j * angular_frequency * capacitance)
    return impedance(f=f, z=capacitor_impedance, z0=z0)


@partial(jax.jit, inline=True)
def inductor(
    *,
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    inductance: sax.FloatLike = 1e-12,
    z0: sax.ComplexLike = 50,
) -> sax.SDict:
    r"""Ideal two-port inductor model.

    Args:
        f: Frequency in Hz
        inductance: Inductance in Henries
        z0: Reference impedance in Ω.

    Returns:
        S-dictionary representing the inductor element

    References:
        Pozar

    Examples:
        ```python
        # mkdocs: render
        import matplotlib.pyplot as plt
        import numpy as np
        import sax
        import jaxellip  # pyright: ignore[reportMissingImports]
        import scipy.constants


        sax.set_port_naming_strategy("optical")

        f = np.linspace(1e9, 10e9, 500)
        s = sax.models.rf.inductor(f=f, inductance=1e-9, z0=50)
        plt.figure()
        plt.plot(f / 1e9, np.abs(s[("o1", "o1")]), label="|S11|")
        plt.plot(f / 1e9, np.abs(s[("o1", "o2")]), label="|S12|")
        plt.plot(f / 1e9, np.abs(s[("o2", "o2")]), label="|S22|")
        plt.xlabel("Frequency [GHz]")
        plt.ylabel("Magnitude")
        plt.legend()
        ```
    """
    angular_frequency = 2 * jnp.pi * jnp.asarray(f)
    inductor_impedance = 1j * angular_frequency * inductance
    return impedance(f=f, z=inductor_impedance, z0=z0)


@partial(jax.jit, inline=True, static_argnames=("n_ports"))
def electrical_short(
    *,
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    n_ports: int = 1,
) -> sax.SDict:
    r"""Electrical short connection Sax model.

    Args:
        f: Array of frequency points in Hz
        n_ports: Number of ports to set as shorted

    Returns:
        S-dictionary where $S = -I_\text{n_ports}$

    References:
        Pozar
    """
    return gamma_0_load(f=f, gamma_0=-1, n_ports=n_ports)


@partial(jax.jit, inline=True, static_argnames=("n_ports"))
def electrical_open(
    *,
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    n_ports: int = 1,
) -> sax.SDict:
    r"""Electrical open connection Sax model.

    Useful for specifying some ports to remain open while not exposing
    them for connections in circuits.

    Args:
        f: Array of frequency points in Hz
        n_ports: Number of ports to set as opened

    Returns:
        S-dictionary where $S = I_\text{n_ports}$

    References:
        Pozar
    """
    return gamma_0_load(f=f, gamma_0=1, n_ports=n_ports)


@jax.jit
def lc_shunt_component(
    f: sax.FloatArrayLike = 5e9,
    inductance: sax.FloatLike = 1e-9,
    capacitance: sax.FloatLike = 1e-12,
    z0: sax.FloatLike = 50,
) -> sax.SDict:
    """SAX component for a 1-port shunted LC resonator.

    ```{svgbob}
             o1
             *
             |
        +----+----+
        |         |
       --- C      C L
       ---        C
        |         |
        +----+----+
             |
           -----
            ---
             -
    ```
    """
    f = jnp.asarray(f)
    instances = {
        "L": inductor(f=f, inductance=inductance, z0=z0),
        "C": capacitor(f=f, capacitance=capacitance, z0=z0),
        "gnd": electrical_short(f=f, n_ports=1),
        "tee_1": tee(f=f),
        "tee_2": tee(f=f),
    }
    connections = {
        "L,o1": "tee_1,o1",
        "C,o1": "tee_1,o2",
        "L,o2": "tee_2,o1",
        "C,o2": "tee_2,o2",
        "gnd,o1": "tee_2,o3",
    }
    ports = {
        "o1": "tee_1,o3",
    }

    return sax.backends.evaluate_circuit_fg((connections, ports), instances)


@partial(jax.jit, inline=True)
def ellipk_ratio(m: sax.FloatArrayLike) -> jax.Array:
    """Ratio of complete elliptic integrals of the first kind K(m) / K(1-m)."""
    if jaxellip is None:
        msg = (
            "jaxellip is required for RF models. Install it with `pip install sax[rf]`"
        )
        raise ImportError(msg)
    m_arr = jnp.asarray(m, dtype=float)
    return jaxellip.ellipk(m_arr) / jaxellip.ellipk(1 - m_arr)


@partial(jax.jit, inline=True)
def cpw_epsilon_eff(
    w: sax.FloatArrayLike,
    s: sax.FloatArrayLike,
    h: sax.FloatArrayLike,
    ep_r: sax.FloatArrayLike,
) -> jax.Array:
    r"""Effective permittivity of a CPW on a finite-height substrate.

    $$
    \begin{aligned}
        k_0 &= \frac{w}{w + 2s} \\
        k_1 &= \frac{\sinh(\pi w / 4h)}{\sinh\bigl(\pi(w + 2s) / 4h\bigr)} \\
        q_1 &= \frac{K(k_1^2)/K(1 - k_1^2)}{K(k_0^2)/K(1 - k_0^2)}  \\
        \varepsilon_{\mathrm{eff}} &= 1 + \frac{q_1(\varepsilon_r - 1)}{2}
    \end{aligned}
    $$

    where $K$ is the complete elliptic integral of the first kind in
    the *parameter* convention ($m = k^2$).

    References:
        Simoons, Eq. 2.37;
        Ghione & Naldi

    Args:
        w: Centre-conductor width (m).
        s: Gap to ground plane (m).
        h: Substrate height (m).
        ep_r: Relative permittivity of the substrate.

    Returns:
        Effective permittivity (dimensionless).
    """
    w = jnp.asarray(w, dtype=float)
    s = jnp.asarray(s, dtype=float)
    h = jnp.asarray(h, dtype=float)
    ep_r = jnp.asarray(ep_r, dtype=float)
    k0 = w / (w + 2.0 * s)
    k1 = jnp.sinh(jnp.pi * w / (4.0 * h)) / jnp.sinh(jnp.pi * (w + 2.0 * s) / (4.0 * h))
    q1 = ellipk_ratio(k1**2) / ellipk_ratio(k0**2)
    return 1.0 + q1 * (ep_r - 1.0) / 2.0


@partial(jax.jit, inline=True)
def cpw_z0(
    w: sax.FloatArrayLike,
    s: sax.FloatArrayLike,
    ep_eff: sax.FloatArrayLike,
) -> jax.Array:
    r"""Characteristic impedance of a CPW.

    $$
    Z_0 = \frac{30\pi}{\sqrt{\varepsilon_{\mathrm{eff}}} K(k_0^2)/K(1 - k_0^2)}
    $$


    References:
        Simons, Eq. 2.38.
        Note that our $w$ and $s$ correspond to Simons' $s$ and $w$.

    Args:
        w: Centre-conductor width (m).
        s: Gap to ground plane (m).
        ep_eff: Effective permittivity (see :func:`cpw_epsilon_eff`).

    Returns:
        Characteristic impedance (Ω).
    """
    w = jnp.asarray(w, dtype=float)
    s = jnp.asarray(s, dtype=float)
    ep_eff = jnp.asarray(ep_eff, dtype=float)
    k0 = w / (w + 2.0 * s)
    return 30.0 * jnp.pi / (jnp.sqrt(ep_eff) * ellipk_ratio(k0**2))


@partial(jax.jit, inline=True)
def cpw_thickness_correction(
    w: sax.FloatArrayLike,
    s: sax.FloatArrayLike,
    t: sax.FloatArrayLike,
    ep_eff: sax.FloatArrayLike,
) -> tuple[typing.Any, typing.Any]:
    r"""Apply conductor thickness correction to CPW ε_eff and Z₀.

    First-order correction from Gupta, Garg, Bahl & Bhartia


    $$
    \begin{aligned}
        \Delta &= \frac{1.25\,t}{\pi}
    \left(1 + \ln\\frac{4\pi w}{t}\right) \\
    k_e    &= k_0 + (1 - k_0^2)\,\frac{\Delta}{2s} \\
    \varepsilon_{\mathrm{eff},t}
    &= \varepsilon_{\mathrm{eff}}
    - \frac{0.7\,(\varepsilon_{\mathrm{eff}} - 1)\,t/s}
    {K(k_0^2)/K(1-k_0^2) + 0.7\,t/s} \\
    Z_{0,t} &= \frac{30\pi}
    {\sqrt{\varepsilon_{\mathrm{eff},t}}\;
    K(k_e^2)/K(1-k_e^2)}
    \end{aligned}
    $$

    References:
        Gupta, Garg, Bahl & Bhartia, §7.3, Eqs. 7.98-7.100


    Args:
        w: Centre-conductor width (m).
        s: Gap to ground plane (m).
        t: Conductor thickness (m).
        ep_eff: Uncorrected effective permittivity.

    Returns:
        ``(ep_eff_t, z0_t)`` — thickness-corrected effective permittivity
        and characteristic impedance (Ω).
    """
    w = jnp.asarray(w, dtype=float)
    s = jnp.asarray(s, dtype=float)
    t = jnp.asarray(t, dtype=float)
    ep_eff = jnp.asarray(ep_eff, dtype=float)

    k0 = w / (w + 2.0 * s)
    q0 = ellipk_ratio(k0**2)

    t_safe = jnp.where(t < 1e-15, 1e-15, t)
    delta = (1.25 * t / jnp.pi) * (1.0 + jnp.log(4.0 * jnp.pi * w / t_safe))

    ke = k0 + (1.0 - k0**2) * delta / (2.0 * s)
    ke = jnp.clip(ke, 1e-12, 1.0 - 1e-12)

    ep_eff_t = ep_eff - (0.7 * (ep_eff - 1.0) * t / s) / (q0 + 0.7 * t / s)
    z0_t = 30.0 * jnp.pi / (jnp.sqrt(ep_eff_t) * ellipk_ratio(ke**2))

    ep_eff_t = jnp.where(t <= 0, ep_eff, ep_eff_t)
    z0_t = jnp.where(t <= 0, cpw_z0(w, s, ep_eff), z0_t)

    return ep_eff_t, z0_t


@partial(jax.jit, inline=True)
def microstrip_epsilon_eff(
    w: sax.FloatArrayLike,
    h: sax.FloatArrayLike,
    ep_r: sax.FloatArrayLike,
) -> jax.Array:
    r"""Effective permittivity of a microstrip line.

    Uses the Hammerstad-Jensen formula as given in Pozar.

    $$
    \varepsilon_{\mathrm{eff}} = \frac{\varepsilon_r + 1}{2}
        + \frac{\varepsilon_r - 1}{2} \left(\frac{1}{\sqrt{1 + 12h/w}}
        + 0.04(1 - w/h)^2 \Theta(1 - w/h)\right)
    $$

    where the last term contributes only for narrow strips ($w/h < 1$).

    References:
        Hammerstad & Jensen;
        Pozar, Eqs. 3.195-3.196.

    Args:
        w: Strip width (m).
        h: Substrate height (m).
        ep_r: Relative permittivity of the substrate.

    Returns:
        Effective permittivity (dimensionless).
    """
    w = jnp.asarray(w, dtype=float)
    h = jnp.asarray(h, dtype=float)
    ep_r = jnp.asarray(ep_r, dtype=float)

    u = w / h
    f_u = 1.0 / jnp.sqrt(1.0 + 12.0 / u)

    narrow_correction = 0.04 * (1.0 - u) ** 2
    f_u = jnp.where(u < 1.0, f_u + narrow_correction, f_u)

    return (ep_r + 1.0) / 2.0 + (ep_r - 1.0) / 2.0 * f_u


@partial(jax.jit, inline=True)
def microstrip_z0(
    w: sax.FloatArrayLike,
    h: sax.FloatArrayLike,
    ep_eff: sax.FloatArrayLike,
) -> jax.Array:
    r"""Characteristic impedance of a microstrip line.

    Uses the Hammerstad-Jensen approximation as given in
    Pozar.


    $$
    \begin{aligned}
        Z_0 = \begin{cases}
    \displaystyle\frac{60}{\sqrt{\varepsilon_{\mathrm{eff}}}}
    \ln\!\left(\frac{8h}{w} + \frac{w}{4h}\right)
    & w/h \le 1 \\[6pt]
    \displaystyle\frac{120\pi}
    {\sqrt{\varepsilon_{\mathrm{eff}}}\,
    \bigl[w/h + 1.393 + 0.667\ln(w/h + 1.444)\bigr]}
    & w/h \ge 1
    \end{cases}
    \end{aligned}
    $$


    References:
        Hammerstad & Jensen;
        Pozar, Eqs. 3.197-3.198.


    Args:
        w: Strip width (m).
        h: Substrate height (m).
        ep_eff: Effective permittivity (see :func:`microstrip_epsilon_eff`).

    Returns:
        Characteristic impedance (Ω).
    """
    w = jnp.asarray(w, dtype=float)
    h = jnp.asarray(h, dtype=float)
    ep_eff = jnp.asarray(ep_eff, dtype=float)

    u = w / h
    z_narrow = (60.0 / jnp.sqrt(ep_eff)) * jnp.log(8.0 / u + u / 4.0)
    z_wide = (
        120.0 * jnp.pi / (jnp.sqrt(ep_eff) * (u + 1.393 + 0.667 * jnp.log(u + 1.444)))
    )

    return jnp.where(u <= 1.0, z_narrow, z_wide)


@partial(jax.jit, inline=True)
def microstrip_thickness_correction(
    w: sax.FloatArrayLike,
    h: sax.FloatArrayLike,
    t: sax.FloatArrayLike,
    ep_r: sax.FloatArrayLike,
    ep_eff: sax.FloatArrayLike,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Conductor thickness correction for a microstrip line.

    Uses the widely-adopted Schneider correction as presented in
    Pozar and Gupta et al.



    $$
    \begin{aligned}
        w_e &= w + \frac{t}{\pi}
    \ln\frac{4e}{\sqrt{(t/h)^2 + (t/(wPI + 1.1tPI))^2}} \\
    \varepsilon_{\mathrm{eff},t}
    &= \varepsilon_{\mathrm{eff}}
    - \frac{(\varepsilon_r - 1)\,t/h}
    {4.6\,\sqrt{w/h}}
    \end{aligned}
    $$


    Then the corrected $Z_0$ is computed with the effective width
    $w_e$ and corrected $\varepsilon_{\mathrm{eff},t}$.

    References:
        Pozar, §3.8;
        Gupta, Garg, Bahl & Bhartia

    Args:
        w: Strip width (m).
        h: Substrate height (m).
        t: Conductor thickness (m).
        ep_r: Relative permittivity of the substrate.
        ep_eff: Uncorrected effective permittivity.

    Returns:
        ``(w_eff, ep_eff_t, z0_t)`` — effective width (m),
        thickness-corrected effective permittivity,
        and characteristic impedance (Ω).
    """
    w = jnp.asarray(w, dtype=float)
    h = jnp.asarray(h, dtype=float)
    t = jnp.asarray(t, dtype=float)
    ep_r = jnp.asarray(ep_r, dtype=float)
    ep_eff = jnp.asarray(ep_eff, dtype=float)

    term = jnp.sqrt((t / h) ** 2 + (t / (w * jnp.pi + 1.1 * t * jnp.pi)) ** 2)
    term_safe = jnp.where(term < 1e-15, 1.0, term)
    w_eff = w + (t / jnp.pi) * jnp.log(4.0 * jnp.e / term_safe)

    ep_eff_t = ep_eff - (ep_r - 1.0) * t / h / (4.6 * jnp.sqrt(w / h))
    z0_t = microstrip_z0(w_eff, h, ep_eff_t)

    w_eff = jnp.where(t <= 0, w, w_eff)
    ep_eff_t = jnp.where(t <= 0, ep_eff, ep_eff_t)
    z0_t = jnp.where(t <= 0, microstrip_z0(w, h, ep_eff), z0_t)

    return w_eff, ep_eff_t, z0_t


@partial(jax.jit, inline=True)
def propagation_constant(
    f: sax.FloatArrayLike,
    ep_eff: sax.FloatArrayLike,
    tand: sax.FloatArrayLike = 0.0,
    ep_r: sax.FloatArrayLike = 1.0,
) -> jax.Array:
    r"""Complex propagation constant of a quasi-TEM transmission line.

    For the general lossy case

    $$
    \gamma = \alpha_d + j\,\beta
    $$




    where the **dielectric attenuation** is



    $$
    \alpha_d = \frac{\pi f}{C_M_S}
    \frac{\varepsilon_r}{\sqrt{\varepsilon_{\mathrm{eff}}}}
    \frac{\varepsilon_{\mathrm{eff}} - 1}
    {\varepsilon_r - 1}
    \tan\delta
    $$




    and the **phase constant** is



    $$
    \beta = \frac{2\pi f}{C_M_S}\,\sqrt{\varepsilon_{\mathrm{eff}}}
    $$




    For a superconducting line ($\tan\delta = 0$) the propagation
    is purely imaginary: $\gamma = j\beta$.

    References:
        Pozar, §3.8

    Args:
        f: Frequency (Hz).
        ep_eff: Effective permittivity.
        tand: Dielectric loss tangent (default 0 — lossless).
        ep_r: Substrate relative permittivity (only needed when ``tand > 0``).

    Returns:
        Complex propagation constant $\gamma$ (1/m).
    """
    f = jnp.asarray(f, dtype=float)
    ep_eff = jnp.asarray(ep_eff, dtype=float)
    tand = jnp.asarray(tand, dtype=float)
    ep_r = jnp.asarray(ep_r, dtype=float)

    beta = 2.0 * jnp.pi * f * jnp.sqrt(ep_eff) / C_M_S

    denom = jnp.where(jnp.abs(ep_r - 1.0) < 1e-15, 1.0, ep_r - 1.0)
    alpha_d = (
        jnp.pi * f / C_M_S * (ep_r / jnp.sqrt(ep_eff)) * ((ep_eff - 1.0) / denom) * tand
    )
    alpha_d = jnp.where(jnp.abs(ep_r - 1.0) < 1e-15, 0.0, alpha_d)

    return alpha_d + 1j * beta


@partial(jax.jit, inline=True)
def transmission_line_s_params(
    gamma: sax.ComplexLike,
    z0: sax.ComplexLike,
    length: sax.FloatArrayLike,
    z_ref: sax.ComplexLike | None = None,
) -> tuple[jax.Array, jax.Array]:
    r"""S-parameters of a uniform transmission line (ABCD→S conversion).

    The ABCD matrix of a line with characteristic impedance $Z_0$,
    propagation constant $\gamma$, and length $\ell$ is:



    $$
    \begin{aligned}
        \begin{pmatrix} A & B \\ C & D \end{pmatrix}
    = \begin{pmatrix}
        \cosh\theta & Z_0\sinh\theta \\
        \sinh\theta / Z_0 & \cosh\theta
    \end{pmatrix}, \quad \theta = \gamma\ell
    \end{aligned}
    $$



    Converting to S-parameters referenced to $Z_{\mathrm{ref}}$



    $$
    \begin{aligned}
        S_{11} &= \frac{A + B/Z_{\mathrm{ref}} - CZ_{\mathrm{ref}} - D}{
            A + B/Z_{\mathrm{ref}} + CZ_{\mathrm{ref}} + D} \\
        S_{21} &= \frac{2}{A + B/Z_{\mathrm{ref}} + CZ_{\mathrm{ref}} + D}
    \end{aligned}
    $$




    When ``z_ref`` is ``None`` the reference impedance defaults to ``z0``
    (matched case), giving $S_{11} = 0$ and
    $S_{21} = e^{-\gamma\ell}$.

    References:
        Pozar, Table 4.2

    Args:
        gamma: Complex propagation constant (1/m).
        z0: Characteristic impedance (Ω).
        length: Physical length (m).
        z_ref: Reference (port) impedance (Ω).  Defaults to ``z0``.

    Returns:
        ``(S11, S21)`` — complex S-parameter arrays.
    """
    gamma_arr = jnp.asarray(gamma, dtype=complex)
    z0_arr = jnp.asarray(z0, dtype=complex)
    length_arr = jnp.asarray(length, dtype=float)

    if z_ref is None:
        z_ref = z0
    z_ref_arr = jnp.asarray(z_ref, dtype=complex)

    theta = gamma_arr * length_arr

    cosh_t = jnp.cosh(theta)
    sinh_t = jnp.sinh(theta)

    a = cosh_t
    b = z0_arr * sinh_t
    c = sinh_t / z0_arr

    denom = a + b / z_ref_arr + c * z_ref_arr + a
    s11 = (b / z_ref_arr - c * z_ref_arr) / denom
    s21 = 2.0 / denom

    return s11, s21


@partial(jax.jit, inline=True)
def coplanar_waveguide(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: sax.FloatLike = 1000.0,
    width: sax.FloatLike = 10.0,
    gap: sax.FloatLike = 5.0,
    thickness: sax.FloatLike = 0.0,
    substrate_thickness: sax.FloatLike = 500.0,
    ep_r: sax.FloatLike = 11.45,
    tand: sax.FloatLike = 0.0,
) -> sax.SDict:
    r"""S-parameter model for a straight coplanar waveguide.

    Computes S-parameters analytically using conformal-mapping CPW theory
    following Simons and the Qucs-S CPW model.
    Conductor thickness corrections use the first-order model of
    Gupta, Garg, Bahl, and Bhartia.

    References:
       Simons, ch. 2;
       Gupta, Garg, Bahl & Bhartia;
       Qucs technical documentation, §12.4

    Args:
        f: Array of frequency points in Hz
        length: Physical length in µm
        width: Centre-conductor width in µm
        gap: Gap between centre conductor and ground plane in µm
        thickness: Conductor thickness in µm
        substrate_thickness: Substrate height in µm
        ep_r: Relative permittivity of the substrate
        tand: Dielectric loss tangent

    Returns:
        sax.SDict: S-parameters dictionary
    """
    f = jnp.asarray(f)
    f_flat = f.ravel()

    w_m = width * 1e-6
    s_m = gap * 1e-6
    t_m = thickness * 1e-6
    h_m = substrate_thickness * 1e-6
    length_m = jnp.asarray(length) * 1e-6

    ep_eff = cpw_epsilon_eff(w_m, s_m, h_m, ep_r)

    # Thickness is a scalar, so we use jnp.where to conditionally apply corrections.
    # Using jnp.where is safe:
    ep_eff_t, z0_val = cpw_thickness_correction(w_m, s_m, t_m, ep_eff)

    gamma = propagation_constant(f_flat, ep_eff_t, tand=tand, ep_r=ep_r)
    s11, s21 = transmission_line_s_params(gamma, z0_val, length_m)

    sdict: sax.SDict = {
        ("o1", "o1"): s11.reshape(f.shape),
        ("o1", "o2"): s21.reshape(f.shape),
        ("o2", "o2"): s11.reshape(f.shape),
    }
    return sax.reciprocal(sdict)


@partial(jax.jit, inline=True)
def microstrip(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: sax.FloatLike = 1000.0,
    width: sax.FloatLike = 10.0,
    substrate_thickness: sax.FloatLike = 500.0,
    thickness: sax.FloatLike = 0.2,
    ep_r: sax.FloatLike = 11.45,
    tand: sax.FloatLike = 0.0,
) -> sax.SDict:
    r"""S-parameter model for a straight microstrip transmission line.

    Computes S-parameters analytically using the Hammerstad-Jensen closed-form
    expressions for effective permittivity and characteristic impedance, as
    described in Pozar.
    Conductor thickness corrections follow
    Gupta et al.

    References:
        Hammerstad & Jensen;
        Pozar, ch. 3, §3.8;
        Gupta, Garg, Bahl & Bhartia, §2.2.4

    Args:
        f: Array of frequency points in Hz.
        length: Physical length in µm.
        width: Strip width in µm.
        substrate_thickness: Substrate height in µm.
        thickness: Conductor thickness in µm (default 0.2 µm = 200 nm).
        ep_r: Relative permittivity of the substrate (default 11.45 for Si).
        tand: Dielectric loss tangent (default 0 — lossless).

    Returns:
        sax.SDict: S-parameters dictionary.
    """
    f = jnp.asarray(f)
    f_flat = f.ravel()

    w_m = width * 1e-6
    h_m = substrate_thickness * 1e-6
    t_m = thickness * 1e-6
    length_m = jnp.asarray(length) * 1e-6

    ep_eff = microstrip_epsilon_eff(w_m, h_m, ep_r)
    _w_eff, ep_eff_t, z0_val = microstrip_thickness_correction(
        w_m, h_m, t_m, ep_r, ep_eff
    )

    gamma = propagation_constant(f_flat, ep_eff_t, tand=tand, ep_r=ep_r)
    s11, s21 = transmission_line_s_params(gamma, z0_val, length_m)

    sdict: sax.SDict = {
        ("o1", "o1"): s11.reshape(f.shape),
        ("o1", "o2"): s21.reshape(f.shape),
        ("o2", "o2"): s11.reshape(f.shape),
    }
    return sax.reciprocal(sdict)
