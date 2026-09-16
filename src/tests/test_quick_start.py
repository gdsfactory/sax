import jax.numpy as jnp

import sax


def test_quick_start() -> None:
    """Runs the core parts of the quick start notebook.

    This does not use jax.jit, to support the cupy backend.
    """

    def coupler(coupling: float = 0.5) -> sax.SDict:
        kappa = coupling**0.5
        tau = (1 - coupling) ** 0.5
        return sax.reciprocal(
            {
                ("in0", "out0"): tau,
                ("in0", "out1"): 1j * kappa,
                ("in1", "out0"): 1j * kappa,
                ("in1", "out1"): tau,
            }
        )

    coupler(coupling=0.3)

    def waveguide(
        wl: float = 1.55,
        wl0: float = 1.55,
        neff: float = 2.34,
        ng: float = 3.4,
        length: float = 10.0,
        loss: float = 0.0,
    ) -> sax.SDict:
        dwl = wl - wl0
        dneff_dwl = (ng - neff) / wl0
        neff = neff - dwl * dneff_dwl
        phase = 2 * jnp.pi * neff * length / wl
        transmission = 10 ** (-loss * length / 20) * jnp.exp(1j * phase)
        return sax.reciprocal(
            {
                ("in0", "out0"): transmission,
            }
        )

    mzi, _info = sax.circuit(
        netlist={
            "instances": {
                "lft": "coupler",
                "top": "waveguide",
                "btm": "waveguide",
                "rgt": "coupler",
            },
            "connections": {
                "lft,out0": "btm,in0",
                "btm,out0": "rgt,in0",
                "lft,out1": "top,in0",
                "top,out0": "rgt,in1",
            },
            "ports": {
                "in0": "lft,in0",
                "in1": "lft,in1",
                "out0": "rgt,out0",
                "out1": "rgt,out1",
            },
        },
        models={
            "coupler": coupler,
            "waveguide": waveguide,
        },
    )

    mzi()
    mzi(top={"length": 25.0}, btm={"length": 15.0})
    wl = jnp.linspace(1.51, 1.59, 1000)
    mzi(wl=wl, top={"length": 25.0}, btm={"length": 15.0})
