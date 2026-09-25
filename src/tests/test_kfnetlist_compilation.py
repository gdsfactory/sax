"""SAX compiles kfnetlist topology without a second instance or wiring schema."""

import jax.numpy as jnp
import numpy as np
import pytest
from kfnetlist import Netlist, NetlistPort, PortArrayRef, PortRef

import sax


def _asymmetric(gain: float = 2.0) -> sax.SDict:
    return {("in", "out"): jnp.asarray(gain), ("out", "in"): 0.5}


def test_array_endpoints_stay_in_kfnetlist_until_expansion() -> None:
    netlist = Netlist()
    netlist.create_inst("wg", "pdk", "waveguide", na=2)
    netlist.create_net(PortArrayRef("wg", "out", 1, 1), PortArrayRef("wg", "in", 2, 1))
    netlist.create_port("in")
    netlist.create_port("out")
    netlist.create_net(NetlistPort("in"), PortArrayRef("wg", "in", 1, 1))
    netlist.create_net(NetlistPort("out"), PortArrayRef("wg", "out", 2, 1))

    model, _ = sax.circuit(netlist, {"waveguide": _asymmetric})
    result = model()
    np.testing.assert_allclose(result["in", "out"], 4.0)
    np.testing.assert_allclose(result["out", "in"], 0.25)


def test_mode_expansion_uses_kfnetlist_nets() -> None:
    netlist = Netlist()
    netlist.create_inst("wg", "pdk", "waveguide")
    for port in ("in", "out"):
        netlist.create_port(port)
        netlist.create_net(NetlistPort(port), PortRef("wg", port))

    model, _ = sax.circuit(
        netlist,
        {"waveguide": sax.multimode(_asymmetric, modes=("TE", "TM"))},
    )
    result = model()
    for mode in ("TE", "TM"):
        np.testing.assert_allclose(result[f"in@{mode}", f"out@{mode}"], 2.0)
        np.testing.assert_allclose(result[f"out@{mode}", f"in@{mode}"], 0.5)


@pytest.mark.parametrize("backend", ["klu", "fg"])
def test_probe_insertion_uses_kfnetlist_instances(backend: str) -> None:
    netlist = Netlist()
    for name in ("a", "b"):
        netlist.create_inst(name, "pdk", "waveguide")
    netlist.create_net(PortRef("a", "out"), PortRef("b", "in"))
    for port, instance, child_port in (
        ("in", "a", "in"),
        ("out", "b", "out"),
    ):
        netlist.create_port(port)
        netlist.create_net(NetlistPort(port), PortRef(instance, child_port))

    model, _ = sax.circuit(
        netlist, {"waveguide": _asymmetric}, backend=backend, probes={"tap": "a,out"}
    )
    result = model()
    np.testing.assert_allclose(result["in", "out"], 4.0)
    assert {"tap_fwd", "tap_bwd"} <= set(sax.get_ports(result))
