import jax.numpy as jnp
import numpy as np
import pytest
from kfnetlist import Netlist, NetlistPort, PortRef

import sax


def _component(value: sax.FloatArrayLike = 0.8) -> sax.SDict:
    return sax.reciprocal({("in0", "out0"): jnp.asarray(value)})


def _multilink_netlist() -> Netlist:
    netlist = Netlist()
    for name in ("a", "b", "c"):
        netlist.create_inst(name, "pdk", "component")
    netlist.create_net(PortRef("a", "out0"), PortRef("b", "in0"))
    netlist.create_net(PortRef("a", "out0"), PortRef("c", "in0"))
    for name, instance, port in (
        ("in0", "a", "in0"),
        ("out0", "b", "out0"),
        ("out1", "c", "out0"),
    ):
        netlist.create_port(name)
        netlist.create_net(NetlistPort(name), PortRef(instance, port))
    return netlist


@pytest.mark.parametrize("backend", ["fg", "additive"])
def test_pairwise_backends_reject_multilinks(backend: sax.BackendLike) -> None:
    with pytest.raises(ValueError, match="Multiply connected ports"):
        sax.circuit(_multilink_netlist(), {"component": _component}, backend=backend)


def test_klu_multilinks_copy_not_power_conserving_split() -> None:
    model, _ = sax.circuit(
        _multilink_netlist(), {"component": _component}, backend="klu"
    )
    result = model()
    np.testing.assert_allclose(result["in0", "out0"], 0.8**2)
    np.testing.assert_allclose(result["in0", "out1"], 0.8**2)


def test_additive_quantities_are_path_lengths_not_amplitude_products() -> None:
    netlist = Netlist()
    for name in ("a", "b"):
        netlist.create_inst(name, "pdk", "component")
    netlist.create_net(PortRef("a", "out0"), PortRef("b", "in0"))
    for name, instance, port in (("in0", "a", "in0"), ("out0", "b", "out0")):
        netlist.create_port(name)
        netlist.create_net(NetlistPort(name), PortRef(instance, port))
    model, _ = sax.circuit(
        netlist,
        {"component": _component},
        backend="additive",
    )
    result = model(a={"value": 2.0}, b={"value": 3.0})
    assert isinstance(result["in0", "out0"], list)
    np.testing.assert_allclose(result["in0", "out0"], [[5.0]])
