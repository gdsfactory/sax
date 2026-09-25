"""Runtime settings remain separate from serialized kfnetlist topology."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from kfnetlist import Netlist, NetlistPort, PortRef

import sax


def _gain(gain: float = 1.0) -> sax.SDict:
    value = jnp.asarray(gain)
    return {("in", "out"): value, ("out", "in"): value / 2}


def _chain() -> Netlist:
    netlist = Netlist()
    netlist.create_inst("a", "pdk", "gain", {"gain": 2.0}, info={"gain": 99})
    netlist.create_inst("b", "pdk", "gain", {"gain": 3.0})
    netlist.create_net(PortRef("a", "out"), PortRef("b", "in"))
    for name, instance, port in (("in", "a", "in"), ("out", "b", "out")):
        netlist.create_port(name)
        netlist.create_net(NetlistPort(name), PortRef(instance, port))
    return netlist


@pytest.mark.parametrize("backend", ["klu", "fg"])
def test_json_instance_settings_and_runtime_overrides(backend: str) -> None:
    netlist = _chain()
    before = netlist.to_dict()
    model, _ = sax.circuit(netlist, {"gain": _gain}, backend=backend)

    np.testing.assert_allclose(model()["in", "out"], 6.0)
    np.testing.assert_allclose(model(gain=4.0, b={"gain": 5.0})["in", "out"], 20.0)
    np.testing.assert_allclose(jax.jit(model)()["in", "out"], 6.0)
    assert netlist.to_dict() == before


def test_runtime_array_and_gradient_stay_outside_netlist() -> None:
    netlist = _chain()
    model, _ = sax.circuit(netlist, {"gain": _gain})
    values = jnp.asarray([1.0, 2.0, 3.0])
    np.testing.assert_allclose(model(a={"gain": values})["in", "out"], 3 * values)
    derivative = jax.grad(
        lambda gain: jnp.abs(model(a={"gain": gain})["in", "out"]) ** 2
    )(2.0)
    np.testing.assert_allclose(derivative, 36.0)
    assert netlist.instances["a"].settings == {"gain": 2.0}
    assert netlist.instances["a"].info == {"gain": 99}
