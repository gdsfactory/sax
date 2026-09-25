"""Small numerical sanity suite; run with `just smoke`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from kfnetlist import HierarchicalNetlist, Netlist, NetlistPort, PortRef

import sax


def _component(wl: sax.FloatArrayLike = 1.55, gain: float = 1.0) -> sax.SDict:
    return sax.reciprocal({("in0", "out0"): gain * jnp.exp(1j * jnp.asarray(wl))})


def test_asymmetric_representations() -> None:
    original: sax.SDict = {
        ("a", "b"): jnp.asarray(2 + 3j),
        ("b", "a"): jnp.asarray(4 - 1j),
    }
    dense, ports = sax.sdense(original)
    np.testing.assert_allclose(dense[ports["b"], ports["a"]], 2 + 3j)
    for converted in (sax.sdict(sax.scoo(original)), sax.sdict((dense, ports))):
        for key, value in original.items():
            np.testing.assert_allclose(converted[key], value)


@pytest.mark.parametrize("backend", ["klu", "fg"])
def test_circuit_settings_jit_gradient(backend: sax.BackendLike) -> None:
    net = Netlist()
    net.create_inst("a", "pdk", "component", {"gain": 2.0})
    net.create_inst("b", "pdk", "component")
    net.create_net(PortRef("a", "out0"), PortRef("b", "in0"))
    for name, instance, port in (("in", "a", "in0"), ("out", "b", "out0")):
        net.create_port(name)
        net.create_net(NetlistPort(name=name), PortRef(instance, port))
    model, _ = sax.circuit(net, {"component": _component}, backend=backend)
    wl = jnp.array([1.5, 1.6])
    np.testing.assert_allclose(
        model(wl=wl, gain=2.0, b={"gain": 3.0})["in", "out"],
        6 * jnp.exp(2j * wl),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        jax.jit(model)(wl=wl)["in", "out"],
        2 * jnp.exp(2j * wl),
        rtol=1e-5,
    )
    gradient = jax.grad(lambda g: jnp.abs(model(a={"gain": g})["in", "out"]) ** 2)(2.0)
    np.testing.assert_allclose(gradient, 4.0, rtol=1e-5)


def test_optical_model_limit() -> None:
    result = sax.models.straight(length=0.0, wl=jnp.array([1.5, 1.6]))
    np.testing.assert_allclose(result["in0", "out0"], 1.0)


def _uncoupled_layout() -> sax.SDict:
    # Layout connectivity alone cannot describe the analytical coupling.
    return sax.reciprocal({("in0", "out0"): jnp.asarray(0.0)})


def test_reference_specific_model_keeps_factory_settings() -> None:
    """Two referenced children share a factory but retain independent settings."""
    child = Netlist()
    child.create_inst("layout", "pdk", "uncoupled_layout")
    for port in ("in0", "out0"):
        child.create_port(port)
        child.create_net(NetlistPort(name=port), PortRef("layout", port))

    top = Netlist()
    for name, gain in (("a", 0.2), ("b", 0.3)):
        top.create_inst(
            name, "pdk", "coupled", {"gain": gain}, netlist_id=f"coupled_{name}"
        )
        for port in ("in0", "out0"):
            top.create_port(f"{name}_{port}")
            top.create_net(NetlistPort(name=f"{name}_{port}"), PortRef(name, port))

    document = HierarchicalNetlist({"top": top, "coupled_a": child, "coupled_b": child})
    model, _ = sax.circuit(document, {"coupled": _component}, top_level_name="top")
    result = model(wl=1.55)
    np.testing.assert_allclose(result["a_in0", "a_out0"], 0.2 * jnp.exp(1.55j))
    np.testing.assert_allclose(result["b_in0", "b_out0"], 0.3 * jnp.exp(1.55j))
