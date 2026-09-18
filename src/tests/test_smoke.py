"""Small numerical sanity suite; run with `just smoke`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
    net = {
        "instances": {
            "a": {"component": "component", "settings": {"gain": 2.0}},
            "b": "component",
        },
        "connections": {"a,out0": "b,in0"},
        "ports": {"in": "a,in0", "out": "b,out0"},
    }
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


def _native_issue120_fixture() -> dict:
    """Native placed hierarchy: one factory, two distinct instantiated cells."""
    kf = pytest.importorskip("kfnetlist")
    del kf
    from kfnetlist import Netlist, Placement, PlacedNetlist

    port = sax.native.NetlistPort
    ref = sax.native.PortRef

    def sub() -> PlacedNetlist:
        nl = PlacedNetlist()
        nl.create_inst(
            "layout",
            kcl="D",
            component="uncoupled_layout",
            settings={},
            cell="uncoupled_cell",
            placement=Placement(
                x=0.0,
                y=0.0,
                orientation=0.0,
                mirror=False,
                bbox={"left": 0.0, "bottom": 0.0, "right": 1.0, "top": 1.0},
            ),
        )
        nl.create_port("in0")
        nl.create_port("out0")
        nl.create_net(port(name="in0"), ref(instance="layout", port="in0"))
        nl.create_net(ref(instance="layout", port="out0"), port(name="out0"))
        return nl

    top = PlacedNetlist()
    top.create_inst(
        "a", kcl="D", component="coupled", settings={"gain": 0.2}, cell="coupled_a"
    )
    top.create_inst(
        "b", kcl="D", component="coupled", settings={"gain": 0.3}, cell="coupled_b"
    )
    for name in ("a_in", "a_out", "b_in", "b_out"):
        top.create_port(name)
    top.create_net(port(name="a_in"), ref(instance="a", port="in0"))
    top.create_net(ref(instance="a", port="out0"), port(name="a_out"))
    top.create_net(port(name="b_in"), ref(instance="b", port="in0"))
    top.create_net(ref(instance="b", port="out0"), port(name="b_out"))

    return {"coupled_a": sub(), "coupled_b": sub(), "top": top}


def test_native_counted_hierarchy_model_identity() -> None:
    """Issue #120 acceptance: one factory model covers every parameterization.

    Native ``PlacedNetlist`` input keeps the factory name (``component``) and
    the instantiated cell (``cell``) distinct, so both variants resolve to the
    shared analytical model with their own settings.
    """
    cells = _native_issue120_fixture()
    models = {"coupled": _component, "uncoupled_layout": _uncoupled_layout}
    model, _ = sax.circuit(cells, models, top_level_name="top", backend="klu")
    result = model(wl=1.55)
    np.testing.assert_allclose(result["a_in", "a_out"], 0.2 * jnp.exp(1.55j))
    np.testing.assert_allclose(result["b_in", "b_out"], 0.3 * jnp.exp(1.55j))


def test_legacy_counted_names_remain_ambiguous() -> None:
    """Legacy counted names have lost factory provenance; do not guess them.

    The un-numbered variant resolves to the analytical model, while the counted
    variant silently descends into the layout subcircuit because legacy input
    carries no separate factory identity. This documents the limitation rather
    than hiding it; it is not a contract to strip numeric suffixes.
    """
    recnet = {
        "top_level": {
            "instances": {
                "a": {"component": "coupled", "settings": {"gain": 0.2}},
                "b": {"component": "coupled2", "settings": {"gain": 0.3}},
            },
            "ports": {
                "a_in": "a,in0",
                "a_out": "a,out0",
                "b_in": "b,in0",
                "b_out": "b,out0",
            },
        },
        **{
            name: {
                "instances": {"layout": "uncoupled_layout"},
                "ports": {"in0": "layout,in0", "out0": "layout,out0"},
            }
            for name in ("coupled", "coupled2")
        },
    }
    models = {"coupled": _component, "uncoupled_layout": _uncoupled_layout}
    model, _ = sax.circuit(recnet, models, backend="klu")
    result = model(wl=1.55)
    np.testing.assert_allclose(result["a_in", "a_out"], 0.2 * jnp.exp(1.55j))
    # No factory provenance: the counted variant uses the layout subnet.
    np.testing.assert_allclose(result["b_in", "b_out"], 0.0)
    # An explicit alias supplies the missing provenance without suffix guessing.
    aliased, _ = sax.circuit(recnet, {**models, "coupled2": _component}, backend="klu")
    np.testing.assert_allclose(aliased(wl=1.55)["b_in", "b_out"], 0.3 * jnp.exp(1.55j))
