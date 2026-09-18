import json
from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from kfnetlist import Netlist, NetlistPort, PlacedNetlist, Placement, PortRef

import sax
from sax import native


def _gain(gain: sax.ComplexArrayLike = 1.0) -> sax.SDict:
    return {("in", "out"): jnp.asarray(gain), ("out", "in"): jnp.asarray(gain) / 2}


def _flat(instance: object) -> dict:
    return {"instances": {"a": instance}, "ports": {"in": "a,in", "out": "a,out"}}


@pytest.mark.parametrize("backend", ["klu", "fg"])
def test_partial_instances_keep_distinct_defaults_and_overrides(
    backend: sax.BackendLike,
) -> None:
    net = {
        "instances": {"a": partial(_gain, gain=2), "b": partial(_gain, gain=3)},
        "connections": {"a,out": "b,in"},
        "ports": {"in": "a,in", "out": "b,out"},
    }
    model, _ = sax.circuit(net, backend=backend)
    np.testing.assert_allclose(model()["in", "out"], 6)
    np.testing.assert_allclose(model(gain=4, b={"gain": 5})["in", "out"], 20)
    np.testing.assert_allclose(jax.jit(model)()["in", "out"], 6)
    assert isinstance(net["instances"]["a"], partial)


def test_callable_binding_does_not_override_same_named_factory() -> None:
    def first() -> sax.SDict:
        return _gain(2)

    def second() -> sax.SDict:
        return _gain(3)

    first.__name__ = second.__name__ = "same"
    net = {
        "instances": {"a": first, "b": second, "c": "same"},
        "connections": {"a,out": "b,in", "b,out": "c,in"},
        "ports": {"in": "a,in", "out": "c,out"},
    }
    model, _ = sax.circuit(net, {"same": partial(_gain, gain=5)})
    np.testing.assert_allclose(model()["in", "out"], 30)


def test_positional_partial_rejected() -> None:
    with pytest.raises(ValueError, match="positional arguments"):
        sax.circuit(_flat(partial(_gain, 3)))


def test_legacy_info_precedence_native_info_stays_metadata() -> None:
    net = _flat({"component": "gain", "settings": {"gain": 2}, "info": {"gain": 3}})
    before = deepcopy(net)
    model, _ = sax.circuit(net, {"gain": _gain})
    np.testing.assert_allclose(model()["in", "out"], 3)
    assert net == before
    nl = Netlist()
    nl.create_inst(
        "a", kcl="", component="gain", settings={"gain": 2}, info={"gain": 3}
    )
    for port in ("in", "out"):
        nl.create_port(port)
        nl.create_net(NetlistPort(name=port), PortRef(instance="a", port=port))
    model, _ = sax.circuit(nl, {"gain": _gain})
    np.testing.assert_allclose(model()["in", "out"], 2)


@pytest.mark.parametrize("value", [np.array([2, 3]), jnp.array([2, 3]), 2 + 3j])
@pytest.mark.parametrize("backend", ["klu", "fg"])
def test_python_numeric_settings(value: object, backend: sax.BackendLike) -> None:
    net = _flat({"component": "gain", "settings": {"gain": value}})
    model, _ = sax.circuit(net, {"gain": _gain}, backend=backend)
    np.testing.assert_allclose(model()["in", "out"], value)
    np.testing.assert_allclose(jax.jit(model)()["in", "out"], value)
    np.testing.assert_allclose(model(a={"gain": 7})["in", "out"], 7)
    assert sax.get_required_circuit_models(net, {"gain": _gain}) == ["gain"]
    derivative = jax.grad(lambda g: jnp.abs(model(gain=g)["in", "out"]) ** 2)(2.0)
    np.testing.assert_allclose(derivative, 4)


def test_trace_time_settings_stay_outside_native_serialization() -> None:
    def objective(gain: jax.Array) -> jax.Array:
        model, _ = sax.circuit(
            _flat({"component": "gain", "settings": {"gain": gain}}), {"gain": _gain}
        )
        return jnp.abs(model()["in", "out"]) ** 2

    np.testing.assert_allclose(jax.grad(objective)(3.0), 6)


def test_legacy_placement_normalization() -> None:
    def placed(placement: dict | None = None) -> sax.SDict:
        return _gain((placement or {"x": 11})["x"])

    net = _flat("placed")
    model, _ = sax.circuit(net, {"placed": placed})
    np.testing.assert_allclose(model()["in", "out"], 11)
    net["placements"] = {
        "a": {"x": 5, "dx": 2, "y": 3, "dy": 4, "rotation": 450.4, "mirror": True}
    }
    model, _ = sax.circuit(net, {"placed": placed})
    np.testing.assert_allclose(model()["in", "out"], 7)
    nl = native.from_legacy_flat(net)
    assert isinstance(nl, PlacedNetlist)
    assert native.placements(nl)["a"] == {
        "x": 7,
        "y": 7,
        "rotation": 90,
        "mirror": True,
    }


def _placed_hierarchy() -> dict[str, Netlist]:
    top = PlacedNetlist()
    top.create_inst(
        "a",
        kcl="D",
        component="factory",
        cell="child",
        info={"nested": {"label": "original"}},
        placement=Placement(
            x=7,
            y=8,
            orientation=90,
            mirror=True,
            bbox={"left": 0, "bottom": 0, "right": 1, "top": 2},
        ),
    )
    for port in ("in", "out"):
        top.create_port(port)
        top.create_net(NetlistPort(name=port), PortRef(instance="a", port=port))
    child = native.from_legacy_flat(
        _flat({"component": "gain", "settings": {"gain": 3}})
    )
    return {"top": top, "child": child}


@pytest.mark.parametrize("encoding", ["objects", "dict", "json"])
def test_placed_hierarchy_serialization_keeps_fallback(encoding: str) -> None:
    cells = _placed_hierarchy()
    value = (
        cells
        if encoding == "objects"
        else {name: nl.to_dict() for name, nl in cells.items()}
    )
    if encoding == "json":
        value = json.dumps(value)
    model, _ = sax.circuit(value, {"gain": _gain})
    np.testing.assert_allclose(model()["in", "out"], 3)
    np.testing.assert_allclose(model()["out", "in"], 1.5)


@pytest.mark.parametrize("placed", [False, True])
def test_native_copy_is_independent_and_preserves_data(placed: bool) -> None:
    nl = _placed_hierarchy()["top"] if placed else Netlist()
    nl.create_inst(
        "array",
        kcl="D",
        component="gain",
        na=2,
        nb=3,
        settings={"nested": {"gain": [1, 2]}},
        info={"tag": "keep"},
    )
    before = nl.to_dict()
    copied = native.copy_netlist(nl)
    assert type(copied) is type(nl)
    assert copied.to_dict() == before
    copied.remove_instances(["array"])
    assert nl.to_dict() == before
    assert "array" not in copied.instances
    if placed:
        assert copied.instances["a"].cell == "child"
