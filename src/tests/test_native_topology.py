from functools import partial

import numpy as np
import pytest
from kfnetlist import Netlist, NetlistPort, PlacedNetlist, PortArrayRef, PortRef

import sax
from sax import native
from sax.netlists import remove_unused_instances


def _gain(gain: float = 1.0) -> sax.SDict:
    return {("in", "out"): gain, ("out", "in"): gain / 2}


def _single(component: str = "gain", name: str = "a") -> Netlist:
    return native.from_legacy_flat(
        {
            "instances": {name: component},
            "ports": {"in": f"{name},in", "out": f"{name},out"},
        }
    )


def test_disconnected_models_are_not_required() -> None:
    nl = _single()
    nl.create_inst("unused", kcl="", component="missing")
    nl.create_inst("layout", kcl="", component="layout")
    cells = {
        "top": nl,
        "layout": native.from_legacy_flat({"instances": {"shape": "polygon"}}),
    }
    original = {name: net.to_dict() for name, net in cells.items()}
    model, info = sax.circuit(cells, {"gain": _gain})
    np.testing.assert_allclose(model()["in", "out"], 1)
    assert set(info.dag) == {"top", "gain"}
    assert sax.get_required_circuit_models(cells) == ["gain"]
    assert {name: net.to_dict() for name, net in cells.items()} == original


def test_hierarchy_pruning_preserves_placed_types_and_reserved_names() -> None:
    nl = PlacedNetlist()
    nl.create_inst("__port_0", kcl="", component="gain", cell="leaf")
    nl.create_port("in")
    nl.create_net(NetlistPort(name="in"), PortRef(instance="__port_0", port="in"))
    nl.create_inst("unused", kcl="", component="missing")
    original = nl.to_dict()
    pruned = remove_unused_instances({"top": nl})
    assert list(pruned) == ["top"]
    assert isinstance(pruned["top"], PlacedNetlist)
    assert set(pruned["top"].instances) == {"__port_0"}
    assert pruned["top"].instances["__port_0"].cell == "leaf"
    assert nl.to_dict() == original


def _placed_two() -> dict[str, Netlist]:
    top = PlacedNetlist()
    for name, factory in (("a", "analytical"), ("b", "layout")):
        top.create_inst(name, kcl="", component=factory, cell="child")
        for port in ("in", "out"):
            top.create_port(f"{name}_{port}")
            top.create_net(
                NetlistPort(name=f"{name}_{port}"), PortRef(instance=name, port=port)
            )
    return {"top": top, "child": _single()}


@pytest.mark.parametrize("backend", ["klu", "fg"])
def test_flatten_preserves_one_modelled_instance_of_shared_child(
    backend: sax.BackendLike,
) -> None:
    cells = _placed_two()
    original = {name: nl.to_dict() for name, nl in cells.items()}
    models = {"analytical": partial(_gain, gain=7), "gain": _gain}
    expected, _ = sax.circuit(cells, models, backend=backend)
    flat = native.flatten_netlist(cells, "top", models=models)
    assert "a" in flat.instances and "b__a" in flat.instances
    assert flat.instances["a"].cell == "child"
    actual, _ = sax.circuit(flat, models, backend=backend)
    for key, value in expected().items():
        np.testing.assert_allclose(actual()[key], value)
    assert {name: nl.to_dict() for name, nl in cells.items()} == original
    flattened = native.flatten_recursive_netlist(cells, models=models)
    assert "a" in flattened["top"].instances


def test_flatten_respects_exact_cell_model_override() -> None:
    cells = {"top": _single("child"), "child": _single()}
    flat = native.flatten_netlist(cells, "top", models={"child": _gain})
    assert set(flat.instances) == {"a"}


def test_hierarchical_probe_rejects_parent_collision_and_opaque_model() -> None:
    cells = {"top": _single("child", "sub"), "child": _single()}
    top = cells["top"]
    top.create_port("tap_fwd")
    top.create_net(NetlistPort(name="tap_fwd"), PortRef(instance="sub", port="extra"))
    with pytest.raises(ValueError, match="conflict with existing ports"):
        sax.circuit(cells, {"gain": _gain}, probes={"tap": "sub.a,out"})
    opaque = {"top": _single("child", "sub"), "child": _single()}
    with pytest.raises(ValueError, match="Only sub-circuits"):
        sax.circuit(opaque, {"child": _gain}, probes={"tap": "sub.a,out"})


def test_portless_probe_path_survives_pruning() -> None:
    cells = {
        "top": native.from_legacy_flat({"instances": {"sub": "child"}}),
        "child": native.from_legacy_flat(
            {"instances": {"a": "gain", "unused": "missing"}}
        ),
    }
    model, _ = sax.circuit(cells, {"gain": _gain}, probes={"tap": "sub.a,out"})
    assert set(sax.get_ports(model())) == {"tap_fwd", "tap_bwd"}


def test_all_internal_ports_removed_has_useful_error() -> None:
    net = {
        "instances": {"a": "gain", "b": "gain"},
        "connections": {"a,out": "b,in"},
        "ports": {"mid": "a,out"},
    }
    with pytest.raises(ValueError, match="at least 1 port"):
        sax.circuit(net, {"gain": _gain}, on_internal_port="ignore")


@pytest.mark.parametrize("case", ["unconnected", "external_only", "alias", "junction"])
def test_unsupported_native_topology_is_explicit(case: str) -> None:
    nl = _single()
    if case == "unconnected":
        nl.create_port("floating")
        message = "no instance connection"
    elif case == "external_only":
        nl.create_port("floating")
        nl.create_net(NetlistPort(name="floating"))
        message = "External-only"
    elif case == "alias":
        nl.create_port("alias")
        nl.create_net(NetlistPort(name="alias"), PortRef(instance="a", port="in"))
        message = "aliases"
    else:
        for name in ("b", "c"):
            nl.create_inst(name, kcl="", component="gain")
        nl.create_net(*(PortRef(instance=name, port="out") for name in ("a", "b", "c")))
        message = "junction model"
    with pytest.raises(ValueError, match=message):
        sax.circuit(nl, {"gain": _gain})


def test_explicit_multilinks_keep_klu_coefficients_and_fg_restriction() -> None:
    flat = {
        "instances": {name: "gain" for name in ("a", "b", "c")},
        "nets": [
            {"p1": "a,out", "p2": "b,in"},
            {"p1": "a,out", "p2": "c,in"},
            {"p1": "a,out", "p2": "b,in"},
        ],
        "ports": {"in": "a,in", "out1": "b,out", "out2": "c,out"},
    }
    nl = native.from_legacy_flat(flat)
    model, _ = sax.circuit(nl, {"gain": _gain})
    np.testing.assert_allclose(model()["in", "out1"], 1)
    np.testing.assert_allclose(model()["out2", "in"], 0.25)
    with pytest.raises(ValueError, match="Multiply connected"):
        sax.circuit(nl, {"gain": _gain}, backend="fg")


def test_array_probe_and_transform_preserve_settings() -> None:
    nl = Netlist()
    nl.create_inst("arr", kcl="", component="gain", na=2, nb=1, settings={"gain": 3})
    for port, index in (("in", 1), ("out", 2)):
        nl.create_port(port)
        nl.create_net(
            NetlistPort(name=port),
            PortArrayRef(instance="arr", port=port, ia=index, ib=1),
        )
    nl.create_net(
        PortArrayRef(instance="arr", port="out", ia=1, ib=1),
        PortArrayRef(instance="arr", port="in", ia=2, ib=1),
    )
    renamed = native.rename_models(
        native.rename_instances(nl, {"arr": "array"}), {"gain": "renamed"}
    )
    model, _ = sax.circuit(
        renamed, {"renamed": _gain}, probes={"tap": "array<0.0>,out"}
    )
    np.testing.assert_allclose(model()["in", "out"], 9)
    assert "tap_fwd" in sax.get_ports(model())
    assert "arr" in nl.instances
