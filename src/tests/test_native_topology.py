"""Circuit topology and probe policies on plain kfnetlist objects."""

import jax.numpy as jnp
import numpy as np
import pytest
from kfnetlist import HierarchicalNetlist, Netlist, NetlistPort, PortArrayRef, PortRef

import sax


def _gain(gain: float = 1.0) -> sax.SDict:
    return {("in", "out"): jnp.asarray(gain), ("out", "in"): jnp.asarray(gain) / 2}


def _single(*, component: str = "gain", name: str = "a") -> Netlist:
    netlist = Netlist()
    netlist.create_inst(name, "pdk", component)
    for port in ("in", "out"):
        netlist.create_port(port)
        netlist.create_net(NetlistPort(port), PortRef(name, port))
    return netlist


def _parent(*, ports: bool = True) -> HierarchicalNetlist:
    top = Netlist()
    top.create_inst("sub", "pdk", "make_child", netlist_id="child")
    if ports:
        for port in ("in", "out"):
            top.create_port(port)
            top.create_net(NetlistPort(port), PortRef("sub", port))
    return HierarchicalNetlist({"top_level": top, "child": _single()})


def test_disconnected_models_are_not_required_and_input_is_unchanged() -> None:
    netlist = _single()
    netlist.create_inst("unused", "pdk", "missing")
    before = netlist.to_dict()
    model, info = sax.circuit(netlist, {"gain": _gain})
    np.testing.assert_allclose(model()["in", "out"], 1.0)
    assert set(info.dag) == {"top_level", "gain"}
    assert sax.get_required_circuit_models(netlist) == ["gain"]
    assert netlist.to_dict() == before


def test_hierarchical_probe_checks_parent_port_conflict_and_opaque_model() -> None:
    hierarchy = _parent()
    top = hierarchy["top_level"]
    top.create_port("tap_fwd")
    top.create_net(NetlistPort("tap_fwd"), PortRef("sub", "extra"))
    with pytest.raises(ValueError, match="conflict with existing ports"):
        sax.circuit(hierarchy, {"gain": _gain}, probes={"tap": "sub.a,out"})

    with pytest.raises(ValueError, match="Only sub-circuits"):
        sax.circuit(_parent(), {"child": _gain}, probes={"tap": "sub.a,out"})


def test_probe_path_through_portless_hierarchy_survives_pruning() -> None:
    hierarchy = _parent(ports=False)
    child = hierarchy["child"]
    child.create_inst("unused", "pdk", "missing")
    model, _ = sax.circuit(hierarchy, {"gain": _gain}, probes={"tap": "sub.a,out"})
    assert set(sax.get_ports(model())) == {"tap_fwd", "tap_bwd"}


def test_dropping_every_internal_port_has_useful_error() -> None:
    netlist = Netlist()
    for name in ("a", "b"):
        netlist.create_inst(name, "pdk", "gain")
    netlist.create_net(PortRef("a", "out"), PortRef("b", "in"))
    netlist.create_port("mid")
    netlist.create_net(NetlistPort("mid"), PortRef("a", "out"))
    with pytest.raises(ValueError, match="at least 1 port"):
        sax.circuit(netlist, {"gain": _gain}, on_internal_port="ignore")


@pytest.mark.parametrize("case", ["unconnected", "external_only", "alias", "junction"])
def test_unsupported_topology_is_explicit(case: str) -> None:
    netlist = _single()
    if case == "unconnected":
        netlist.create_port("floating")
        message = "no instance connection"
    elif case == "external_only":
        netlist.create_port("floating")
        netlist.create_net(NetlistPort("floating"))
        message = "External-only"
    elif case == "alias":
        netlist.create_port("alias")
        netlist.create_net(NetlistPort("alias"), PortRef("a", "in"))
        message = "aliases"
    else:
        for name in ("b", "c"):
            netlist.create_inst(name, "pdk", "gain")
        netlist.create_net(*(PortRef(name, "out") for name in ("a", "b", "c")))
        message = "junction model"
    with pytest.raises(ValueError, match=message):
        sax.circuit(netlist, {"gain": _gain})


def test_array_probe_preserves_settings_and_input() -> None:
    netlist = Netlist()
    netlist.create_inst("arr", "pdk", "gain", {"gain": 3}, na=2)
    netlist.create_net(
        PortArrayRef("arr", "out", 1, 1), PortArrayRef("arr", "in", 2, 1)
    )
    for port, index in (("in", 1), ("out", 2)):
        netlist.create_port(port)
        netlist.create_net(NetlistPort(port), PortArrayRef("arr", port, index, 1))
    before = netlist.to_dict()
    model, _ = sax.circuit(netlist, {"gain": _gain}, probes={"tap": "arr<0.0>,out"})
    np.testing.assert_allclose(model()["in", "out"], 9)
    assert "tap_fwd" in sax.get_ports(model())
    assert netlist.to_dict() == before


@pytest.mark.parametrize("count", [1, 2])
def test_hierarchical_probe_array_names_match_expansion(count: int) -> None:
    top = Netlist()
    top.create_inst("sub", "pdk", "make_child", na=count, netlist_id="child")
    for port in ("in", "out"):
        top.create_port(port)
        top.create_net(NetlistPort(port), PortArrayRef("sub", port, 1, 1))
    hierarchy = HierarchicalNetlist({"top_level": top, "child": _single()})
    model, _ = sax.circuit(
        hierarchy, {"gain": _gain}, probes={"tap": "sub<0.0>.a<0.0>,out"}
    )
    assert {"tap_fwd", "tap_bwd"} <= set(sax.get_ports(model()))
    np.testing.assert_allclose(model()["in", "out"], 1)
    with pytest.raises(ValueError, match=r"outside.*array dimensions"):
        sax.circuit(hierarchy, {"gain": _gain}, probes={"tap": f"sub<{count}.0>.a,out"})
