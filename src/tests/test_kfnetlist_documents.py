"""KFNetlist documents remain the circuit input through serialization."""

import numpy as np
from kfnetlist import HierarchicalNetlist, Netlist, NetlistPort, PortRef

import sax


def _chain() -> Netlist:
    netlist = Netlist()
    netlist.create_inst("a", "pdk", "straight", {"length": 10.0})
    netlist.create_inst("b", "pdk", "straight", {"length": 20.0})
    netlist.create_net(PortRef("a", "out0"), PortRef("b", "in0"))
    for name, instance, port in (
        ("in", "a", "in0"),
        ("out", "b", "out0"),
    ):
        netlist.create_port(name)
        netlist.create_net(NetlistPort(name), PortRef(instance, port))
    return netlist


def _child() -> Netlist:
    child = Netlist()
    child.create_inst("wg", "pdk", "straight", {"length": 10.0})
    for name in ("in0", "out0"):
        child.create_port(name)
        child.create_net(NetlistPort(name), PortRef("wg", name))
    return child


def _parent() -> HierarchicalNetlist:
    top = Netlist()
    top.create_inst("sub", "pdk", "make_child", netlist_id="child")
    for name, child_port in (("in", "in0"), ("out", "out0")):
        top.create_port(name)
        top.create_net(NetlistPort(name), PortRef("sub", child_port))
    return HierarchicalNetlist({"top_level": top, "child": _child()})


def test_plain_netlist_dict_and_json_roundtrip_simulate() -> None:
    original = _chain()
    converted = (
        original,
        Netlist.from_dict(original.to_dict()),
        Netlist.from_json(original.to_json()),
    )
    expected = None
    for netlist in converted:
        model, _ = sax.circuit(netlist, {"straight": sax.models.straight})
        value = model()["in", "out"]
        if expected is None:
            expected = value
        np.testing.assert_allclose(value, expected)
        assert netlist.instances["a"].settings["length"] == 10.0


def test_hierarchical_json_roundtrip_keeps_reference() -> None:
    document = _parent()
    loaded = HierarchicalNetlist.from_json(document.to_json())
    assert loaded["top_level"].instances["sub"].netlist_id == "child"
    model, _ = sax.circuit(loaded, {"straight": sax.models.straight})
    assert ("in", "out") in model()


def test_one_port_netlist_simulates() -> None:
    netlist = Netlist()
    netlist.create_inst("wg", "pdk", "straight")
    netlist.create_port("in")
    netlist.create_net(NetlistPort("in"), PortRef("wg", "in0"))
    model, _ = sax.circuit(netlist, {"straight": sax.models.straight})
    assert set(sax.get_ports(model())) == {"in"}


def test_unused_child_document_does_not_require_model() -> None:
    document = _parent()
    unused = Netlist()
    unused.create_inst("unknown", "pdk", "missing")
    document["unused"] = unused
    model, _ = sax.circuit(document, {"straight": sax.models.straight})
    assert ("in", "out") in model()


def test_reference_model_can_replace_child_after_roundtrip() -> None:
    document = HierarchicalNetlist.from_dict(_parent().to_dict())
    model, _ = sax.circuit(document, {"child": lambda: {("in0", "out0"): 3.0}})
    np.testing.assert_allclose(model()["in", "out"], 3.0)
