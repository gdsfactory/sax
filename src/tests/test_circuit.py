"""Circuits constructed directly from kfnetlist topology."""

from kfnetlist import HierarchicalNetlist, Netlist, NetlistPort, PortRef

import sax


def _mzi() -> Netlist:
    netlist = Netlist()
    for name, component in (
        ("lft", "coupler"),
        ("top", "waveguide"),
        ("btm", "waveguide"),
        ("rgt", "coupler"),
    ):
        netlist.create_inst(name, "pdk", component)
    for first, second in (
        (("lft", "out0"), ("btm", "in0")),
        (("btm", "out0"), ("rgt", "in0")),
        (("lft", "out1"), ("top", "in0")),
        (("top", "out0"), ("rgt", "in1")),
    ):
        netlist.create_net(PortRef(*first), PortRef(*second))
    for name, instance, port in (
        ("in0", "lft", "in0"),
        ("in1", "lft", "in1"),
        ("out0", "rgt", "out0"),
        ("out1", "rgt", "out1"),
    ):
        netlist.create_port(name)
        netlist.create_net(NetlistPort(name), PortRef(instance, port))
    return netlist


def test_mzi_circuit() -> None:
    model, _ = sax.circuit(
        _mzi(),
        {"coupler": sax.models.coupler_ideal, "waveguide": sax.models.straight},
    )
    assert set(sax.get_ports(model())) == {"in0", "in1", "out0", "out1"}


def test_unused_portless_child_is_not_built() -> None:
    top = Netlist()
    top.create_inst("wg", "pdk", "waveguide")
    for port, child_port in (("in", "in0"), ("out", "out0")):
        top.create_port(port)
        top.create_net(NetlistPort(port), PortRef("wg", child_port))
    unused = Netlist()
    unused.create_inst("x", "pdk", "unknown")
    hierarchy = HierarchicalNetlist({"top_level": top, "unused": unused})
    model, _ = sax.circuit(hierarchy, {"waveguide": sax.models.straight})
    assert set(sax.get_ports(model())) == {"in", "out"}
