import numpy as np
import pytest
from kfnetlist import HierarchicalNetlist, Netlist, NetlistPort, PortRef

import sax


@pytest.mark.parametrize(
    "components",
    [("top_level",), ("top_level", "child"), ("top_level", "child", "grandchild")],
)
def test_dependency_cycles_have_explicit_diagnostic(
    components: tuple[str, ...],
) -> None:
    netlists = {}
    for index, name in enumerate(components):
        netlist = Netlist()
        netlist.create_inst(
            "sub",
            "pdk",
            "subcircuit",
            netlist_id=components[(index + 1) % len(components)],
        )
        for port in ("in", "out"):
            netlist.create_port(port)
            netlist.create_net(NetlistPort(port), PortRef("sub", port))
        netlists[name] = netlist
    with pytest.raises(ValueError, match="cyclic"):
        HierarchicalNetlist(netlists)


def test_optical_feedback_is_not_a_hierarchy_cycle() -> None:
    net = Netlist()
    net.create_inst("c", "pdk", "coupler")
    net.create_inst("w", "pdk", "waveguide")
    net.create_net(PortRef("c", "out1"), PortRef("w", "in0"))
    net.create_net(PortRef("w", "out0"), PortRef("c", "in1"))
    for name, port in (("in", "in0"), ("out", "out0")):
        net.create_port(name)
        net.create_net(NetlistPort(name), PortRef("c", port))
    models = {"coupler": sax.models.coupler_ideal, "waveguide": sax.models.straight}
    klu, _ = sax.circuit(net, models, backend="klu")
    fg, _ = sax.circuit(net, models, backend="fg")
    for key, value in klu().items():
        assert np.isfinite(value).all()
        np.testing.assert_allclose(value, fg()[key], atol=1e-10)
