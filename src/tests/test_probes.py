"""Probe behavior with kfnetlist topology throughout compilation."""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from kfnetlist import HierarchicalNetlist, Netlist, NetlistPort, PortRef

import sax


def _waveguide(gain: float = 0.8) -> sax.SDict:
    value = jnp.asarray(gain)
    return {("in", "out"): value, ("out", "in"): value / 2}


def _chain(count: int = 2) -> Netlist:
    netlist = Netlist()
    for index in range(count):
        name = f"wg{index + 1}"
        netlist.create_inst(name, "pdk", "waveguide")
        if index:
            netlist.create_net(PortRef(f"wg{index}", "out"), PortRef(name, "in"))
    for name, instance, port in (
        ("in", "wg1", "in"),
        ("out", f"wg{count}", "out"),
    ):
        netlist.create_port(name)
        netlist.create_net(NetlistPort(name), PortRef(instance, port))
    return netlist


def _hierarchy(*, two_levels: bool = False) -> HierarchicalNetlist:
    leaf = _chain()
    top = Netlist()
    if two_levels:
        middle = Netlist()
        middle.create_inst("sub_inst", "pdk", "make_leaf", netlist_id="leaf")
        for port in ("in", "out"):
            middle.create_port(port)
            middle.create_net(NetlistPort(port), PortRef("sub_inst", port))
        top.create_inst("inner_inst", "pdk", "make_middle", netlist_id="middle")
        child_name = "inner_inst"
        children = {"middle": middle, "leaf": leaf}
    else:
        top.create_inst("sub", "pdk", "make_leaf", netlist_id="leaf")
        child_name = "sub"
        children = {"leaf": leaf}
    for port in ("in", "out"):
        top.create_port(port)
        top.create_net(NetlistPort(port), PortRef(child_name, port))
    return HierarchicalNetlist({"top_level": top, **children})


def test_ideal_probe_model_has_through_and_tap_ports() -> None:
    response = sax.models.ideal_probe()
    assert set(sax.get_ports(response)) == {"in", "out", "tap_fwd", "tap_bwd"}
    np.testing.assert_allclose(response["in", "out"], 1)
    np.testing.assert_allclose(response["out", "in"], 1)
    np.testing.assert_allclose(response["tap_fwd", "tap_bwd"], 0)


@pytest.mark.parametrize("target", ["wg1,out", "wg2,in"])
def test_probing_either_side_preserves_asymmetric_transmission(target: str) -> None:
    netlist = _chain()
    plain, _ = sax.circuit(netlist, {"waveguide": _waveguide})
    probed, _ = sax.circuit(netlist, {"waveguide": _waveguide}, probes={"tap": target})
    before, after = plain(), probed()
    assert set(sax.get_ports(after)) == {"in", "out", "tap_fwd", "tap_bwd"}
    np.testing.assert_allclose(after["in", "out"], before["in", "out"])
    np.testing.assert_allclose(after["out", "in"], before["out", "in"])


def test_two_probes_create_four_taps_without_changing_through_path() -> None:
    netlist = _chain(3)
    model, _ = sax.circuit(
        netlist,
        {"waveguide": _waveguide},
        probes={"first": "wg1,out", "second": "wg2,out"},
    )
    response = model()
    assert set(sax.get_ports(response)) == {
        "in",
        "out",
        "first_fwd",
        "first_bwd",
        "second_fwd",
        "second_bwd",
    }
    np.testing.assert_allclose(response["in", "out"], 0.8**3)


@pytest.mark.parametrize("target", ["wg1,in", "wg2,out"])
def test_boundary_probe_keeps_original_port(target: str) -> None:
    model, _ = sax.circuit(_chain(), {"waveguide": _waveguide}, probes={"tap": target})
    response = model()
    assert set(sax.get_ports(response)) == {"in", "out", "tap_fwd", "tap_bwd"}
    np.testing.assert_allclose(response["in", "out"], 0.8**2)


def test_probe_on_unconnected_instance_port_exposes_both_taps() -> None:
    netlist = _chain()
    netlist.create_inst("extra", "pdk", "waveguide")
    model, _ = sax.circuit(
        netlist, {"waveguide": _waveguide}, probes={"tap": "extra,in"}
    )
    assert {"tap_fwd", "tap_bwd"} <= set(sax.get_ports(model()))


@pytest.mark.parametrize("conflict", ["port", "instance"])
def test_probe_generated_names_cannot_collide(conflict: str) -> None:
    netlist = _chain()
    if conflict == "port":
        netlist.create_port("tap_fwd")
        netlist.create_net(NetlistPort("tap_fwd"), PortRef("wg2", "in"))
        message = "conflict"
    else:
        netlist.create_inst("_probe_tap", "pdk", "waveguide")
        message = "conflict"
    with pytest.raises(ValueError, match=message):
        sax.circuit(netlist, {"waveguide": _waveguide}, probes={"tap": "wg1,out"})


@pytest.mark.parametrize("policy", ["warn", "ignore", "as_probes"])
def test_external_port_on_internal_node_follows_policy(policy: str) -> None:
    netlist = _chain()
    netlist.create_port("mid")
    netlist.create_net(NetlistPort("mid"), PortRef("wg1", "out"))
    if policy == "ignore":
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model, _ = sax.circuit(
                netlist, {"waveguide": _waveguide}, on_internal_port=policy
            )
    else:
        with pytest.warns(UserWarning, match="internal node"):
            model, _ = sax.circuit(
                netlist, {"waveguide": _waveguide}, on_internal_port=policy
            )
    response = model()
    ports = set(sax.get_ports(response))
    assert "mid" not in ports
    if policy == "as_probes":
        assert {"mid_fwd", "mid_bwd"} <= ports
    else:
        assert ports == {"in", "out"}
    np.testing.assert_allclose(response["in", "out"], 0.8**2)


@pytest.mark.parametrize("two_levels", [False, True])
def test_hierarchical_probe_traverses_reference_ids(two_levels: bool) -> None:  # noqa: FBT001
    hierarchy = _hierarchy(two_levels=two_levels)
    target = "inner_inst.sub_inst.wg1,out" if two_levels else "sub.wg1,out"
    plain, _ = sax.circuit(hierarchy, {"waveguide": _waveguide})
    probed, _ = sax.circuit(
        hierarchy, {"waveguide": _waveguide}, probes={"tap": target}
    )
    response = probed()
    assert {"tap_fwd", "tap_bwd"} <= set(sax.get_ports(response))
    np.testing.assert_allclose(response["in", "out"], plain()["in", "out"])


def test_hierarchical_boundary_probe_and_invalid_path() -> None:
    hierarchy = _hierarchy()
    probed, _ = sax.circuit(
        hierarchy,
        {"waveguide": _waveguide},
        probes={"tap": "sub.wg1,in"},
    )
    np.testing.assert_allclose(probed()["in", "out"], 0.8**2)
    with pytest.raises(ValueError, match=r"Unknown|not found|does not"):
        sax.circuit(
            hierarchy,
            {"waveguide": _waveguide},
            probes={"tap": "missing.wg1,in"},
        )
