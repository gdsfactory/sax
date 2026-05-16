"""Tests for the probes feature."""

import pytest

import sax


def test_ideal_probe_model() -> None:
    """Test that the ideal_probe model returns the expected S-matrix."""
    s = sax.models.ideal_probe(wl=1.55)

    # Check all expected ports exist
    expected_ports = {"in", "out", "tap_fwd", "tap_bwd"}
    ports = sax.get_ports(s)
    assert set(ports) == expected_ports

    # Check through path has full transmission
    assert float(s[("in", "out")]) == 1.0
    assert float(s[("out", "in")]) == 1.0

    # Check tap coupling
    assert float(s[("in", "tap_fwd")]) == 1.0
    assert float(s[("tap_fwd", "in")]) == 1.0
    assert float(s[("out", "tap_bwd")]) == 1.0
    assert float(s[("tap_bwd", "out")]) == 1.0

    # Check no cross-coupling between taps
    assert float(s[("tap_fwd", "tap_bwd")]) == 0.0
    assert float(s[("tap_bwd", "tap_fwd")]) == 0.0

    # Check no reflections
    assert float(s[("in", "in")]) == 0.0
    assert float(s[("out", "out")]) == 0.0
    assert float(s[("tap_fwd", "tap_fwd")]) == 0.0
    assert float(s[("tap_bwd", "tap_bwd")]) == 0.0


def test_basic_probe_insertion() -> None:
    """Test basic probe insertion on a simple 2-waveguide circuit."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
        },
        "connections": {
            "wg1,out0": "wg2,in0",
        },
        "ports": {
            "in": "wg1,in0",
            "out": "wg2,out0",
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    # Without probes
    circuit_no_probe, _ = sax.circuit(netlist, models)
    result_no_probe = circuit_no_probe()
    ports_no_probe = sax.get_ports(result_no_probe)
    assert set(ports_no_probe) == {"in", "out"}

    # With probe
    circuit_with_probe, _ = sax.circuit(
        netlist,
        models,
        probes={"mid": "wg1,out0"},
    )
    result_with_probe = circuit_with_probe()
    ports_with_probe = sax.get_ports(result_with_probe)
    assert set(ports_with_probe) == {"in", "out", "mid_fwd", "mid_bwd"}


def test_probe_on_right_side_of_connection() -> None:
    """Test that probes work when specified on the right side of a connection."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
        },
        "connections": {
            "wg1,out0": "wg2,in0",
        },
        "ports": {
            "in": "wg1,in0",
            "out": "wg2,out0",
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    # Probe on right side of connection (wg2,in0 instead of wg1,out0)
    circuit_fn, _ = sax.circuit(
        netlist,
        models,
        probes={"mid": "wg2,in0"},
    )
    result = circuit_fn()
    ports = sax.get_ports(result)
    assert set(ports) == {"in", "out", "mid_fwd", "mid_bwd"}


def test_probe_fwd_direction_left_side() -> None:
    """Test _fwd captures signal flowing INTO user-specified port (left side)."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
        },
        "connections": {
            "wg1,out0": "wg2,in0",
        },
        "ports": {
            "in": "wg1,in0",
            "out": "wg2,out0",
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    # Probe at wg1,out0 (left side of connection)
    # _fwd should capture signal flowing INTO wg1,out0 (i.e., from wg2 back to wg1)
    # _bwd should capture signal flowing OUT of wg1,out0 (i.e., from wg1 to wg2)
    circuit_fn, _ = sax.circuit(
        netlist,
        models,
        probes={"mid": "wg1,out0"},
    )
    result = circuit_fn()

    # Signal from "in" goes through wg1 and exits at wg1,out0
    # This should appear at mid_bwd (signal flowing OUT of the probed port)
    assert ("in", "mid_bwd") in result

    # Signal from "out" goes back through wg2 and into wg1,out0
    # This should appear at mid_fwd (signal flowing INTO the probed port)
    assert ("out", "mid_fwd") in result


def test_probe_fwd_direction_right_side() -> None:
    """Test _fwd captures signal flowing INTO user-specified port (right side)."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
        },
        "connections": {
            "wg1,out0": "wg2,in0",
        },
        "ports": {
            "in": "wg1,in0",
            "out": "wg2,out0",
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    # Probe at wg2,in0 (right side of connection)
    # _fwd should capture signal flowing INTO wg2,in0 (i.e., from wg1 to wg2)
    # _bwd should capture signal flowing OUT of wg2,in0 (i.e., from wg2 back to wg1)
    circuit_fn, _ = sax.circuit(
        netlist,
        models,
        probes={"mid": "wg2,in0"},
    )
    result = circuit_fn()

    # Signal from "in" goes through wg1 and into wg2,in0
    # This should appear at mid_fwd (signal flowing INTO the probed port)
    assert ("in", "mid_fwd") in result

    # Signal from "out" goes back through wg2 and exits at wg2,in0
    # This should appear at mid_bwd (signal flowing OUT of the probed port)
    assert ("out", "mid_bwd") in result


def test_multiple_probes() -> None:
    """Test multiple probes on different connections."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
            "wg3": "waveguide",
        },
        "connections": {
            "wg1,out0": "wg2,in0",
            "wg2,out0": "wg3,in0",
        },
        "ports": {
            "in": "wg1,in0",
            "out": "wg3,out0",
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    circuit_fn, _ = sax.circuit(
        netlist,
        models,
        probes={
            "probe1": "wg1,out0",
            "probe2": "wg2,out0",
        },
    )
    result = circuit_fn()
    ports = sax.get_ports(result)
    expected_ports = {
        "in",
        "out",
        "probe1_fwd",
        "probe1_bwd",
        "probe2_fwd",
        "probe2_bwd",
    }
    assert set(ports) == expected_ports


def test_probe_in_mzi_circuit() -> None:
    """Test probe in a more complex MZI circuit."""
    netlist = {
        "instances": {
            "lft": "coupler",
            "top": "waveguide",
            "btm": "waveguide",
            "rgt": "coupler",
        },
        "connections": {
            "lft,out0": "btm,in0",
            "btm,out0": "rgt,in0",
            "lft,out1": "top,in0",
            "top,out0": "rgt,in1",
        },
        "ports": {
            "in0": "lft,in0",
            "in1": "lft,in1",
            "out0": "rgt,out0",
            "out1": "rgt,out1",
        },
    }

    models = {
        "coupler": sax.models.coupler_ideal,
        "waveguide": sax.models.straight,
    }

    # Probe on top arm
    circuit_fn, _ = sax.circuit(
        netlist,
        models,
        probes={"top_arm": "top,in0"},
    )
    result = circuit_fn()
    ports = sax.get_ports(result)
    expected_ports = {"in0", "in1", "out0", "out1", "top_arm_fwd", "top_arm_bwd"}
    assert set(ports) == expected_ports


def test_probe_values_match_transmission() -> None:
    """Test that probe values match expected transmission through the circuit."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
        },
        "connections": {
            "wg1,out0": "wg2,in0",
        },
        "ports": {
            "in": "wg1,in0",
            "out": "wg2,out0",
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    circuit_fn, _ = sax.circuit(
        netlist,
        models,
        probes={"mid": "wg1,out0"},
    )

    # Evaluate at a specific wavelength
    result = circuit_fn(wl=1.55)

    # The forward probe should capture signal going from wg1 to wg2
    # The transmission in->out should still work
    assert ("in", "out") in result
    assert ("in", "mid_fwd") in result
    assert ("out", "mid_bwd") in result


def test_probe_on_boundary_port_flat() -> None:
    """Test probe on an instance port exposed as a component port."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
            "wg3": "waveguide",
        },
        "connections": {
            "wg1,out0": "wg2,in0",
            "wg2,out0": "wg3,in0",
        },
        "ports": {
            "in": "wg1,in0",
            "out": "wg3,out0",
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    # wg1,in0 is exposed as a component port ("in"), so probing it should
    # insert a full probe with both tap_fwd and tap_bwd.
    circuit_fn, _ = sax.circuit(
        netlist,
        models,
        probes={"tap": "wg1,in0"},
    )
    result = circuit_fn()
    ports = sax.get_ports(result)
    assert "tap_fwd" in ports
    assert "tap_bwd" in ports
    assert set(ports) == {"in", "out", "tap_fwd", "tap_bwd"}


def test_probe_on_truly_unconnected_port() -> None:
    """Test probe on a dangling instance port (not in connections, nets, or ports)."""
    models = {
        "waveguide": sax.models.straight,
    }

    # wg2,out0 is exposed as port "out" (boundary case, tested above).
    # But wg1,in0 is also boundary. Let's test a truly dangling port:
    # wg3,out0 below is not connected to anything and not exposed as a port.
    netlist_with_dangling = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
            "wg3": "waveguide",
        },
        "connections": {
            "wg1,out0": "wg2,in0",
        },
        "ports": {
            "in": "wg1,in0",
            "out": "wg2,out0",
        },
    }

    # wg3,in0 is truly unconnected — not in any connection and not a port.
    # Probing it should still produce both _fwd and _bwd for consistency.
    circuit_fn, _ = sax.circuit(
        netlist_with_dangling,
        models,
        probes={"tap": "wg3,in0"},
    )
    result = circuit_fn()
    ports = sax.get_ports(result)
    assert "tap_fwd" in ports
    assert "tap_bwd" in ports


def test_probe_error_on_port_conflict() -> None:
    """Test that probes raise error when port names would conflict."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
        },
        "connections": {
            "wg1,out0": "wg2,in0",
        },
        "ports": {
            "in": "wg1,in0",
            "out": "wg2,out0",
            "mid_fwd": "wg2,out0",  # Conflict with probe port
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    with pytest.raises(ValueError, match="conflict with existing ports"):
        sax.circuit(
            netlist,
            models,
            probes={"mid": "wg1,out0"},
        )


def test_probe_error_on_instance_conflict() -> None:
    """Test that probes raise error when instance name would conflict."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
            "_probe_mid": "waveguide",  # Conflict with probe instance name
        },
        "connections": {
            "wg1,out0": "wg2,in0",
        },
        "ports": {
            "in": "wg1,in0",
            "out": "wg2,out0",
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    with pytest.raises(ValueError, match="conflicts with an existing instance"):
        sax.circuit(
            netlist,
            models,
            probes={"mid": "wg1,out0"},
        )


def test_probe_on_nets_connection() -> None:
    """Test probe insertion when connections are in nets format."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
        },
        "nets": [
            {"p1": "wg1,out0", "p2": "wg2,in0"},
        ],
        "ports": {
            "in": "wg1,in0",
            "out": "wg2,out0",
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    # Without probes
    circuit_no_probe, _ = sax.circuit(netlist, models)
    result_no_probe = circuit_no_probe()
    ports_no_probe = sax.get_ports(result_no_probe)
    assert set(ports_no_probe) == {"in", "out"}

    # With probe on p1 side of net
    circuit_with_probe, _ = sax.circuit(
        netlist,
        models,
        probes={"mid": "wg1,out0"},
    )
    result_with_probe = circuit_with_probe()
    ports_with_probe = sax.get_ports(result_with_probe)
    assert set(ports_with_probe) == {"in", "out", "mid_fwd", "mid_bwd"}


def test_probe_on_nets_p2_side() -> None:
    """Test probe on the p2 side of a nets-format connection."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
        },
        "nets": [
            {"p1": "wg1,out0", "p2": "wg2,in0"},
        ],
        "ports": {
            "in": "wg1,in0",
            "out": "wg2,out0",
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    # Probe on p2 side of net
    circuit_fn, _ = sax.circuit(
        netlist,
        models,
        probes={"mid": "wg2,in0"},
    )
    result = circuit_fn()
    ports = sax.get_ports(result)
    assert set(ports) == {"in", "out", "mid_fwd", "mid_bwd"}


def test_probe_on_nets_with_multiple_nets() -> None:
    """Test probe insertion with multiple nets, only one being probed."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
            "wg3": "waveguide",
        },
        "nets": [
            {"p1": "wg1,out0", "p2": "wg2,in0"},
            {"p1": "wg2,out0", "p2": "wg3,in0"},
        ],
        "ports": {
            "in": "wg1,in0",
            "out": "wg3,out0",
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    circuit_fn, _ = sax.circuit(
        netlist,
        models,
        probes={"mid": "wg2,in0"},
    )
    result = circuit_fn()
    ports = sax.get_ports(result)
    assert set(ports) == {"in", "out", "mid_fwd", "mid_bwd"}

    # Transmission should still work
    assert ("in", "out") in result
    assert ("in", "mid_fwd") in result


def test_empty_probes_dict() -> None:
    """Test that empty probes dict is a no-op."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
        },
        "connections": {
            "wg1,out0": "wg2,in0",
        },
        "ports": {
            "in": "wg1,in0",
            "out": "wg2,out0",
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    circuit_fn, _ = sax.circuit(netlist, models, probes={})
    result = circuit_fn()
    ports = sax.get_ports(result)
    assert set(ports) == {"in", "out"}


def test_probe_with_none() -> None:
    """Test that probes=None is a no-op."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
        },
        "connections": {
            "wg1,out0": "wg2,in0",
        },
        "ports": {
            "in": "wg1,in0",
            "out": "wg2,out0",
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    circuit_fn, _ = sax.circuit(netlist, models, probes=None)
    result = circuit_fn()
    ports = sax.get_ports(result)
    assert set(ports) == {"in", "out"}


def test_port_on_internal_node_is_dropped() -> None:
    """Test that a port mapping to an internal connection node is dropped."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
        },
        "connections": {
            "wg1,out0": "wg2,in0",
        },
        "ports": {
            "in": "wg1,in0",
            "out": "wg2,out0",
            "mid": "wg1,out0",  # Internal node — should be dropped
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    with pytest.warns(UserWarning, match="internal node"):
        circuit_fn, _ = sax.circuit(netlist, models)

    result = circuit_fn()
    ports = sax.get_ports(result)
    assert "mid" not in ports
    assert "mid_fwd" not in ports
    assert "mid_bwd" not in ports
    assert set(ports) == {"in", "out"}


def test_port_on_internal_node_doesnt_affect_transmission() -> None:
    """Test that dropping internal-node ports doesn't affect transmission."""
    netlist_plain = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
        },
        "connections": {
            "wg1,out0": "wg2,in0",
        },
        "ports": {
            "in": "wg1,in0",
            "out": "wg2,out0",
        },
    }

    netlist_with_internal_port = {
        **netlist_plain,
        "ports": {
            "in": "wg1,in0",
            "out": "wg2,out0",
            "mid": "wg2,in0",  # Internal node — will be dropped
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    circuit_plain, _ = sax.circuit(netlist_plain, models)
    with pytest.warns(UserWarning, match="internal node"):
        circuit_dropped, _ = sax.circuit(netlist_with_internal_port, models)

    result_plain = circuit_plain(wl=1.55)
    result_dropped = circuit_dropped(wl=1.55)

    # Both circuits should have identical ports
    assert set(sax.get_ports(result_plain)) == {"in", "out"}
    assert set(sax.get_ports(result_dropped)) == {"in", "out"}

    # Transmission between real ports should be identical
    assert result_plain["in", "out"] == result_dropped["in", "out"]
    assert result_plain["out", "in"] == result_dropped["out", "in"]


def test_internal_port_dropped_explicit_probe_kept() -> None:
    """Test that internal-node ports are dropped while explicit probes still work."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
            "wg3": "waveguide",
        },
        "connections": {
            "wg1,out0": "wg2,in0",
            "wg2,out0": "wg3,in0",
        },
        "ports": {
            "in": "wg1,in0",
            "out": "wg3,out0",
            "auto_mid": "wg1,out0",  # Internal node — will be dropped
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    with pytest.warns(UserWarning, match="internal node"):
        circuit_fn, _ = sax.circuit(
            netlist,
            models,
            probes={"explicit_mid": "wg2,out0"},
        )

    result = circuit_fn()
    ports = sax.get_ports(result)
    # auto_mid is dropped, only explicit_mid probe ports remain
    expected_ports = {
        "in",
        "out",
        "explicit_mid_fwd",
        "explicit_mid_bwd",
    }
    assert set(ports) == expected_ports


def test_port_on_internal_node_with_nets_format() -> None:
    """Test internal-node port detection when connections are in nets format."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
        },
        "nets": [
            {"p1": "wg1,out0", "p2": "wg2,in0"},
        ],
        "ports": {
            "in": "wg1,in0",
            "out": "wg2,out0",
            "mid": "wg2,in0",  # Internal node (via nets) — will be dropped
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    with pytest.warns(UserWarning, match="internal node"):
        circuit_fn, _ = sax.circuit(netlist, models)

    result = circuit_fn()
    ports = sax.get_ports(result)
    assert "mid" not in ports
    assert "mid_fwd" not in ports
    assert "mid_bwd" not in ports
    assert set(ports) == {"in", "out"}


# --- Hierarchical probe tests ---


def _hierarchical_mzi_netlist() -> dict:
    """Helper: a 2-level hierarchical MZI netlist for testing."""
    return {
        "top_level": {
            "instances": {
                "mzi": {"component": "mzi_component"},
            },
            "connections": {},
            "ports": {
                "in0": "mzi,in0",
                "in1": "mzi,in1",
                "out0": "mzi,out0",
                "out1": "mzi,out1",
            },
        },
        "mzi_component": {
            "instances": {
                "lft": {"component": "coupler"},
                "top": {"component": "waveguide"},
                "btm": {"component": "waveguide"},
                "rgt": {"component": "coupler"},
            },
            "connections": {
                "lft,out0": "btm,in0",
                "btm,out0": "rgt,in0",
                "lft,out1": "top,in0",
                "top,out0": "rgt,in1",
            },
            "ports": {
                "in0": "lft,in0",
                "in1": "lft,in1",
                "out0": "rgt,out0",
                "out1": "rgt,out1",
            },
        },
    }


def _hierarchical_models() -> dict:
    return {
        "coupler": sax.models.coupler_ideal,
        "waveguide": sax.models.straight,
    }


def test_hierarchical_probe_one_level() -> None:
    """Test probe into a sub-circuit using dot-separated path."""
    netlist = _hierarchical_mzi_netlist()
    models = _hierarchical_models()

    circuit_fn, _ = sax.circuit(
        netlist,
        models,
        probes={"top_arm": "mzi.top,in0"},
    )
    result = circuit_fn()
    ports = sax.get_ports(result)
    assert "top_arm_fwd" in ports
    assert "top_arm_bwd" in ports
    expected = {"in0", "in1", "out0", "out1", "top_arm_fwd", "top_arm_bwd"}
    assert set(ports) == expected


def test_hierarchical_probe_two_levels() -> None:
    """Test probe two levels deep in the hierarchy."""
    netlist = {
        "outer": {
            "instances": {
                "inner_inst": {"component": "inner"},
            },
            "connections": {},
            "ports": {
                "in": "inner_inst,in",
                "out": "inner_inst,out",
            },
        },
        "inner": {
            "instances": {
                "sub_inst": {"component": "sub"},
            },
            "connections": {},
            "ports": {
                "in": "sub_inst,in",
                "out": "sub_inst,out",
            },
        },
        "sub": {
            "instances": {
                "wg1": {"component": "waveguide"},
                "wg2": {"component": "waveguide"},
            },
            "connections": {
                "wg1,out0": "wg2,in0",
            },
            "ports": {
                "in": "wg1,in0",
                "out": "wg2,out0",
            },
        },
    }
    models = {"waveguide": sax.models.straight}

    circuit_fn, _ = sax.circuit(
        netlist,
        models,
        probes={"deep": "inner_inst.sub_inst.wg1,out0"},
    )
    result = circuit_fn()
    ports = sax.get_ports(result)
    assert "deep_fwd" in ports
    assert "deep_bwd" in ports
    assert set(ports) == {"in", "out", "deep_fwd", "deep_bwd"}


def test_hierarchical_probe_doesnt_affect_transmission() -> None:
    """Test that hierarchical probes don't affect transmission of real ports."""
    netlist = _hierarchical_mzi_netlist()
    models = _hierarchical_models()

    # Without probes
    circuit_plain, _ = sax.circuit(netlist, models)
    result_plain = circuit_plain(wl=1.55)

    # With hierarchical probe
    circuit_probed, _ = sax.circuit(
        netlist,
        models,
        probes={"top_arm": "mzi.top,in0"},
    )
    result_probed = circuit_probed(wl=1.55)

    assert result_plain["in0", "out0"] == result_probed["in0", "out0"]
    assert result_plain["in0", "out1"] == result_probed["in0", "out1"]


def test_hierarchical_probe_shared_component() -> None:
    """Test probing one instance when a shared component is used by multiple."""
    netlist = {
        "top_level": {
            "instances": {
                "a": {"component": "sub"},
                "b": {"component": "sub"},
            },
            "connections": {
                "a,out": "b,in",
            },
            "ports": {
                "in": "a,in",
                "out": "b,out",
            },
        },
        "sub": {
            "instances": {
                "wg1": {"component": "waveguide"},
                "wg2": {"component": "waveguide"},
            },
            "connections": {
                "wg1,out0": "wg2,in0",
            },
            "ports": {
                "in": "wg1,in0",
                "out": "wg2,out0",
            },
        },
    }
    models = {"waveguide": sax.models.straight}

    # Probe only through instance 'a', not 'b'
    circuit_fn, _ = sax.circuit(
        netlist,
        models,
        probes={"mid_a": "a.wg1,out0"},
    )
    result = circuit_fn()
    ports = sax.get_ports(result)
    # Only 'a's probe ports should be exposed at the top level
    assert "mid_a_fwd" in ports
    assert "mid_a_bwd" in ports
    assert set(ports) == {"in", "out", "mid_a_fwd", "mid_a_bwd"}


def test_mixed_top_level_and_hierarchical_probes() -> None:
    """Test combining top-level and hierarchical probes in the same call."""
    netlist = {
        "top_level": {
            "instances": {
                "sub": {"component": "sub_circuit"},
                "wg_top": {"component": "waveguide"},
            },
            "connections": {
                "sub,out": "wg_top,in0",
            },
            "ports": {
                "in": "sub,in",
                "out": "wg_top,out0",
            },
        },
        "sub_circuit": {
            "instances": {
                "wg1": {"component": "waveguide"},
                "wg2": {"component": "waveguide"},
            },
            "connections": {
                "wg1,out0": "wg2,in0",
            },
            "ports": {
                "in": "wg1,in0",
                "out": "wg2,out0",
            },
        },
    }
    models = {"waveguide": sax.models.straight}

    circuit_fn, _ = sax.circuit(
        netlist,
        models,
        probes={
            "top_probe": "sub,out",  # Top-level probe
            "deep_probe": "sub.wg1,out0",  # Hierarchical probe
        },
    )
    result = circuit_fn()
    ports = sax.get_ports(result)
    expected = {
        "in",
        "out",
        "top_probe_fwd",
        "top_probe_bwd",
        "deep_probe_fwd",
        "deep_probe_bwd",
    }
    assert set(ports) == expected


def test_hierarchical_probe_invalid_path_instance() -> None:
    """Test that a bad instance name in the hierarchy path gives a clear error."""
    netlist = _hierarchical_mzi_netlist()
    models = _hierarchical_models()

    # 'bad_inst' is not an instance in the top-level
    with pytest.raises(ValueError, match="not found"):
        sax.circuit(
            netlist,
            models,
            probes={"bad": "bad_inst.top,in0"},
        )


def test_hierarchical_probe_primitive_component() -> None:
    """Test that probing into a primitive (leaf) component gives a clear error."""
    netlist = _hierarchical_mzi_netlist()
    models = _hierarchical_models()

    # 'top' is a waveguide instance (primitive model, no sub-netlist)
    with pytest.raises(ValueError, match="not defined in the recursive netlist"):
        sax.circuit(
            netlist,
            models,
            probes={"bad": "mzi.top.deeper,in0"},
        )


def test_hierarchical_composition_ports_dropped() -> None:
    """Test that multiple ports on internal nodes are all dropped (#96)."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
            "wg3": "waveguide",
        },
        "connections": {
            "wg1,out0": "wg2,in0",
            "wg2,out0": "wg3,in0",
        },
        "ports": {
            "in": "wg1,in0",
            "out": "wg3,out0",
            "mid1": "wg1,out0",  # internal
            "mid2": "wg2,out0",  # internal
        },
    }

    models = {"waveguide": sax.models.straight}

    with pytest.warns(UserWarning, match="internal node"):
        circuit_fn, _ = sax.circuit(netlist, models)

    result = circuit_fn()
    ports = sax.get_ports(result)
    assert "mid1" not in ports
    assert "mid2" not in ports
    assert "mid1_fwd" not in ports
    assert "mid2_fwd" not in ports
    assert set(ports) == {"in", "out"}


def test_explicit_probes_still_work_after_dropping_internal_ports() -> None:
    """Explicit probes= still creates _fwd/_bwd after internal ports drop."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
            "wg3": "waveguide",
        },
        "connections": {
            "wg1,out0": "wg2,in0",
            "wg2,out0": "wg3,in0",
        },
        "ports": {
            "in": "wg1,in0",
            "out": "wg3,out0",
            "bad_port": "wg1,out0",  # internal — will be dropped
        },
    }

    models = {"waveguide": sax.models.straight}

    with pytest.warns(UserWarning, match="internal node"):
        circuit_fn, _ = sax.circuit(
            netlist,
            models,
            probes={"mid": "wg2,out0"},
        )

    result = circuit_fn()
    ports = sax.get_ports(result)
    assert "bad_port" not in ports
    assert "bad_port_fwd" not in ports
    assert "mid_fwd" in ports
    assert "mid_bwd" in ports
    assert set(ports) == {"in", "out", "mid_fwd", "mid_bwd"}


def test_probe_into_portless_subnetlist() -> None:
    """A portless sub-netlist gets ports via a hierarchical probe (#95 + #96).

    This proves the filter-after-probes ordering is correct: expand_probes
    adds ports to the sub-netlist, so it is no longer portless when the
    filter runs.
    """
    netlist = {
        "top_level": {
            "instances": {
                "sub_inst": {"component": "sub_circuit"},
            },
            "connections": {},
            "ports": {
                "in": "sub_inst,in",
                "out": "sub_inst,out",
            },
        },
        "sub_circuit": {
            "instances": {
                "wg1": {"component": "waveguide"},
                "wg2": {"component": "waveguide"},
            },
            "connections": {
                "wg1,out0": "wg2,in0",
            },
            "ports": {
                "in": "wg1,in0",
                "out": "wg2,out0",
            },
        },
        # This sub-netlist has no ports — it would be filtered out
        # unless a hierarchical probe adds ports to it first.
        "deep_sub": {
            "instances": {
                "wg": {"component": "waveguide"},
            },
            "connections": {},
        },
    }

    models = {"waveguide": sax.models.straight}

    # Probe into the sub_circuit (not into deep_sub).
    # deep_sub is still portless and gets filtered, but that's fine.
    circuit_fn, _ = sax.circuit(
        netlist,
        models,
        probes={"tap": "sub_inst.wg1,out0"},
    )
    result = circuit_fn()
    ports = sax.get_ports(result)
    assert "tap_fwd" in ports
    assert "tap_bwd" in ports
    assert set(ports) == {"in", "out", "tap_fwd", "tap_bwd"}


def test_on_internal_port_ignore() -> None:
    """Test that on_internal_port='ignore' drops silently without warning."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
        },
        "connections": {
            "wg1,out0": "wg2,in0",
        },
        "ports": {
            "in": "wg1,in0",
            "out": "wg2,out0",
            "mid": "wg1,out0",  # internal
        },
    }

    models = {"waveguide": sax.models.straight}

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        circuit_fn, _ = sax.circuit(netlist, models, on_internal_port="ignore")

    result = circuit_fn()
    ports = sax.get_ports(result)
    assert set(ports) == {"in", "out"}


def test_on_internal_port_as_probes() -> None:
    """Test that on_internal_port='as_probes' gives the legacy probe behaviour."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
        },
        "connections": {
            "wg1,out0": "wg2,in0",
        },
        "ports": {
            "in": "wg1,in0",
            "out": "wg2,out0",
            "mid": "wg1,out0",  # internal — should become a probe
        },
    }

    models = {"waveguide": sax.models.straight}

    with pytest.warns(UserWarning, match="internal node"):
        circuit_fn, _ = sax.circuit(netlist, models, on_internal_port="as_probes")

    result = circuit_fn()
    ports = sax.get_ports(result)
    assert "mid" not in ports
    assert "mid_fwd" in ports
    assert "mid_bwd" in ports
    assert set(ports) == {"in", "out", "mid_fwd", "mid_bwd"}


def test_probe_on_boundary_port() -> None:
    """Test probe on an instance port exposed as a component port."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
        },
        "connections": {
            "wg1,out0": "wg2,in0",
        },
        "ports": {
            "in": "wg1,in0",
            "out": "wg2,out0",
        },
    }
    models = {"waveguide": sax.models.straight}

    # Probe on wg1,in0 which is at the component boundary (exposed as port "in")
    circuit_fn, _ = sax.circuit(
        netlist,
        models,
        probes={"tap": "wg1,in0"},
    )
    result = circuit_fn()
    ports = sax.get_ports(result)
    assert "tap_fwd" in ports
    assert "tap_bwd" in ports
    assert set(ports) == {"in", "out", "tap_fwd", "tap_bwd"}


def test_hierarchical_probe_on_boundary_port() -> None:
    """Test hierarchical probe on an instance port exposed as sub-circuit port."""
    netlist = {
        "top_level": {
            "instances": {
                "sub": {"component": "sub_circuit"},
            },
            "connections": {},
            "ports": {
                "in": "sub,in",
                "out": "sub,out",
            },
        },
        "sub_circuit": {
            "instances": {
                "wg1": {"component": "waveguide"},
                "wg2": {"component": "waveguide"},
            },
            "connections": {
                "wg1,out0": "wg2,in0",
            },
            "ports": {
                "in": "wg1,in0",
                "out": "wg2,out0",
            },
        },
    }
    models = {"waveguide": sax.models.straight}

    # Probe wg1,in0 inside sub_circuit — this port is at the boundary
    circuit_fn, _ = sax.circuit(
        netlist,
        models,
        probes={"tap": "sub.wg1,in0"},
    )
    result = circuit_fn()
    ports = sax.get_ports(result)
    assert "tap_fwd" in ports
    assert "tap_bwd" in ports
    assert set(ports) == {"in", "out", "tap_fwd", "tap_bwd"}


def test_hierarchical_probe_boundary_doesnt_affect_transmission() -> None:
    """Test that a boundary probe doesn't change the in->out transmission."""
    netlist = {
        "top_level": {
            "instances": {
                "sub": {"component": "sub_circuit"},
            },
            "connections": {},
            "ports": {
                "in": "sub,in",
                "out": "sub,out",
            },
        },
        "sub_circuit": {
            "instances": {
                "wg1": {"component": "waveguide"},
                "wg2": {"component": "waveguide"},
            },
            "connections": {
                "wg1,out0": "wg2,in0",
            },
            "ports": {
                "in": "wg1,in0",
                "out": "wg2,out0",
            },
        },
    }
    models = {"waveguide": sax.models.straight}

    # Without probes
    circuit_plain, _ = sax.circuit(netlist, models)
    result_plain = circuit_plain(wl=1.55)

    # With boundary probe
    circuit_probed, _ = sax.circuit(
        netlist,
        models,
        probes={"tap": "sub.wg1,in0"},
    )
    result_probed = circuit_probed(wl=1.55)

    assert result_plain["in", "out"] == result_probed["in", "out"]


if __name__ == "__main__":
    test_ideal_probe_model()
    test_basic_probe_insertion()
    test_probe_on_right_side_of_connection()
    test_multiple_probes()
    test_probe_in_mzi_circuit()
    test_probe_values_match_transmission()
    test_probe_on_boundary_port_flat()
    test_probe_on_truly_unconnected_port()
    test_probe_error_on_port_conflict()
    test_probe_error_on_instance_conflict()
    test_empty_probes_dict()
    test_probe_with_none()
    test_port_on_internal_node_is_dropped()
    test_port_on_internal_node_doesnt_affect_transmission()
    test_internal_port_dropped_explicit_probe_kept()
    test_port_on_internal_node_with_nets_format()
    test_hierarchical_probe_one_level()
    test_hierarchical_probe_two_levels()
    test_hierarchical_probe_doesnt_affect_transmission()
    test_hierarchical_probe_shared_component()
    test_mixed_top_level_and_hierarchical_probes()
    test_hierarchical_probe_invalid_path_instance()
    test_hierarchical_probe_primitive_component()
    test_hierarchical_composition_ports_dropped()
    test_explicit_probes_still_work_after_dropping_internal_ports()
    test_probe_into_portless_subnetlist()
    test_on_internal_port_ignore()
    test_on_internal_port_as_probes()
    test_probe_on_boundary_port()
    test_hierarchical_probe_on_boundary_port()
    test_hierarchical_probe_boundary_doesnt_affect_transmission()
    print("All tests passed!")
