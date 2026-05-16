import sax

SAMPLE_NETLIST = {
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


def test_circuit() -> None:
    _mzi, _info = sax.circuit(
        netlist=SAMPLE_NETLIST,
        models={
            "coupler": sax.models.coupler_ideal,
            "waveguide": sax.models.straight,
        },
    )


def test_1port_circuit() -> None:
    """Test that 1-port circuits are supported."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
        },
        "connections": {},
        "ports": {
            "in": "wg1,in0",
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    circuit, _info = sax.circuit(netlist, models)
    result = circuit()

    # Verify the circuit has exactly 1 port
    ports = sax.get_ports(result)
    assert len(ports) == 1
    assert "in" in ports

    # Verify the S-matrix has the expected structure
    assert ("in", "in") in result


def test_circuit_with_portless_subnetlist() -> None:
    """Test that a sub-netlist without a 'ports' key is filtered out (#95)."""
    netlist = {
        "top_level": {
            "instances": {
                "wg": {"component": "waveguide"},
            },
            "connections": {},
            "ports": {
                "in": "wg,in0",
                "out": "wg,out0",
            },
        },
        # Sub-netlist with no "ports" key at all
        "unused_sub": {
            "instances": {
                "x": {"component": "waveguide"},
            },
            "connections": {},
        },
    }

    models = {"waveguide": sax.models.straight}
    circuit, _ = sax.circuit(netlist, models)
    result = circuit()
    assert set(sax.get_ports(result)) == {"in", "out"}


def test_circuit_with_empty_ports_subnetlist() -> None:
    """Test that a sub-netlist with 'ports': {} is filtered out (#95)."""
    netlist = {
        "top_level": {
            "instances": {
                "wg": {"component": "waveguide"},
            },
            "connections": {},
            "ports": {
                "in": "wg,in0",
                "out": "wg,out0",
            },
        },
        # Sub-netlist with empty ports dict
        "unused_sub": {
            "instances": {
                "x": {"component": "waveguide"},
            },
            "connections": {},
            "ports": {},
        },
    }

    models = {"waveguide": sax.models.straight}
    circuit, _ = sax.circuit(netlist, models)
    result = circuit()
    assert set(sax.get_ports(result)) == {"in", "out"}


if __name__ == "__main__":
    print(test_circuit())
    print(test_1port_circuit())
    print(test_circuit_with_portless_subnetlist())
    print(test_circuit_with_empty_ports_subnetlist())
