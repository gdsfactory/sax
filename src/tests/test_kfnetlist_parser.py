"""Tests for the kfnetlist parser.

Mirrors the SAX netlist and circuit test patterns but uses kfnetlist input format.
"""

from __future__ import annotations

import json

import pytest

import sax

kfnetlist = pytest.importorskip("kfnetlist")
from kfnetlist import (  # noqa: E402
    Net,
    Netlist,
    NetlistPort,
    PortArrayRef,
    PortRef,
)


# ---------------------------------------------------------------------------
# Helpers — build kfnetlist objects for common topologies
# ---------------------------------------------------------------------------


def _make_mzi() -> Netlist:
    """MZI: mirrors SAMPLE_NETLIST from test_circuit.py / test_netlist.py."""
    nl = Netlist()
    nl.create_port("in0")
    nl.create_port("in1")
    nl.create_port("out0")
    nl.create_port("out1")
    nl.create_inst("lft", kcl="PDK", component="coupler", settings={})
    nl.create_inst("top", kcl="PDK", component="waveguide", settings={})
    nl.create_inst("btm", kcl="PDK", component="waveguide", settings={})
    nl.create_inst("rgt", kcl="PDK", component="coupler", settings={})
    nl.create_net(NetlistPort(name="in0"), PortRef(instance="lft", port="in0"))
    nl.create_net(NetlistPort(name="in1"), PortRef(instance="lft", port="in1"))
    nl.create_net(NetlistPort(name="out0"), PortRef(instance="rgt", port="out0"))
    nl.create_net(NetlistPort(name="out1"), PortRef(instance="rgt", port="out1"))
    nl.create_net(
        PortRef(instance="lft", port="out0"), PortRef(instance="btm", port="in0")
    )
    nl.create_net(
        PortRef(instance="btm", port="out0"), PortRef(instance="rgt", port="in0")
    )
    nl.create_net(
        PortRef(instance="lft", port="out1"), PortRef(instance="top", port="in0")
    )
    nl.create_net(
        PortRef(instance="top", port="out0"), PortRef(instance="rgt", port="in1")
    )
    return nl


def _make_straight_chain() -> Netlist:
    """Two waveguides chained: in0 -> wg1 -> wg2 -> out0."""
    nl = Netlist()
    nl.create_port("in0")
    nl.create_port("out0")
    nl.create_inst(
        "wg1",
        kcl="PDK",
        component="straight",
        settings={"length": 10.0, "loss_dB_cm": 0.0},
    )
    nl.create_inst(
        "wg2",
        kcl="PDK",
        component="straight",
        settings={"length": 20.0, "loss_dB_cm": 0.0},
    )
    nl.create_net(NetlistPort(name="in0"), PortRef(instance="wg1", port="in0"))
    nl.create_net(
        PortRef(instance="wg1", port="out0"), PortRef(instance="wg2", port="in0")
    )
    nl.create_net(PortRef(instance="wg2", port="out0"), NetlistPort(name="out0"))
    return nl


# ---------------------------------------------------------------------------
# Equivalent SAX-native netlists for comparison
# ---------------------------------------------------------------------------

SAX_MZI_NETLIST = {
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


# ---------------------------------------------------------------------------
# Basic parsing tests
# ---------------------------------------------------------------------------


class TestParseKfnetlist:
    """Tests for parse_kfnetlist with various input types."""

    def test_from_object(self) -> None:
        nl = _make_straight_chain()
        recnet = sax.parse_kfnetlist(nl)
        assert "top_level" in recnet
        flat = recnet["top_level"]
        assert "wg1" in flat["instances"]
        assert "wg2" in flat["instances"]
        assert flat["instances"]["wg1"]["component"] == "straight"

    def test_from_dict(self) -> None:
        nl = _make_straight_chain()
        recnet = sax.parse_kfnetlist(nl.to_dict())
        assert "top_level" in recnet
        assert recnet["top_level"]["ports"]["in0"] == "wg1,in0"

    def test_from_json(self) -> None:
        nl = _make_straight_chain()
        recnet = sax.parse_kfnetlist(nl.to_json())
        assert "top_level" in recnet
        assert recnet["top_level"]["ports"]["out0"] == "wg2,out0"

    def test_custom_top_level_name(self) -> None:
        nl = _make_straight_chain()
        recnet = sax.parse_kfnetlist(nl, top_level_name="my_chain")
        assert "my_chain" in recnet
        assert "top_level" not in recnet

    def test_all_input_formats_equivalent(self) -> None:
        nl = _make_mzi()
        from_obj = sax.parse_kfnetlist(nl)
        from_dict = sax.parse_kfnetlist(nl.to_dict())
        from_json = sax.parse_kfnetlist(nl.to_json())
        assert from_obj == from_dict == from_json

    def test_invalid_input_raises(self) -> None:
        with pytest.raises(TypeError):
            sax.parse_kfnetlist(42)

    def test_kcl_field_dropped(self) -> None:
        nl = _make_straight_chain()
        recnet = sax.parse_kfnetlist(nl)
        for inst in recnet["top_level"]["instances"].values():
            assert "kcl" not in inst


# ---------------------------------------------------------------------------
# Instance conversion
# ---------------------------------------------------------------------------


class TestInstanceConversion:
    def test_component_and_settings(self) -> None:
        nl = _make_straight_chain()
        recnet = sax.parse_kfnetlist(nl)
        wg1 = recnet["top_level"]["instances"]["wg1"]
        assert wg1["component"] == "straight"
        assert wg1["settings"]["length"] == 10.0

    def test_empty_settings_preserved(self) -> None:
        nl = Netlist()
        nl.create_inst("x", kcl="P", component="comp", settings={})
        nl.create_port("p")
        nl.create_net(NetlistPort(name="p"), PortRef(instance="x", port="o"))
        recnet = sax.parse_kfnetlist(nl)
        assert recnet["top_level"]["instances"]["x"]["component"] == "comp"

    def test_1x1_array_not_emitted(self) -> None:
        nl = Netlist()
        nl.create_inst("x", kcl="P", component="comp", settings={}, na=1, nb=1)
        nl.create_port("p")
        nl.create_net(NetlistPort(name="p"), PortRef(instance="x", port="o"))
        recnet = sax.parse_kfnetlist(nl)
        assert "array" not in recnet["top_level"]["instances"]["x"]

    def test_array_converted(self) -> None:
        nl = Netlist()
        nl.create_inst("arr", kcl="P", component="comp", settings={}, na=3, nb=2)
        nl.create_port("p")
        nl.create_net(
            NetlistPort(name="p"),
            PortArrayRef(instance="arr", port="o", ia=1, ib=1),
        )
        recnet = sax.parse_kfnetlist(nl)
        arr_inst = recnet["top_level"]["instances"]["arr"]
        assert arr_inst["array"]["columns"] == 3
        assert arr_inst["array"]["rows"] == 2


# ---------------------------------------------------------------------------
# Connection and port mapping
# ---------------------------------------------------------------------------


class TestConnectionMapping:
    def test_internal_connection(self) -> None:
        nl = _make_straight_chain()
        recnet = sax.parse_kfnetlist(nl)
        conns = recnet["top_level"]["connections"]
        assert "wg1,out0" in conns
        assert conns["wg1,out0"] == "wg2,in0"

    def test_external_ports(self) -> None:
        nl = _make_straight_chain()
        recnet = sax.parse_kfnetlist(nl)
        ports = recnet["top_level"]["ports"]
        assert ports["in0"] == "wg1,in0"
        assert ports["out0"] == "wg2,out0"

    def test_mzi_connections(self) -> None:
        nl = _make_mzi()
        recnet = sax.parse_kfnetlist(nl)
        conns = recnet["top_level"]["connections"]
        assert len(conns) == 4
        conn_pairs = {frozenset((k, v)) for k, v in conns.items()}
        assert frozenset(("lft,out0", "btm,in0")) in conn_pairs
        assert frozenset(("btm,out0", "rgt,in0")) in conn_pairs
        assert frozenset(("lft,out1", "top,in0")) in conn_pairs
        assert frozenset(("top,out0", "rgt,in1")) in conn_pairs

    def test_mzi_ports(self) -> None:
        nl = _make_mzi()
        recnet = sax.parse_kfnetlist(nl)
        ports = recnet["top_level"]["ports"]
        assert len(ports) == 4
        assert ports["in0"] == "lft,in0"
        assert ports["in1"] == "lft,in1"
        assert ports["out0"] == "rgt,out0"
        assert ports["out1"] == "rgt,out1"

    def test_no_connections_when_ports_only(self) -> None:
        nl = Netlist()
        nl.create_inst("wg", kcl="P", component="straight", settings={})
        nl.create_port("in0")
        nl.create_port("out0")
        nl.create_net(NetlistPort(name="in0"), PortRef(instance="wg", port="in0"))
        nl.create_net(PortRef(instance="wg", port="out0"), NetlistPort(name="out0"))
        recnet = sax.parse_kfnetlist(nl)
        flat = recnet["top_level"]
        assert "connections" not in flat or flat.get("connections") == {}


# ---------------------------------------------------------------------------
# Array instance port references
# ---------------------------------------------------------------------------


class TestArrayPortRefs:
    def test_collapsed_portref_gets_array_index(self) -> None:
        """kfnetlist collapses PortArrayRef(ia=1,ib=1) to PortRef for all arrays."""
        nl = Netlist()
        nl.create_port("in0")
        nl.create_port("out0")
        nl.create_inst(
            "arr", kcl="P", component="straight", settings={}, na=2, nb=1
        )
        nl.create_net(
            NetlistPort(name="in0"),
            PortArrayRef(instance="arr", port="in0", ia=1, ib=1),
        )
        nl.create_net(
            PortArrayRef(instance="arr", port="out0", ia=2, ib=1),
            NetlistPort(name="out0"),
        )
        recnet = sax.parse_kfnetlist(nl)
        ports = recnet["top_level"]["ports"]
        assert ports["in0"] == "arr<0.0>,in0"
        assert ports["out0"] == "arr<1.0>,out0"

    def test_array_internal_connection(self) -> None:
        nl = Netlist()
        nl.create_port("in0")
        nl.create_port("out0")
        nl.create_inst(
            "arr", kcl="P", component="straight", settings={}, na=2, nb=1
        )
        nl.create_net(
            NetlistPort(name="in0"),
            PortArrayRef(instance="arr", port="in0", ia=1, ib=1),
        )
        nl.create_net(
            PortArrayRef(instance="arr", port="out0", ia=1, ib=1),
            PortArrayRef(instance="arr", port="in0", ia=2, ib=1),
        )
        nl.create_net(
            PortArrayRef(instance="arr", port="out0", ia=2, ib=1),
            NetlistPort(name="out0"),
        )
        recnet = sax.parse_kfnetlist(nl)
        conns = recnet["top_level"]["connections"]
        assert conns["arr<0.0>,out0"] == "arr<1.0>,in0"

    def test_2d_array(self) -> None:
        nl = Netlist()
        nl.create_inst(
            "grid", kcl="P", component="comp", settings={}, na=2, nb=3
        )
        nl.create_port("p")
        nl.create_net(
            NetlistPort(name="p"),
            PortArrayRef(instance="grid", port="o", ia=2, ib=3),
        )
        recnet = sax.parse_kfnetlist(nl)
        assert recnet["top_level"]["ports"]["p"] == "grid<1.2>,o"


# ---------------------------------------------------------------------------
# Recursive netlist parsing
# ---------------------------------------------------------------------------


class TestParseKfnetlistRecursive:
    def test_basic(self) -> None:
        nl_top = _make_straight_chain()
        nl_sub = Netlist()
        nl_sub.create_inst("wg", kcl="P", component="straight", settings={})
        nl_sub.create_port("in0")
        nl_sub.create_port("out0")
        nl_sub.create_net(
            NetlistPort(name="in0"), PortRef(instance="wg", port="in0")
        )
        nl_sub.create_net(
            PortRef(instance="wg", port="out0"), NetlistPort(name="out0")
        )

        recnet = sax.parse_kfnetlist_recursive(
            {"top": nl_top.to_dict(), "sub": nl_sub.to_dict()}
        )
        assert "top" in recnet
        assert "sub" in recnet
        assert recnet["top"]["instances"]["wg1"]["component"] == "straight"
        assert recnet["sub"]["instances"]["wg"]["component"] == "straight"

    def test_from_json_string(self) -> None:
        nl = _make_straight_chain()
        d = {"my_circuit": nl.to_dict()}
        recnet = sax.parse_kfnetlist_recursive(json.dumps(d))
        assert "my_circuit" in recnet

    def test_from_kfnetlist_objects(self) -> None:
        nl1 = _make_straight_chain()
        nl2 = _make_mzi()
        recnet = sax.parse_kfnetlist_recursive({"chain": nl1, "mzi": nl2})
        assert "chain" in recnet
        assert "mzi" in recnet


# ---------------------------------------------------------------------------
# End-to-end circuit tests (mirrors test_circuit.py patterns)
# ---------------------------------------------------------------------------


class TestCircuitFromKfnetlist:
    def test_mzi_circuit(self) -> None:
        """Mirror of test_circuit.test_circuit using kfnetlist input."""
        nl = _make_mzi()
        recnet = sax.parse_kfnetlist(nl)
        mzi, info = sax.circuit(
            netlist=recnet,
            models={
                "coupler": sax.models.coupler_ideal,
                "waveguide": sax.models.straight,
            },
        )
        result = mzi()
        ports = sax.get_ports(result)
        assert sorted(ports) == ["in0", "in1", "out0", "out1"]

    def test_mzi_matches_native_netlist(self) -> None:
        """kfnetlist-parsed MZI should produce identical S-params as native."""
        nl = _make_mzi()
        recnet_kf = sax.parse_kfnetlist(nl)
        models = {
            "coupler": sax.models.coupler_ideal,
            "waveguide": sax.models.straight,
        }
        mzi_kf, _ = sax.circuit(netlist=recnet_kf, models=models)
        mzi_native, _ = sax.circuit(netlist=SAX_MZI_NETLIST, models=models)
        result_kf = mzi_kf()
        result_native = mzi_native()
        for key in result_native:
            assert abs(complex(result_kf[key]) - complex(result_native[key])) < 1e-12

    def test_straight_chain_circuit(self) -> None:
        nl = _make_straight_chain()
        recnet = sax.parse_kfnetlist(nl)
        circuit_fn, info = sax.circuit(
            netlist=recnet,
            models={"straight": sax.models.straight},
        )
        result = circuit_fn()
        ports = sax.get_ports(result)
        assert sorted(ports) == ["in0", "out0"]

    def test_1port_circuit(self) -> None:
        """Mirror of test_circuit.test_1port_circuit."""
        nl = Netlist()
        nl.create_inst("wg1", kcl="PDK", component="waveguide", settings={})
        nl.create_port("in")
        nl.create_net(NetlistPort(name="in"), PortRef(instance="wg1", port="in0"))

        recnet = sax.parse_kfnetlist(nl)
        circuit_fn, _ = sax.circuit(
            recnet, models={"waveguide": sax.models.straight}
        )
        result = circuit_fn()
        ports = sax.get_ports(result)
        assert len(ports) == 1
        assert "in" in ports
        assert ("in", "in") in result

    def test_circuit_with_settings(self) -> None:
        nl = Netlist()
        nl.create_port("in0")
        nl.create_port("out0")
        nl.create_inst(
            "wg1",
            kcl="PDK",
            component="straight",
            settings={"length": 100.0, "loss_dB_cm": 1.0},
        )
        nl.create_net(
            NetlistPort(name="in0"), PortRef(instance="wg1", port="in0")
        )
        nl.create_net(
            PortRef(instance="wg1", port="out0"), NetlistPort(name="out0")
        )
        recnet = sax.parse_kfnetlist(nl)
        circuit_fn, _ = sax.circuit(
            recnet, models={"straight": sax.models.straight}
        )
        result = circuit_fn()
        assert abs(result[("in0", "out0")]) < 1.0


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


class TestJsonRoundTrip:
    def test_dict_round_trip(self) -> None:
        nl = _make_mzi()
        d = nl.to_dict()
        recnet1 = sax.parse_kfnetlist(d)
        nl2 = Netlist.from_dict(d)
        recnet2 = sax.parse_kfnetlist(nl2)
        assert recnet1 == recnet2

    def test_json_round_trip(self) -> None:
        nl = _make_mzi()
        j = nl.to_json()
        recnet1 = sax.parse_kfnetlist(j)
        nl2 = Netlist.from_json(j)
        recnet2 = sax.parse_kfnetlist(nl2)
        assert recnet1 == recnet2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_netlist(self) -> None:
        nl = Netlist()
        recnet = sax.parse_kfnetlist(nl)
        flat = recnet["top_level"]
        assert flat["instances"] == {}
        assert "connections" not in flat or flat.get("connections") == {}

    def test_instances_only_no_nets(self) -> None:
        nl = Netlist()
        nl.create_inst("x", kcl="P", component="comp", settings={})
        recnet = sax.parse_kfnetlist(nl)
        flat = recnet["top_level"]
        assert "x" in flat["instances"]

    def test_non_array_instance_no_angle_brackets(self) -> None:
        nl = _make_straight_chain()
        recnet = sax.parse_kfnetlist(nl)
        for val in recnet["top_level"].get("connections", {}).values():
            assert "<" not in val
        for val in recnet["top_level"].get("ports", {}).values():
            assert "<" not in val

    def test_settings_values_preserved(self) -> None:
        nl = Netlist()
        nl.create_inst(
            "x",
            kcl="P",
            component="comp",
            settings={"a": 1, "b": 2.5, "c": "hello", "d": True},
        )
        nl.create_port("p")
        nl.create_net(NetlistPort(name="p"), PortRef(instance="x", port="o"))
        recnet = sax.parse_kfnetlist(nl)
        s = recnet["top_level"]["instances"]["x"]["settings"]
        assert s["a"] == 1
        assert s["b"] == 2.5
        assert s["c"] == "hello"
        assert s["d"] is True
