"""SAX's plain kfnetlist hierarchy contract during the reference migration."""

import jax.numpy as jnp
import numpy as np
import pytest
from kfnetlist import (
    HierarchicalNetlist,
    Netlist,
    NetlistInstance,
    NetlistPort,
    PlacedNetlist,
    PortRef,
)

import sax


def _model(value: float = 1.0) -> sax.SDict:
    return {("in", "out"): jnp.asarray(value), ("out", "in"): jnp.asarray(value) / 2}


def _document() -> dict[str, Netlist]:
    child = Netlist()
    child.create_inst("leaf", "pdk", "waveguide", {"value": 3.0})
    for port in ("in", "out"):
        child.create_port(port)
        child.create_net(NetlistPort(name=port), PortRef("leaf", port))
    top = Netlist()
    top.create_inst("a", "pdk", "make_child", netlist_id="child")
    top.create_inst("b", "pdk", "make_child", netlist_id="child")
    for name, instance, port in (
        ("a_in", "a", "in"),
        ("a_out", "a", "out"),
        ("b_in", "b", "in"),
        ("b_out", "b", "out"),
    ):
        top.create_port(name)
        top.create_net(NetlistPort(name=name), PortRef(instance, port))
    return {"top_level": top, "child": child}


def test_plain_reference_traversal() -> None:
    model, _ = sax.circuit(HierarchicalNetlist(_document()), {"waveguide": _model})
    result = model()
    np.testing.assert_allclose(result["a_in", "a_out"], 3)
    np.testing.assert_allclose(result["b_out", "b_in"], 1.5)


def test_hierarchical_netlist_object_is_a_circuit_input() -> None:
    document = HierarchicalNetlist(_document())
    assert sax.HierarchicalNetlist is HierarchicalNetlist
    model, _ = sax.circuit(document, {"waveguide": _model})
    np.testing.assert_allclose(model()["a_in", "a_out"], 3)
    document["top_level"].create_inst(
        "broken", "pdk", "make_other", netlist_id="missing"
    )
    with pytest.raises(ValueError, match=r"missing.*does not exist"):
        sax.circuit(document, {"waveguide": _model})


def test_factory_model_replaces_referenced_child() -> None:
    model, _ = sax.circuit(
        HierarchicalNetlist(_document()), {"make_child": lambda: _model(7)}
    )
    np.testing.assert_allclose(model()["a_in", "a_out"], 7)


def test_missing_child_reference_is_not_factory_fallback() -> None:
    cells = _document()
    cells["top_level"].create_inst("bad", "pdk", "waveguide", netlist_id="missing")
    with pytest.raises(ValueError, match=r"missing|does not exist"):
        sax.circuit(HierarchicalNetlist(cells), {"waveguide": _model})


def test_reference_override_precedes_factory_and_preserves_other_variant() -> None:
    cells = _document()
    second = Netlist.from_dict(cells["child"].to_dict())
    cells["other"] = second
    top = cells["top_level"]
    top.create_inst("c", "pdk", "make_child", netlist_id="other")
    for port in ("in", "out"):
        top.create_port(f"c_{port}")
        top.create_net(NetlistPort(name=f"c_{port}"), PortRef("c", port))
    models = {
        "child": lambda: _model(7),
        "make_child": lambda: _model(2),
    }
    model, _ = sax.circuit(HierarchicalNetlist(cells), models)
    np.testing.assert_allclose(model()["a_in", "a_out"], 7)
    np.testing.assert_allclose(model()["b_in", "b_out"], 7)
    np.testing.assert_allclose(model()["c_in", "c_out"], 2)


def test_hierarchical_probe_uses_explicit_reference() -> None:
    model, _ = sax.circuit(
        HierarchicalNetlist(_document()),
        {"waveguide": _model},
        probes={"tap": "a.leaf,in"},
    )
    result = model()
    assert any("tap_fwd" in pair for pair in result)
    assert any("tap_bwd" in pair for pair in result)


def test_reference_cycle_rejected_before_model_substitution() -> None:
    cells = _document()
    cells["child"].create_inst("loop", "pdk", "make_top", netlist_id="top_level")
    with pytest.raises(ValueError, match="cyclic"):
        sax.circuit(HierarchicalNetlist(cells), {"make_child": _model})


def test_public_netlist_types_use_kfnetlist() -> None:
    assert sax.Netlist is Netlist
    assert sax.saxtypes.Netlist is Netlist
    assert sax.NetlistInstance is NetlistInstance
    assert sax.HierarchicalNetlist is HierarchicalNetlist


@pytest.mark.parametrize(
    "invalid", [{"instances": {}}, "{}", _document(), PlacedNetlist()]
)
def test_circuit_rejects_non_kfnetlist_inputs(invalid: object) -> None:
    with pytest.raises(TypeError, match=r"kfnetlist\.Netlist"):
        sax.circuit(invalid, {"waveguide": _model})  # type: ignore[arg-type]


def test_mosaic_parser_emits_kfnetlist_directly() -> None:
    parsed = sax.parse_mosaic(
        {
            "a": {
                "type": "cell",
                "model": "waveguide",
                "props": {"value": 2},
                "nets": {"in": "input", "out": "output"},
            }
        }
    )
    assert type(parsed) is Netlist
    model, _ = sax.circuit(parsed, {"waveguide": _model})
    np.testing.assert_allclose(model()["input", "output"], 2)
