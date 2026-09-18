"""Native kfnetlist input, hierarchy identity, and model resolution.

These tests exercise the canonical native path: ``sax.circuit`` accepts native
``kfnetlist.Netlist`` / ``PlacedNetlist`` objects directly, resolves models by
factory name, and falls back to distinct instantiated cells. See
``specs/changes/kfnetlist-canonical.md``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sax
from sax import native

kfnetlist = pytest.importorskip("kfnetlist")
from kfnetlist import (  # noqa: E402
    Netlist,
    PlacedNetlist,
    Placement,
)

NetlistPort = native.NetlistPort
PortRef = native.PortRef


def _transmission(gain: float, wl: float = 1.55) -> sax.SDict:
    return sax.reciprocal({("in0", "out0"): gain * jnp.exp(1j * jnp.asarray(wl))})


def _coupled(wl: float = 1.55, gain: float = 1.0) -> sax.SDict:
    return _transmission(gain, wl)


def _leaf(v: float = 1.0) -> sax.SDict:
    return sax.reciprocal({("in0", "out0"): jnp.asarray(v)})


def _chain(components: list[tuple[str, dict]], ports: bool = True) -> Netlist:
    nl = Netlist()
    for name, spec in components:
        nl.create_inst(
            name,
            kcl="D",
            component=spec["component"],
            settings=spec.get("settings", {}),
        )
    if ports:
        nl.create_port("in0")
        nl.create_port("out0")
    return nl


def _placed_sub(leaf: str, value: float) -> PlacedNetlist:
    sub = PlacedNetlist()
    sub.create_inst(
        "layout",
        kcl="D",
        component=leaf,
        settings={"v": value},
        cell=leaf + "_cell",
        placement=Placement(
            x=0.0,
            y=0.0,
            orientation=0.0,
            mirror=False,
            bbox={"left": 0.0, "bottom": 0.0, "right": 1.0, "top": 1.0},
        ),
    )
    sub.create_port("in0")
    sub.create_port("out0")
    sub.create_net(NetlistPort(name="in0"), PortRef(instance="layout", port="in0"))
    sub.create_net(PortRef(instance="layout", port="out0"), NetlistPort(name="out0"))
    return sub


def _placed_top() -> PlacedNetlist:
    top = PlacedNetlist()
    top.create_inst(
        "a", kcl="D", component="coupled", settings={"gain": 0.2}, cell="cell_a"
    )
    top.create_inst(
        "b", kcl="D", component="coupled", settings={"gain": 0.3}, cell="cell_b"
    )
    for port in ("a_in", "a_out", "b_in", "b_out"):
        top.create_port(port)
    top.create_net(NetlistPort(name="a_in"), PortRef(instance="a", port="in0"))
    top.create_net(PortRef(instance="a", port="out0"), NetlistPort(name="a_out"))
    top.create_net(NetlistPort(name="b_in"), PortRef(instance="b", port="in0"))
    top.create_net(PortRef(instance="b", port="out0"), NetlistPort(name="b_out"))
    return top


# ---------------------------------------------------------------------------
# Issue #120: factory models apply to every parameterization
# ---------------------------------------------------------------------------


def test_factory_model_applies_to_all_variants() -> None:
    cells = {
        "cell_a": _placed_sub("uncoupled", 10.0),
        "cell_b": _placed_sub("uncoupled", 20.0),
        "top": _placed_top(),
    }
    models = {
        "coupled": _coupled,
        "uncoupled": lambda **_: _leaf(0.0),
    }
    model, info = sax.circuit(cells, models, top_level_name="top")
    result = model(wl=1.55)
    np.testing.assert_allclose(result["a_in", "a_out"], 0.2 * jnp.exp(1.55j))
    np.testing.assert_allclose(result["b_in", "b_out"], 0.3 * jnp.exp(1.55j))
    # The analytical model replaces the subtree: child cells are not traversed.
    assert sorted(info.dag.nodes) == ["coupled", "top"]


def test_distinct_cell_fallback_without_factory_model() -> None:
    cells = {
        "cell_a": _placed_sub("leaf_a", 10.0),
        "cell_b": _placed_sub("leaf_b", 20.0),
        "top": _placed_top(),
    }

    def leaf_a(v: float = 1.0) -> sax.SDict:
        return _leaf(v)

    def leaf_b(v: float = 1.0) -> sax.SDict:
        return _leaf(v + 1.0)

    model, info = sax.circuit(
        cells, {"leaf_a": leaf_a, "leaf_b": leaf_b}, top_level_name="top"
    )
    result = model()
    np.testing.assert_allclose(result["a_in", "a_out"], 10.0)
    np.testing.assert_allclose(result["b_in", "b_out"], 21.0)
    assert sorted(info.dag.nodes) == ["cell_a", "cell_b", "leaf_a", "leaf_b", "top"]


def test_cell_specific_override_beats_factory_model() -> None:
    cells = {
        "cell_a": _placed_sub("uncoupled", 0.0),
        "cell_b": _placed_sub("uncoupled", 0.0),
        "top": _placed_top(),
    }
    models = {
        "coupled": lambda **_: _leaf(99.0),
        "cell_a": lambda **_: _leaf(7.0),
        "cell_b": lambda **_: _leaf(8.0),
    }
    model, _ = sax.circuit(cells, models, top_level_name="top")
    result = model()
    np.testing.assert_allclose(result["a_in", "a_out"], 7.0)
    np.testing.assert_allclose(result["b_in", "b_out"], 8.0)


def test_missing_model_reports_factory_and_cell() -> None:
    cells = {
        "cell_a": _placed_sub("leaf_a", 10.0),
        "cell_b": _placed_sub("leaf_b", 20.0),
        "top": _placed_top(),
    }
    with pytest.raises(ValueError, match="Missing models"):
        sax.circuit(cells, {"leaf_a": _leaf}, top_level_name="top")


def test_get_required_circuit_models_native() -> None:
    cells = {
        "cell_a": _placed_sub("leaf_a", 10.0),
        "cell_b": _placed_sub("leaf_b", 20.0),
        "top": _placed_top(),
    }
    required = sax.get_required_circuit_models(
        cells, {"leaf_a": _leaf}, top_level_name="top"
    )
    # Matches legacy behavior: all leaf models are reported; the ``models``
    # argument does not currently filter them out.
    assert "leaf_a" in required
    assert "leaf_b" in required


# ---------------------------------------------------------------------------
# Native input is accepted directly and through JSON
# ---------------------------------------------------------------------------


def test_native_flat_netlist_accepted() -> None:
    nl = Netlist()
    nl.create_port("in0")
    nl.create_port("out0")
    nl.create_inst("wg", kcl="D", component="straight", settings={"length": 10.0})
    nl.create_net(NetlistPort(name="in0"), PortRef(instance="wg", port="in0"))
    nl.create_net(PortRef(instance="wg", port="out0"), NetlistPort(name="out0"))
    model, _ = sax.circuit(nl, {"straight": sax.models.straight})
    assert abs(complex(model()["in0", "out0"])) <= 1.0


def test_native_json_accepted() -> None:
    nl = Netlist()
    nl.create_port("in0")
    nl.create_port("out0")
    nl.create_inst("wg", kcl="D", component="straight", settings={"length": 0.0})
    nl.create_net(NetlistPort(name="in0"), PortRef(instance="wg", port="in0"))
    nl.create_net(PortRef(instance="wg", port="out0"), NetlistPort(name="out0"))
    model, _ = sax.circuit(nl.to_json(), {"straight": sax.models.straight})
    np.testing.assert_allclose(complex(model()["in0", "out0"]), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Legacy adaptation into native
# ---------------------------------------------------------------------------


def test_legacy_flat_adapts_to_native() -> None:
    flat = {
        "instances": {
            "a": {"component": "leaf", "settings": {"v": 2.0}},
            "b": "leaf",
        },
        "connections": {"a,out0": "b,in0"},
        "ports": {"in": "a,in0", "out": "b,out0"},
    }
    nl = native.from_legacy_flat(flat)
    assert isinstance(nl, Netlist)

    def leaf(v: float = 1.0) -> sax.SDict:
        return _leaf(v)

    model, _ = sax.circuit(nl, {"leaf": leaf})
    np.testing.assert_allclose(model()["in", "out"], 2.0)


# ---------------------------------------------------------------------------
# Arrays, settings, JIT, and gradients
# ---------------------------------------------------------------------------


def test_native_array_expansion() -> None:
    nl = Netlist()
    nl.create_port("in0")
    nl.create_inst("arr", kcl="D", component="leaf", settings={"v": 3.0}, na=2, nb=1)
    nl.create_net(
        NetlistPort(name="in0"),
        native.PortArrayRef(instance="arr", port="in0", ia=1, ib=1),
    )
    model, _ = sax.circuit(nl, {"leaf": lambda v=1.0: _leaf(v)})
    result = model()
    assert ("in0", "in0") in result


@pytest.mark.parametrize("backend", ["klu", "fg"])
def test_native_settings_jit_gradient(backend: sax.BackendLike) -> None:
    nl = Netlist()
    nl.create_port("in0")
    nl.create_port("out0")
    nl.create_inst("wg", kcl="D", component="coupled", settings={"gain": 2.0})
    nl.create_net(NetlistPort(name="in0"), PortRef(instance="wg", port="in0"))
    nl.create_net(PortRef(instance="wg", port="out0"), NetlistPort(name="out0"))
    model, _ = sax.circuit(nl, {"coupled": _coupled}, backend=backend)
    wl = jnp.array([1.5, 1.6])
    np.testing.assert_allclose(
        model(wl=wl)["in0", "out0"], 2 * jnp.exp(1j * wl), rtol=1e-5
    )
    np.testing.assert_allclose(
        jax.jit(model)(wl=wl)["in0", "out0"], 2 * jnp.exp(1j * wl), rtol=1e-5
    )
    gradient = jax.grad(lambda g: jnp.abs(model(gain=g)["in0", "out0"]) ** 2)(2.0)
    np.testing.assert_allclose(gradient, 4.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# Probes and internal ports on native input
# ---------------------------------------------------------------------------


def _two_leaves() -> Netlist:
    nl = Netlist()
    nl.create_port("in0")
    nl.create_port("out0")
    nl.create_inst("a", kcl="D", component="wg", settings={"v": 2.0})
    nl.create_inst("b", kcl="D", component="wg", settings={"v": 3.0})
    nl.create_net(NetlistPort(name="in0"), PortRef(instance="a", port="in0"))
    nl.create_net(PortRef(instance="a", port="out0"), PortRef(instance="b", port="in0"))
    nl.create_net(PortRef(instance="b", port="out0"), NetlistPort(name="out0"))
    return nl


def test_native_flat_probe() -> None:
    model, _ = sax.circuit(
        _two_leaves(), {"wg": lambda v=1.0: _leaf(v)}, probes={"mid": "a,out0"}
    )
    result = model()
    np.testing.assert_allclose(complex(result["in0", "out0"]), 6.0)
    # mid_fwd measures the wave travelling into a,out0 (from the b side).
    np.testing.assert_allclose(complex(result["out0", "mid_fwd"]), 3.0)
    assert "mid_bwd" in {k[0] for k in result} | {k[1] for k in result}


def test_native_internal_port_warn_drops_port() -> None:
    nl = _two_leaves()
    nl.create_port("bad")
    nl.create_net(NetlistPort(name="bad"), PortRef(instance="a", port="out0"))
    with pytest.warns(UserWarning, match="internal node"):
        model, _ = sax.circuit(nl, {"wg": lambda v=1.0: _leaf(v)})
    ports = {k[0] for k in model()} | {k[1] for k in model()}
    assert "bad" not in ports


def test_native_hierarchical_probe() -> None:
    from kfnetlist import PlacedNetlist

    sub = PlacedNetlist()
    sub.create_inst("a", kcl="D", component="wg", settings={"v": 2.0}, cell="wg_a")
    sub.create_inst("b", kcl="D", component="wg", settings={"v": 3.0}, cell="wg_b")
    sub.create_port("in0")
    sub.create_port("out0")
    sub.create_net(NetlistPort(name="in0"), PortRef(instance="a", port="in0"))
    sub.create_net(
        PortRef(instance="a", port="out0"), PortRef(instance="b", port="in0")
    )
    sub.create_net(PortRef(instance="b", port="out0"), NetlistPort(name="out0"))

    top = PlacedNetlist()
    top.create_inst("sub", kcl="D", component="sub", cell="sub_cell")
    top.create_port("in0")
    top.create_port("out0")
    top.create_net(NetlistPort(name="in0"), PortRef(instance="sub", port="in0"))
    top.create_net(PortRef(instance="sub", port="out0"), NetlistPort(name="out0"))

    cells = {"sub_cell": sub, "top": top}
    model, _ = sax.circuit(
        cells,
        {"wg": lambda v=1.0: _leaf(v)},
        top_level_name="top",
        probes={"mid": "sub.a,out0"},
    )
    result = model()
    np.testing.assert_allclose(complex(result["in0", "out0"]), 6.0)
    np.testing.assert_allclose(complex(result["out0", "mid_fwd"]), 3.0)


# ---------------------------------------------------------------------------
# PIC YAML → native
# ---------------------------------------------------------------------------


def test_load_pic_yaml_flat() -> None:
    doc = (
        "instances:\n"
        "  a:\n    component: wg\n    settings: {v: 2.0}\n"
        "  b:\n    component: wg\n"
        "connections:\n  a,out0: b,in0\n"
        "ports:\n  in0: a,in0\n  out0: b,out0\n"
    )
    cells, root = native.load_pic_yaml(doc)
    assert isinstance(cells[root], Netlist)
    model, _ = sax.circuit(cells, {"wg": lambda v=1.0: _leaf(v)}, top_level_name=root)
    np.testing.assert_allclose(complex(model()["in0", "out0"]), 2.0)


def test_load_pic_yaml_modules() -> None:
    doc = (
        "modules:\n"
        "  top:\n"
        "    instances:\n      sub:\n        component: child\n"
        "    ports:\n      in: sub,in0\n"
        "  child:\n"
        "    instances:\n      w:\n        component: wg\n"
        "    ports:\n      in0: w,in0\n      out0: w,out0\n"
        "toplevel: top\n"
    )
    cells, root = native.load_pic_yaml(doc)
    assert root == "top"
    assert set(cells) == {"top", "child"}
    assert isinstance(cells["child"], Netlist)


def test_load_native_recursive_netlist(tmp_path) -> None:
    top = tmp_path / "top.pic.yml"
    top.write_text(
        "instances:\n  sub:\n    component: child\nports:\n  in0: sub,in0\n  out0: sub,out0\n"
    )
    child = tmp_path / "child.pic.yml"
    child.write_text(
        "instances:\n  w:\n    component: wg\nports:\n  in0: w,in0\n  out0: w,out0\n"
    )
    cells, root = native.load_native_recursive_netlist(top)
    assert root == "top"
    assert set(cells) == {"top", "child"}
    assert all(isinstance(v, Netlist) for v in cells.values())
    model, _ = sax.circuit(cells, {"wg": lambda v=1.0: _leaf(v)}, top_level_name=root)
    assert ("in0", "out0") in model()


# ---------------------------------------------------------------------------
# Architectural: circuit construction bypasses the old SAX netlist schema
# ---------------------------------------------------------------------------


def test_circuit_bypasses_legacy_kfnetlist_adapter(monkeypatch) -> None:
    """`circuit` must not call the kfnetlist→SAX-dictionary adapter."""
    import sax.parsers.kfnetlist as kfparser

    def _boom(*args, **kwargs):
        raise AssertionError("legacy kfnetlist adapter was called")

    monkeypatch.setattr(kfparser, "_convert_flat", _boom)
    monkeypatch.setattr(kfparser, "parse_kfnetlist", _boom)

    flat = {
        "instances": {"a": {"component": "wg", "settings": {"v": 2.0}}, "b": "wg"},
        "connections": {"a,out0": "b,in0"},
        "ports": {"in0": "a,in0", "out0": "b,out0"},
    }
    model, _ = sax.circuit(flat, {"wg": lambda v=1.0: _leaf(v)})
    np.testing.assert_allclose(complex(model()["in0", "out0"]), 2.0)

    nl = _two_leaves()
    native_model, _ = sax.circuit(nl, {"wg": lambda v=1.0: _leaf(v)})
    np.testing.assert_allclose(complex(native_model()["in0", "out0"]), 6.0)


def test_circuit_does_not_use_legacy_recursive_netlist_helper(monkeypatch) -> None:
    import sax.netlists as netlists

    def _boom(*args, **kwargs):
        raise AssertionError("legacy recursive-netlist normalization was used")

    monkeypatch.setattr(netlists, "netlist", _boom)
    monkeypatch.setattr(netlists, "remove_unused_instances", _boom)
    flat = {
        "instances": {"a": "wg"},
        "connections": {},
        "ports": {"in0": "a,in0", "out0": "a,out0"},
    }
    model, _ = sax.circuit(flat, {"wg": lambda v=1.0: _leaf(v)})
    assert ("in0", "out0") in model()


def test_explicit_native_pic_loader_returns_native() -> None:
    doc = "instances:\n  a:\n    component: wg\nports:\n  in0: a,in0\n  out0: a,out0\n"
    cells, root = native.load_pic_yaml(doc)
    assert isinstance(cells[root], Netlist)
    nl = native.load_native_netlist(doc)
    assert isinstance(nl, Netlist)


# ---------------------------------------------------------------------------
# Native transforms and input isolation
# ---------------------------------------------------------------------------


def _wrap(component: str) -> Netlist:
    nl = Netlist()
    nl.create_port("in0")
    nl.create_port("out0")
    nl.create_inst("x", kcl="D", component=component)
    nl.create_net(NetlistPort(name="in0"), PortRef(instance="x", port="in0"))
    nl.create_net(PortRef(instance="x", port="out0"), NetlistPort(name="out0"))
    return nl


def _leaf_cell() -> Netlist:
    nl = Netlist()
    nl.create_port("in0")
    nl.create_port("out0")
    nl.create_inst("w", kcl="D", component="wg")
    nl.create_net(NetlistPort(name="in0"), PortRef(instance="w", port="in0"))
    nl.create_net(PortRef(instance="w", port="out0"), NetlistPort(name="out0"))
    return nl


def test_native_flatten_netlist() -> None:
    cells = {"top": _wrap("mid"), "mid": _wrap("leaf"), "leaf": _leaf_cell()}
    flat = native.flatten_netlist(cells, "top")
    assert isinstance(flat, Netlist)
    instances, nets, ports = native.lower(flat)
    assert "x__x__w" in instances
    assert ports == {"in0": "x__x__w,in0", "out0": "x__x__w,out0"}


def test_native_flatten_recursive_netlist() -> None:
    cells = {"top": _wrap("mid"), "mid": _wrap("leaf"), "leaf": _leaf_cell()}
    flattened = native.flatten_recursive_netlist(cells)
    assert all(isinstance(v, Netlist) for v in flattened.values())
    assert "x__x__w" in flattened["top"].instances


def test_circuit_does_not_mutate_native_input() -> None:
    nl = _two_leaves()
    before = nl.to_dict()
    model, _ = sax.circuit(nl, {"wg": lambda v=1.0: _leaf(v)}, probes={"mid": "a,out0"})
    del model
    assert nl.to_dict() == before


def test_sax_flatten_netlist_dispatches_native() -> None:
    cells = {"top": _wrap("mid"), "mid": _wrap("leaf"), "leaf": _leaf_cell()}
    flat = sax.flatten_netlist(cells, sep="__")
    assert isinstance(flat, Netlist)
    assert "x__x__w" in native.lower(flat)[0]


def test_native_remove_unused_instances() -> None:
    nl = _two_leaves()
    nl.create_inst("unused", kcl="D", component="wg")
    pruned = native.remove_unused_instances(nl)
    assert "unused" not in pruned.instances
    assert {"a", "b"} <= set(pruned.instances)


def test_native_rename_instances_and_models() -> None:
    nl = _two_leaves()
    renamed = native.rename_instances(nl, {"a": "x", "b": "y"})
    assert {"x", "y"} <= set(renamed.instances)
    assert "a" not in renamed.instances
    endpoints = [
        member.to_dict()
        for net in renamed.nets
        for member in net
        if isinstance(member.to_dict(), dict) and "instance" in member.to_dict()
    ]
    assert all(m["instance"] in {"x", "y"} for m in endpoints)

    remapped = native.rename_models(nl, {"wg": "wg2"})
    assert all(i.component == "wg2" for i in remapped.instances.values())
