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
        nl.create_inst(name, kcl="D", component=spec["component"], settings=spec.get("settings", {}))
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
            x=0.0, y=0.0, orientation=0.0, mirror=False,
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
    top.create_inst("a", kcl="D", component="coupled", settings={"gain": 0.2}, cell="cell_a")
    top.create_inst("b", kcl="D", component="coupled", settings={"gain": 0.3}, cell="cell_b")
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
    nl.create_net(NetlistPort(name="in0"), native.PortArrayRef(instance="arr", port="in0", ia=1, ib=1))
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
    gradient = jax.grad(
        lambda g: jnp.abs(model(gain=g)["in0", "out0"]) ** 2
    )(2.0)
    np.testing.assert_allclose(gradient, 4.0, rtol=1e-5)
