"""Regression using actual gdsfactory/kfactory extraction, outside smoke."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from kfnetlist import HierarchicalNetlist, Netlist, RefNetlistInstance
from kfnetlist.extract import extract

import sax

if TYPE_CHECKING:
    from gdsfactory import Component


def _analytic(length: float = 10.0) -> sax.SDict:
    return {
        ("in0", "out0"): jnp.asarray(length),
        ("out0", "in0"): jnp.asarray(2 * length),
    }


def _wire(length: float = 10.0) -> sax.SDict:
    return {("o1", "o2"): jnp.asarray(length), ("o2", "o1"): jnp.asarray(2 * length)}


def test_extraction_preserves_factory_and_variant_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gf = pytest.importorskip("gdsfactory")
    kf = pytest.importorskip("kfactory")
    previous_pdk = gf.pdk._ACTIVE_PDK  # noqa: SLF001 - restore optional PDK state
    monkeypatch.setattr(gf.pdk, "_ACTIVE_PDK", previous_pdk)
    gf.gpdk.PDK.activate()
    try:

        @gf.cell
        def sax_identity_wrapper(length: float = 10.0) -> Component:
            component = gf.Component()
            wire = component << gf.components.straight(length=length)
            wire.name = "wire"
            component.add_port("in0", port=wire.ports["o1"])
            component.add_port("out0", port=wire.ports["o2"])
            return component

        top = gf.Component()
        for name, length in (("a", 10.0), ("b", 20.0)):
            inst = top << sax_identity_wrapper(length=length)
            inst.name = name
            if name == "b":
                inst.dmovey(30)
            top.add_port(name + "_in", port=inst.ports["in0"])
            top.add_port(name + "_out", port=inst.ports["out0"])
        cells = extract(
            top,
            wrap_kdb_instance=lambda instance: kf.Instance(
                kcl=top.kcl, instance=instance
            ),
            include_placement=False,
        )
        assert all(isinstance(cell, Netlist) for cell in cells.values())
        a, b = cells[top.name].instances["a"], cells[top.name].instances["b"]
        assert isinstance(a, RefNetlistInstance)
        assert isinstance(b, RefNetlistInstance)
        assert a.component == b.component
        assert a.netlist_id != b.netlist_id
        wire_factory = cells[a.netlist_id].instances["wire"].component
        before = {name: cell.to_dict() for name, cell in cells.items()}
        for models in ({a.component: _analytic}, {wire_factory: _wire}):
            model, _ = sax.circuit(
                HierarchicalNetlist(cells), models, top_level_name=top.name
            )
            result = model()
            np.testing.assert_allclose(result["a_in", "a_out"], 10)
            np.testing.assert_allclose(result["b_in", "b_out"], 20)
            np.testing.assert_allclose(result["b_out", "b_in"], 40)
        assert {name: cell.to_dict() for name, cell in cells.items()} == before
    finally:
        if previous_pdk is not None:
            previous_pdk.activate()
