from copy import deepcopy

import numpy as np
import pytest

import sax
from sax.netlists import rename_instances


def _component(gain: float = 0.8) -> sax.SDict:
    return sax.reciprocal({("in0", "out0"): gain})


def _hierarchy(use_nets: bool) -> sax.RecursiveNetlist:
    child = {
        "instances": {"a": {"component": "wg"}, "b": {"component": "wg"}},
        "ports": {"in": "a,in0", "out": "b,out0"},
    }
    top = {
        "instances": {"left": {"component": "child"}, "right": {"component": "child"}},
        "ports": {"in": "left,in", "out": "right,out"},
    }
    if use_nets:
        child["nets"] = [{"p1": "a,out0", "p2": "b,in0", "name": "inside"}]
        top["nets"] = [{"p1": "left,out", "p2": "right,in", "name": "outside"}]
    else:
        child["connections"] = {"a,out0": "b,in0"}
        top["connections"] = {"left,out": "right,in"}
    return sax.into[sax.RecursiveNetlist]({"top_level": top, "child": child})


@pytest.mark.parametrize("use_nets", [True, False])
@pytest.mark.parametrize("backend", ["klu", "fg"])
def test_flatten_equivalent_and_nonmutating(
    use_nets: bool, backend: sax.BackendLike
) -> None:
    original = _hierarchy(use_nets)
    before = deepcopy(original)
    flattened = sax.flatten_netlist(original, sep="__")
    assert original == before
    hierarchical, _ = sax.circuit(original, {"wg": _component}, backend=backend)
    flat, _ = sax.circuit(flattened, {"wg": _component}, backend=backend)
    for key, value in hierarchical().items():
        np.testing.assert_allclose(flat()[key], value)
    np.testing.assert_allclose(flat()["in", "out"], 0.8**4)
    if use_nets:
        assert len(flattened["nets"]) == 3
        assert {link["name"] for link in flattened["nets"]} == {"inside", "outside"}


def test_rename_nets_preserves_metadata_and_input() -> None:
    net = _hierarchy(True)["child"]
    net["nets"][0]["settings"] = {"length": 5.0}
    original = deepcopy(net)
    renamed = rename_instances(net, {"a": "x", "b": "y"})
    assert net == original
    assert renamed["nets"] == [
        {"p1": "x,out0", "p2": "y,in0", "name": "inside", "settings": {"length": 5.0}}
    ]
    assert renamed["ports"] == {"in": "x,in0", "out": "y,out0"}


def test_flatten_same_child_endpoints_and_multiple_links() -> None:
    rec = _hierarchy(True)
    rec["top_level"]["nets"] += [
        {"p1": "left,in", "p2": "left,out", "name": "feedback"},
        {"p1": "left,out", "p2": "right,out", "name": "extra"},
    ]
    flat = sax.flatten_netlist(rec)
    assert {"p1": "left~a,in0", "p2": "left~b,out0", "name": "feedback"} in flat["nets"]
    assert len(flat["nets"]) == 5
    assert all(
        not endpoint.startswith(("left,", "right,"))
        for link in flat["nets"]
        for endpoint in (link["p1"], link["p2"])
    )
