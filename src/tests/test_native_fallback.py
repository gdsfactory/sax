"""Legacy circuit fallback used when kfnetlist is unavailable (Python 3.11).

The fallback is deprecated: it exists only to preserve supported Python versions
that kfnetlist does not build for. It must remain behaviourally correct.
"""

from __future__ import annotations

import numpy as np
import pytest

import sax
import sax.circuits as circuits


def _wg(v: float = 1.0) -> sax.SDict:
    return sax.reciprocal({("in0", "out0"): np.asarray(v)})


def test_legacy_fallback_without_kfnetlist(monkeypatch) -> None:
    monkeypatch.setattr(circuits, "_NATIVE_AVAILABLE", False)
    net = {
        "instances": {"a": {"component": "wg", "settings": {"v": 2.0}}, "b": "wg"},
        "connections": {"a,out0": "b,in0"},
        "ports": {"in0": "a,in0", "out0": "b,out0"},
    }
    with pytest.warns(DeprecationWarning, match="deprecated legacy netlist path"):
        model, _ = sax.circuit(net, {"wg": _wg})
    np.testing.assert_allclose(complex(model()["in0", "out0"]), 2.0)


def test_legacy_fallback_hierarchy(monkeypatch) -> None:
    monkeypatch.setattr(circuits, "_NATIVE_AVAILABLE", False)
    rec = {
        "top": {
            "instances": {"sub": {"component": "child"}},
            "ports": {"in0": "sub,in0", "out0": "sub,out0"},
        },
        "child": {
            "instances": {"w": {"component": "wg", "settings": {"v": 3.0}}},
            "ports": {"in0": "w,in0", "out0": "w,out0"},
        },
    }
    with pytest.warns(DeprecationWarning):
        model, _ = sax.circuit(rec, {"wg": _wg})
    np.testing.assert_allclose(complex(model()["in0", "out0"]), 3.0)


def test_legacy_fallback_required_models(monkeypatch) -> None:
    monkeypatch.setattr(circuits, "_NATIVE_AVAILABLE", False)
    net = {
        "instances": {"a": {"component": "wg"}},
        "ports": {"in0": "a,in0"},
    }
    with pytest.warns(DeprecationWarning):
        required = sax.get_required_circuit_models(net)
    assert required == ["wg"]
