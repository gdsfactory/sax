from functools import partial

import jax.numpy as jnp
import numpy as np
import pytest
from kfnetlist import NetlistPort, PlacedNetlist, PortRef

import sax
from sax import native


def _model(gain: float = 1.0) -> sax.SDict:
    return {("in", "out"): jnp.asarray(gain), ("out", "in"): jnp.asarray(gain) / 2}


def _two_libraries() -> PlacedNetlist:
    nl = PlacedNetlist()
    for name, library in (("a", "PDK_A"), ("b", "PDK_B")):
        nl.create_inst(name, kcl=library, component="coupler2", cell=f"cell_{name}")
        for port in ("in", "out"):
            nl.create_port(f"{name}_{port}")
            nl.create_net(
                NetlistPort(name=f"{name}_{port}"), PortRef(instance=name, port=port)
            )
    return nl


@pytest.mark.parametrize("backend", ["klu", "fg"])
def test_qualified_factories_and_cell_override(backend: sax.BackendLike) -> None:
    nl = _two_libraries()
    models = {
        "PDK_A::coupler2": partial(_model, gain=2.0),
        "PDK_B::coupler2": partial(_model, gain=3.0),
    }
    model, _ = sax.circuit(nl, models, backend=backend)
    np.testing.assert_allclose(model()["a_in", "a_out"], 2)
    np.testing.assert_allclose(model()["b_out", "b_in"], 1.5)
    assert set(sax.get_required_circuit_models(nl, models)) == set(models)
    models["cell_a"] = partial(_model, gain=7.0)
    model, _ = sax.circuit(nl, models, backend=backend)
    np.testing.assert_allclose(model()["a_in", "a_out"], 7)
    np.testing.assert_allclose(model()["b_in", "b_out"], 3)


def test_unqualified_factory_collision_is_explicit() -> None:
    with pytest.raises(ValueError, match=r"Ambiguous factory.*coupler2"):
        sax.circuit(_two_libraries(), {"coupler2": _model})


def test_single_library_keeps_exact_numeric_factory_name() -> None:
    nl = _two_libraries()
    data = nl.to_dict()
    del data["instances"]["b"]
    data["ports"] = [port for port in data["ports"] if port["name"].startswith("a_")]
    data["nets"] = [
        net
        for net in data["nets"]
        if not any(member.get("instance") == "b" for member in net)
    ]
    nl = PlacedNetlist.from_dict(data)
    model, _ = sax.circuit(nl, {"coupler2": _model})
    np.testing.assert_allclose(model()["a_in", "a_out"], 1)
    with pytest.raises(ValueError, match=r"Missing models.*factory='coupler2'"):
        sax.circuit(nl, {"coupler": _model})


def test_missing_identity_diagnostic_has_path_and_attempts() -> None:
    nl = _two_libraries()
    with pytest.raises(ValueError) as error:
        sax.circuit(nl)
    message = str(error.value)
    for expected in ("top_level.a", "coupler2", "cell_a", "PDK_A::coupler2", "Tried"):
        assert expected in message
    assert native.resolve(nl.instances["a"], {}, {}) is None
