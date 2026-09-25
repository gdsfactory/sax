from functools import partial

import jax.numpy as jnp
import numpy as np
import pytest
from kfnetlist import HierarchicalNetlist, Netlist, NetlistPort, PortRef

import sax


def _model(gain: float = 1.0) -> sax.SDict:
    return {("in", "out"): jnp.asarray(gain), ("out", "in"): jnp.asarray(gain) / 2}


def _two_libraries(
    *, both: bool = True, references: bool = True
) -> HierarchicalNetlist:
    nl = Netlist()
    for name, library in (("a", "PDK_A"), ("b", "PDK_B")):
        if name == "b" and not both:
            continue
        nl.create_inst(
            name,
            kcl=library,
            component="coupler2",
            netlist_id=f"cell_{name}" if references else None,
        )
        for port in ("in", "out"):
            nl.create_port(f"{name}_{port}")
            nl.create_net(
                NetlistPort(name=f"{name}_{port}"), PortRef(instance=name, port=port)
            )
    cells = {"top_level": nl}
    if references:
        cells["cell_a"] = Netlist()
    if both and references:
        cells["cell_b"] = Netlist()
    return HierarchicalNetlist(cells)


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
    nl = _two_libraries(both=False, references=False)
    model, _ = sax.circuit(nl, {"coupler2": _model})
    np.testing.assert_allclose(model()["a_in", "a_out"], 1)
    with pytest.raises(ValueError, match=r"Missing models.*factory='coupler2'"):
        sax.circuit(nl, {"coupler": _model})


def test_missing_identity_diagnostic_has_path_and_attempts() -> None:
    nl = _two_libraries(references=False)
    with pytest.raises(ValueError) as error:
        sax.circuit(nl)
    message = str(error.value)
    for expected in ("top_level.a", "coupler2", "PDK_A::coupler2", "Tried"):
        assert expected in message


def test_distinct_referenced_children_keep_their_instance_settings() -> None:
    top = Netlist()
    children: dict[str, Netlist] = {}
    for name, gain in (("a", 2.0), ("b", 3.0)):
        cell = f"cell_{name}"
        top.create_inst(name, "pdk", "make_child", netlist_id=cell)
        for port in ("in", "out"):
            top.create_port(f"{name}_{port}")
            top.create_net(NetlistPort(f"{name}_{port}"), PortRef(name, port))
        child = Netlist()
        child.create_inst("leaf", "pdk", "coupler2", {"gain": gain})
        for port in ("in", "out"):
            child.create_port(port)
            child.create_net(NetlistPort(port), PortRef("leaf", port))
        children[cell] = child
    document = HierarchicalNetlist({"top_level": top, **children})
    model, _ = sax.circuit(document, {"coupler2": _model})
    np.testing.assert_allclose(model()["a_in", "a_out"], 2.0)
    np.testing.assert_allclose(model()["b_in", "b_out"], 3.0)
