"""Public/native PIC loading, root selection, and numerical parity."""

import json
from copy import deepcopy
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from kfnetlist import Netlist

import sax
from sax import native


def _gain(gain: float = 1.0) -> sax.SDict:
    return {("in0", "out0"): jnp.asarray(gain), ("out0", "in0"): jnp.asarray(2 * gain)}


def _flat(gain: float = 1.0) -> dict:
    return {
        "instances": {"w": {"component": "gain", "settings": {"gain": gain}}},
        "ports": {"in0": "w,in0", "out0": "w,out0"},
    }


def _document() -> dict:
    return {
        "modules": {
            "child": _flat(3.0),
            "top": {
                "instances": {"a": "child", "b": "child"},
                "routes": {"route": {"links": {"a,out0": "b,in0"}}},
                "ports": {"in0": "a,in0", "out0": "b,out0"},
            },
        },
        "toplevel": "top",
    }


def test_pic_document_numerical_parity_and_roots() -> None:
    doc = _document()
    before = deepcopy(doc)
    cells, root = native.load_pic_yaml(yaml.safe_dump(doc, sort_keys=False))
    assert root == "top"
    for data in (doc, cells, {key: value.to_dict() for key, value in cells.items()}):
        model, _ = sax.circuit(data, {"gain": _gain}, top_level_name="top")
        np.testing.assert_allclose(model()["in0", "out0"], 9)
        np.testing.assert_allclose(model()["out0", "in0"], 36)
        np.testing.assert_allclose(
            model(gain=4, b={"w": {"gain": 5}})["in0", "out0"], 20
        )
    default, _ = sax.circuit(doc, {"gain": _gain})
    np.testing.assert_allclose(default()["in0", "out0"], 9)
    selected, _ = sax.circuit(doc, {"gain": _gain}, top_level_name="child")
    np.testing.assert_allclose(selected()["in0", "out0"], 3)
    assert native.load_pic_yaml(doc, top_level_name="child")[1] == "child"
    assert native.to_hierarchy(doc["modules"])[1] == "child"
    assert (
        native.to_hierarchy({"child": _flat(), "top_level": _flat()})[1] == "top_level"
    )
    with pytest.raises(ValueError, match=r"Unknown top-level cell"):
        sax.circuit(doc, {"gain": _gain}, top_level_name="top_level")
    assert doc == before


def test_public_loaders_keep_dicts_native_loaders_keep_hierarchy(
    tmp_path: Path,
) -> None:
    child = tmp_path / "child.custom.yml"
    child.write_text(yaml.safe_dump(_flat(3)))
    top = tmp_path / "my root.custom.yml"
    top.write_text(
        yaml.safe_dump(
            {
                "instances": {"sub": "child"},
                "ports": {"in0": "sub,in0", "out0": "sub,out0"},
            }
        )
    )
    assert isinstance(sax.load_netlist(top), dict)
    public = sax.load_recursive_netlist(top, ext=".custom.yml")
    cells, root = native.load_native_recursive_netlist(top, ext=".custom.yml")
    assert list(cells) == list(public) == ["my_root", "child"]
    assert root == "my_root"
    assert all(isinstance(cell, Netlist) for cell in cells.values())
    for data in (cells, public):
        model, _ = sax.circuit(data, {"gain": _gain})
        np.testing.assert_allclose(model()["in0", "out0"], 3)
    duplicate = tmp_path / "nested"
    duplicate.mkdir()
    (duplicate / "my root.custom.yml").write_text(top.read_text())
    for loader in (sax.load_recursive_netlist, native.load_native_recursive_netlist):
        with pytest.raises(ValueError, match=r"Duplicate recursive netlist"):
            loader(top, ext=".custom.yml")


def test_native_multimodule_file_retains_children(tmp_path: Path) -> None:
    path = tmp_path / "bundle.pic.yml"
    path.write_text(yaml.safe_dump(_document()))
    cells, root = native.load_native_recursive_netlist(path)
    assert list(cells) == ["top", "child"]
    assert root == "top"
    model, _ = sax.circuit(cells, {"gain": _gain})
    np.testing.assert_allclose(model()["in0", "out0"], 9)
    with pytest.raises(ValueError, match=r"retain all cells"):
        native.load_native_netlist(path)


@pytest.mark.parametrize("field", ["settings", "metadata", "info"])
def test_pic_module_fields_rejected_explicitly(field: str) -> None:
    doc = _document()
    doc["modules"]["child"][field] = {"label": "retained by public loader"}
    with pytest.raises(ValueError, match=r"module-level"):
        native.load_pic_yaml(doc)
    assert sax.load_netlist(yaml.safe_dump(doc)) == doc


def test_pic_expressions_rejected_without_evaluation() -> None:
    doc = _document()
    doc["modules"]["child"]["instances"]["w"]["settings"]["gain"] = "${length * 2}"
    with pytest.raises(ValueError, match=r"Unsupported PIC expression.*gain"):
        native.load_pic_yaml(doc)
    with pytest.raises(ValueError, match=r"Unsupported PIC expression.*gain"):
        sax.circuit(doc, {"gain": _gain})


def test_legacy_route_array_indices_translate_to_native() -> None:
    flat = {
        "instances": {"w": {"component": "gain", "array": {"num_a": 2, "num_b": 1}}},
        "routes": {"r": {"links": {"w<0.0>,out0": "w<1.0>,in0"}}},
        "ports": {"in0": "w<0.0>,in0", "out0": "w<1.0>,out0"},
    }
    nl = native.load_native_netlist(flat)
    array = nl.instances["w"].array
    assert array is not None
    assert array.na == 2
    for data in (flat, nl, nl.to_dict(), nl.to_json()):
        model, _ = sax.circuit(data, {"gain": _gain})
        np.testing.assert_allclose(model(gain=3)["in0", "out0"], 9)
        np.testing.assert_allclose(model(gain=3)["out0", "in0"], 36)


@pytest.mark.parametrize("encoding", ["legacy", "native_dict", "native_json"])
def test_cell_named_modules_is_not_a_pic_document(encoding: str) -> None:
    cells = {
        "top": {
            "instances": {"sub": "modules"},
            "ports": {"in0": "sub,in0", "out0": "sub,out0"},
        },
        "modules": _flat(3),
    }
    data = (
        cells
        if encoding == "legacy"
        else {
            name: native.from_legacy_flat(flat).to_dict()
            for name, flat in cells.items()
        }
    )
    if encoding == "native_json":
        data = json.dumps(data)
    model, _ = sax.circuit(data, {"gain": _gain})
    np.testing.assert_allclose(model()["in0", "out0"], 3)
    np.testing.assert_allclose(model()["out0", "in0"], 6)


def test_pic_module_named_instances_remains_a_document() -> None:
    doc = {"modules": {"instances": _flat(3)}}
    cells, root = native.load_pic_yaml(doc)
    assert root == "instances"
    model, _ = sax.circuit(cells, {"gain": _gain})
    np.testing.assert_allclose(model()["in0", "out0"], 3)


@pytest.mark.parametrize("cell_name", ["instances", "ports", "modules"])
def test_native_hierarchy_keys_do_not_select_another_input_format(
    cell_name: str,
) -> None:
    cells = {
        "top": native.from_legacy_flat(
            {
                "instances": {"sub": cell_name},
                "ports": {"in0": "sub,in0", "out0": "sub,out0"},
            }
        ),
        cell_name: native.from_legacy_flat(_flat(3)),
    }
    serialized = {name: cell.to_dict() for name, cell in cells.items()}
    for data in (cells, serialized, json.dumps(serialized)):
        model, _ = sax.circuit(data, {"gain": _gain})
        np.testing.assert_allclose(model()["in0", "out0"], 3)
