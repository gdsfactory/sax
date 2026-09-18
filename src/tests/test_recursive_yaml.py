from pathlib import Path

import pytest

import sax

CONTENT = "instances: {}\nports: {}\n"


@pytest.mark.parametrize("extension", [".pic.yml", ".yaml"])
def test_recursive_discovery_keeps_top_first(tmp_path: Path, extension: str) -> None:
    top = tmp_path / f"z_top{extension}"
    top.write_text(CONTENT)
    (tmp_path / f"a_child{extension}").write_text(CONTENT)
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / f"b_child{extension}").write_text(CONTENT)
    (nested / "ignored.txt").write_text("not yaml")
    loaded = sax.load_recursive_netlist(top, ext=extension)
    assert list(loaded) == ["z_top", "a_child", "b_child"]
    assert all(net == {"instances": {}, "ports": {}} for net in loaded.values())


def test_duplicate_names_rejected(tmp_path: Path) -> None:
    top = tmp_path / "top.pic.yml"
    top.write_text(CONTENT)
    child = tmp_path / "child"
    child.mkdir()
    (child / "top.pic.yml").write_text(CONTENT)
    with pytest.raises(ValueError, match="Duplicate recursive netlist component"):
        sax.load_recursive_netlist(top)
