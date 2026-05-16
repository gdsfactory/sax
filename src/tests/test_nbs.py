"""Test execution of notebooks in the nbs directory."""

import shutil
from collections.abc import Generator
from pathlib import Path

import pytest
from papermill.engines import papermill_engines
from papermill.execute import raise_for_execution_errors
from papermill.iorw import load_notebook_node

TEST_DIR = Path(__file__).resolve().parent
NBS_DIR = TEST_DIR / "nbs"
NBS_FAIL_DIR = TEST_DIR / "failed"

shutil.rmtree(NBS_FAIL_DIR, ignore_errors=True)
NBS_FAIL_DIR.mkdir(exist_ok=True)


def _find_notebooks(*dir_parts: str) -> Generator[Path, None, None]:
    base_dir = TEST_DIR.joinpath(*dir_parts).resolve()
    for path in base_dir.rglob("*.ipynb"):
        if "checkpoint" in path.name:
            continue
        yield path


@pytest.mark.parametrize("path", sorted(_find_notebooks("nbs")))
def test_nbs(path: Path | str) -> None:
    fn = Path(path).name
    nb = load_notebook_node(str(path))
    nb = papermill_engines.execute_notebook_with_engine(
        engine_name=None,
        nb=nb,
        kernel_name="sax",
        input_path=str(path),
        output_path=None,
    )
    raise_for_execution_errors(nb, str(NBS_FAIL_DIR / fn))
