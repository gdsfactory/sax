import subprocess
import sys

import sax


def test_default_is_klu() -> None:
    assert sax.into[sax.Backend]("default") == "klu"
    assert sax.backends.analyze_circuit is sax.backends.analyze_circuit_klu
    assert sax.backends.evaluate_circuit is sax.backends.evaluate_circuit_klu


def test_missing_required_klu_fails_import_without_fallback() -> None:
    code = """
import importlib.abc
import sys

class BlockKLU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "klujax":
            raise ModuleNotFoundError("No module named 'klujax'", name="klujax")
sys.meta_path.insert(0, BlockKLU())
try:
    import sax
except ModuleNotFoundError as exc:
    assert exc.name == "klujax", exc
else:
    raise AssertionError("Missing required klujax unexpectedly allowed import")
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
