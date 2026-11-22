"""Test that all submodules can be imported successfully.

This catches missing dependencies declared in pyproject.toml.
"""

import subprocess
import sys

HEAVY_MODULES = [
    "matplotlib",
    "optax",
    "pandas",
    "scipy",
    "skrf",
    "sympy",
    "xarray",
]


def test_import_sax_is_lazy() -> None:
    """``import sax`` must not eagerly import heavy optional dependencies."""
    code = (
        "import sys; import sax; print([m for m in sys.argv[1:] if m in sys.modules])"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code, *HEAVY_MODULES],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "[]", result.stdout


def test_import_sax() -> None:
    import sax  # noqa: F401


def test_import_backends() -> None:
    import sax.backends  # noqa: F401


def test_import_fit() -> None:
    import sax.fit  # noqa: F401


def test_import_interpolation() -> None:
    import sax.interpolation  # noqa: F401


def test_import_utils() -> None:
    import sax.utils  # noqa: F401


def test_import_parsers_touchstone() -> None:
    import sax.parsers.touchstone  # noqa: F401


def test_import_parsers_lumerical() -> None:
    import sax.parsers.lumerical  # noqa: F401
