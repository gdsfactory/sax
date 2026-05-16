"""Test that all submodules can be imported successfully.

This catches missing dependencies declared in pyproject.toml.
"""


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
