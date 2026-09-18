from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import sax


def _data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"f": f, "port_in": pin, "port_out": pout, "re": value, "im": 0.1}
            for f in [1e9, 2e9]
            for pin, pout, value in [
                ("a", "a", 0.1),
                ("a", "b", 0.5),
                ("b", "a", 0.25),
                ("b", "b", 0.2),
            ]
        ]
    )


def test_repeated_writes_overwrite_without_mutation(tmp_path: Path) -> None:
    frame = _data()
    original = frame.copy(deep=True)
    path = tmp_path / "result.dat"
    path.write_text("old content must disappear")
    assert sax.write_lumerical_dat(frame, path) == path.resolve()
    first = path.read_text()
    sax.write_lumerical_dat(frame, path)
    assert path.read_text() == first
    pd.testing.assert_frame_equal(frame, original)
    with pytest.warns(sax.ExperimentalWarning):
        parsed = sax.parse_lumerical_dat(path)
    assert len(parsed) == len(frame)
    forward = parsed[(parsed.port_in == "port_1") & (parsed.port_out == "port_2")]
    np.testing.assert_allclose(forward.amp * np.exp(1j * forward.phi), 0.5 + 0.1j)


def test_string_writes_are_repeatable_and_file_free() -> None:
    frame = _data()
    with patch.object(Path, "open", side_effect=AssertionError("unexpected file I/O")):
        first = sax.write_lumerical_dat(frame)
        second = sax.write_lumerical_dat(frame)
    assert first == second
    assert isinstance(first, str)
