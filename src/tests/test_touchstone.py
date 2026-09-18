from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import skrf

import sax

# Touchstone 2-port order: S11, S21, S12, S22 (not row-major).
TEXT = "# Hz S RI R 50\n1000000000 .1 .2 .5 .6 .25 -.3 .4 .1\n"
MATRIX = np.array([[0.1 + 0.2j, 0.25 - 0.3j], [0.5 + 0.6j, 0.4 + 0.1j]])


@pytest.mark.parametrize("wavelength", [True, False])
@pytest.mark.parametrize("raw", [True, False])
def test_reader_external_asymmetric_fixture(
    tmp_path: Path, wavelength: bool, raw: bool
) -> None:
    path = tmp_path / "external.s2p"
    path.write_text(TEXT)
    frame = sax.parse_touchstone(
        TEXT if raw else path, ports=["a@TE", "b@TM"], convert_to_wavelength=wavelength
    )
    expected = {
        ("a", "a"): MATRIX[0, 0],
        ("a", "b"): MATRIX[1, 0],
        ("b", "a"): MATRIX[0, 1],
        ("b", "b"): MATRIX[1, 1],
    }
    for row in frame.itertuples():
        np.testing.assert_allclose(
            row.amp * np.exp(1j * row.phi), expected[row.port_in, row.port_out]
        )
    coordinate = "wl" if wavelength else "f"
    np.testing.assert_allclose(
        frame[coordinate], sax.C_UM_S / 1e9 if wavelength else 1e9
    )
    assert set(frame["mode_in"]) == {"TE", "TM"}


def test_default_labels_and_invalid_labels(tmp_path: Path) -> None:
    path = tmp_path / "external.s2p"
    path.write_text(TEXT)
    assert set(sax.parse_touchstone(path)["port_in"]) == {"o1", "o2"}
    for labels in [["a"], ["a", "a"]]:
        with pytest.raises(ValueError, match="unique port labels"):
            sax.parse_touchstone(path, ports=labels)


def test_writer_independent_skrf_reader(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "f": 1e9,
                "port_in": f"p{j}",
                "port_out": f"p{i}",
                "re": MATRIX[i, j].real,
                "im": MATRIX[i, j].imag,
            }
            for j in range(2)
            for i in range(2)
        ]
    )
    original = frame.copy(deep=True)
    path = sax.write_touchstone(frame, tmp_path / "written")
    np.testing.assert_allclose(skrf.Network(str(path)).s[0], MATRIX)
    pd.testing.assert_frame_equal(frame, original)


def test_raw_multiline_three_port() -> None:
    text = "# Hz S RI R 50\n1e9 1 0 2 0 3 0\n4 0 5 0 6 0\n7 0 8 0 9 0\n"
    frame = sax.parse_touchstone(text, convert_to_wavelength=False)
    assert len(frame) == 9
    assert set(frame["port_in"]) == {"o1", "o2", "o3"}
    np.testing.assert_allclose(frame["f"], 1e9)


def test_raw_v2() -> None:
    text = "[Version] 2.0\n# Hz S RI R 50\n[Number of Ports] 1\n[Number of Frequencies] 1\n[Network Data]\n1e9 .5 .25\n[End]\n"
    frame = sax.parse_touchstone(text)
    np.testing.assert_allclose(frame["amp"], abs(0.5 + 0.25j))


def test_raw_invalid_record() -> None:
    with pytest.raises(ValueError, match="infer port count"):
        sax.parse_touchstone("# Hz S RI R 50\n1e9 1 0 2 0\n")
