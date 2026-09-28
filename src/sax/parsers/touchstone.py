"""A touchstone data file parser."""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable
from io import StringIO
from math import isqrt
from pathlib import Path
from typing import cast, overload

import numpy as np
import pandas as pd
import skrf
import skrf as rf
import xarray as xr

import sax

__all__ = [
    "parse_touchstone",
    "read_sdict_touchstone",
    "write_sdict_touchstone",
    "write_touchstone",
]


def parse_touchstone(
    content_or_filename: str | Path,
    *,
    ports: Iterable[str] = (),
    convert_to_wavelength: bool = True,
) -> pd.DataFrame:
    """Load Touchstone S-parameters with input/output-labeled table columns.

    Args:
        content_or_filename: Touchstone content (if it contains newlines) or a
            file path. Raw v1 text must contain complete full-matrix records;
            the port count is inferred from the first record.
        ports: port (or port@mode) labels to use.
            if not given, ports will be labeled as 'o1', 'o2', ...
        convert_to_wavelength: if True, convert frequency to wavelength.

    Returns:
        a pandas DataFrame with the S-parameters.

    Note:
        This function uses skrf.Network to parse the touchstone file.
    """
    if isinstance(content_or_filename, str) and "\n" in content_or_filename:
        with StringIO(content_or_filename) as stream:
            stream.name = _touchstone_name(content_or_filename)
            ntwk = rf.Network(stream)
    else:
        path = Path(content_or_filename).resolve()
        if not path.exists():
            msg = f"Touchstone file {path!r} not found."
            raise FileNotFoundError(msg)
        ntwk = rf.Network(str(path))

    labels = list(ports)
    if not labels:
        labels = [f"o{i + 1}" for i in range(ntwk.nports)]
    if len(labels) != ntwk.nports or len(set(labels)) != len(labels):
        msg = f"Expected {ntwk.nports} unique port labels, got {labels}."
        raise ValueError(msg)
    order = np.argsort(ntwk.f)
    if convert_to_wavelength:
        order = order[::-1]
        coords = {"wl": sax.C_UM_S / ntwk.f[order]}
    else:
        coords = {"f": ntwk.f[order]}
    # scikit-rf uses (output, input), whereas the tidy table labels directions.
    coords["port_out"] = labels
    coords["port_in"] = labels
    xarr = xr.DataArray(ntwk.s[order], coords)
    df = sax.to_df(xarr, target_name="s")
    df["mode_in"] = [_get_mode(pm) for pm in df["port_in"].to_numpy()]
    df["mode_out"] = [_get_mode(pm) for pm in df["port_out"].to_numpy()]
    df["port_in"] = [_get_port(pm) for pm in df["port_in"].to_numpy()]
    df["port_out"] = [_get_port(pm) for pm in df["port_out"].to_numpy()]
    df["amp"] = np.abs(df["s"].to_numpy())
    df["phi"] = np.angle(df.pop("s").to_numpy())
    axis = "wl" if convert_to_wavelength else "f"
    return df[[axis, "port_in", "port_out", "mode_in", "mode_out", "amp", "phi"]]


def _touchstone_name(content: str) -> str:
    """Infer a v1 full-matrix record's rank; v2 declares its own rank."""
    lines = [line.split("!", 1)[0].strip() for line in content.splitlines()]
    if any(line.lower().startswith("[version]") for line in lines):
        return "input.ts"
    count = 0
    for line in lines:
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        # A frequency starts an odd-length row; continuation rows contain pairs.
        if count and len(fields) % 2:
            break
        try:
            [float(field) for field in fields]
        except ValueError as exc:
            msg = "Invalid numeric Touchstone record."
            raise ValueError(msg) from exc
        count += len(fields)
    n = isqrt(max(0, (count - 1) // 2))
    if n < 1 or count != 1 + 2 * n * n:
        msg = "Cannot infer port count from raw Touchstone v1 full-matrix data."
        raise ValueError(msg)
    return f"input.s{n}p"


@overload
def write_touchstone(df: pd.DataFrame, path: None) -> str: ...


@overload
def write_touchstone(df: pd.DataFrame, path: str | Path) -> Path: ...


def write_touchstone(df: pd.DataFrame, path: str | Path | None = None) -> Path | str:
    """Save S-parameter dataframe to a touchstone file.

    Args:
        df: DataFrame with S-parameters in tidy format.
            The dataframe must have the following columns:

                - 'f' or 'wl': frequency or wavelength column.
                - 'port_in': input port labels.
                - 'port_out': output port labels.
                - 'amp' or 're': amplitude or real part of the S-parameters.
                - 'phi' or 'im': phase or imaginary part of the S-parameters.

            The dataframe can also have the following optional columns:

                - 'mode_in': input mode labels.
                - 'mode_out': output mode labels.

        path: Path to save the touchstone file (None if you just want to return
            the contents). You can leave out the file extension to automatically use
            the recommended extension for the touchstone files in the
            format `.sNp`, where `N` is the number of ports.

    Returns:
        Path to the saved touchstone file if `path` is not None else the content.

    Note:
        This function uses skrf.Network.write_touchstone to save the S-parameters.

    """
    df = df.copy()
    in_amp_phi_format, in_wl_format = _validate_columns(df)
    modes = {*df["mode_in"], *df["mode_out"]}
    if in_amp_phi_format:
        df["s"] = df["amp"] * np.exp(1j * df["phi"])
        df = df.drop(columns=["amp", "phi"])
    else:
        df["s"] = df["re"] + 1j * df["im"]
        df = df.drop(columns=["re", "im"])
    if len(modes) > 1:
        df["port_in"] = [
            f"{p}@{m}" for p, m in zip(df["port_in"], df["mode_in"], strict=True)
        ]
        df["port_out"] = [
            f"{p}@{m}" for p, m in zip(df["port_out"], df["mode_out"], strict=True)
        ]
    df = df.drop(columns=["mode_in", "mode_out"])
    if in_wl_format:
        df["f"] = sax.C_UM_S / df.pop("wl")
    df = cast(pd.DataFrame, df[["f", "port_in", "port_out", "s"]])
    xarr = sax.to_xarray(df, target_names=["s"])
    nw = skrf.Network()
    nw.frequency = xarr.coords["f"].to_numpy()
    nw.s = xarr.to_numpy()[:, :, :, 0].swapaxes(1, 2)
    nw.name = "sax touchstone" if path is None else Path(path).stem
    content: str = nw.write_touchstone(return_string=True) or ""
    if not content:
        msg = "Failed to write touchstone content. Is the network empty?"
        raise RuntimeError(msg)
    lines = content.splitlines()
    lines = [line for line in lines if not line.strip().startswith("!")]
    ports = xarr.coords["port_in"].to_numpy()
    lines.insert(0, f"! ports: {', '.join(ports)}")
    content = "\n".join(lines)
    if path is None:
        return content
    n = ports.shape[0]
    suffix = f".s{n}p"
    path = Path(path).resolve()
    if not path.suffix:
        path = Path(f"{path}{suffix}")
    if path.suffix != suffix:
        msg = (
            f"Saving with extension {path.suffix!r}, but for a {n}x{n} touchstone "
            f"s-matrix, the extension {suffix!r} is recommended. "
            "You can leave out the extension from the save path to automatically "
            "use the recommended extension."
        )
        warnings.warn(msg, stacklevel=2)
    path.write_text(content)
    return path


def write_sdict_touchstone(
    sdict: sax.SDict,
    f: sax.ArrayLike,
    path: str | Path,
    *,
    ports: Iterable[str] | None = None,
    z0: float = 50.0,
) -> Path:
    """Write a frequency-swept SAX SDict to a Touchstone file.

    ``f`` is in Hz and must match the SDict's frequency axis. Port names and
    their order are recorded in a comment for a later round trip.
    """
    matrix, port_map = sax.sdense(sdict)
    matrix = np.asarray(matrix, dtype=complex)
    if matrix.ndim == 2:
        matrix = matrix[np.newaxis, ...]
    frequency = np.atleast_1d(np.asarray(f, dtype=float))
    if matrix.ndim != 3 or frequency.ndim != 1 or matrix.shape[0] != frequency.size:
        msg = "f must match the SDict's single frequency axis"
        raise ValueError(msg)

    model_ports = tuple(sorted(port_map, key=port_map.__getitem__))
    labels = model_ports if ports is None else tuple(ports)
    if len(labels) != len(model_ports) or set(labels) != set(model_ports):
        msg = f"ports={labels} is not a permutation of model ports {model_ports}"
        raise ValueError(msg)
    order = [port_map[label] for label in labels]
    matrix = matrix[:, order, :][:, :, order]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    network = skrf.Network()
    network.frequency = frequency
    network.s = matrix
    network.z0 = z0
    network.name = path.stem
    content = network.write_touchstone(return_string=True, form="ri")
    if not content:
        msg = "Failed to write touchstone content. Is the network empty?"
        raise RuntimeError(msg)
    content = (
        "\n".join(
            [
                f"! ports: {', '.join(labels)}",
                *(line for line in content.splitlines() if not line.startswith("!")),
            ]
        )
        + "\n"
    )
    path.write_text(content)
    return path


def read_sdict_touchstone(
    path: str | Path, *, ports: Iterable[str] | None = None
) -> tuple[np.ndarray, sax.SDict]:
    """Read a Touchstone file as frequencies in Hz and a SAX SDict."""
    path = Path(path)
    network = skrf.Network(str(path))
    if np.any(network.z0 != network.z0.flat[0]):
        msg = "per-port or frequency-dependent reference impedances are not supported"
        raise ValueError(msg)
    if ports is None:
        match = re.search(
            r"^\s*!\s*ports\s*:\s*(.+)$", path.read_text(), re.IGNORECASE | re.MULTILINE
        )
        labels = (
            tuple(label.strip() for label in match.group(1).split(","))
            if match
            else tuple(
                network.port_names or (f"o{i + 1}" for i in range(network.nports))
            )
        )
    else:
        labels = tuple(ports)
    if len(labels) != network.nports or len(set(labels)) != len(labels):
        msg = f"expected {network.nports} unique port labels, got {labels}"
        raise ValueError(msg)
    sdict = {
        (port_in, port_out): network.s[:, j, i]
        for i, port_in in enumerate(labels)
        for j, port_out in enumerate(labels)
    }
    return network.f, sdict


def _get_port(pm: str) -> str:
    return pm.split("@", maxsplit=1)[0]


def _get_mode(pm: str) -> str:
    _, *m = pm.split("@")
    return "".join(m) or "1"


def _validate_columns(df: pd.DataFrame) -> tuple[bool, bool]:  # noqa: C901
    amp_phi_format = "amp" in df.columns or "phi" in df.columns
    if amp_phi_format:
        if "amp" not in df.columns:
            msg = (
                "a dataframe in amplitude/phase format must have an 'amp' column. "
                f"Found columns: {', '.join(df.columns)}"
            )
            raise ValueError(msg)
        if "phi" not in df.columns:
            msg = (
                "a dataframe in amplitude/phase format must have a 'phi' column. "
                f"Found columns: {', '.join(df.columns)}"
            )
            raise ValueError(msg)
    else:
        if "re" not in df.columns:
            msg = (
                "a dataframe in real/imaginary format must have a 're' column. "
                f"Found columns: {', '.join(df.columns)}"
            )
            raise ValueError(msg)
        if "im" not in df.columns:
            msg = (
                "a dataframe in real/imaginary format must have an 'im' column. "
                f"Found columns: {', '.join(df.columns)}"
            )
            raise ValueError(msg)
    if "port_in" not in df.columns:
        msg = (
            "the dataframe to convert to touchstone must have a 'port_in' column. "
            f"Found columns: {', '.join(df.columns)}"
        )
        raise ValueError(msg)

    if "port_out" not in df.columns:
        msg = (
            "the dataframe to convert to touchstone must have a 'port_in' column. "
            f"Found columns: {', '.join(df.columns)}"
        )
        raise ValueError(msg)

    if "mode_in" not in df.columns:
        df["mode_in"] = "1"

    if "mode_out" not in df.columns:
        df["mode_out"] = "1"

    if "wl" not in df.columns and "f" not in df.columns:
        msg = (
            "the dataframe to convert to touchstone must have a 'wl' or 'f' column. "
            f"Found columns: {', '.join(df.columns)}"
        )
        raise ValueError(msg)
    wl_format = "wl" in df.columns
    return amp_phi_format, wl_format
