"""A touchstone data file parser."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from hashlib import md5
from pathlib import Path
from tempfile import gettempdir
from typing import cast, overload

import numpy as np
import pandas as pd
import skrf
import skrf as rf
import xarray as xr

import sax

__all__ = ["parse_touchstone", "write_touchstone"]


def parse_touchstone(
    content_or_filename: str | Path,
    *,
    ports: Iterable[str] = (),
    convert_to_wavelength: bool = True,
) -> pd.DataFrame:
    """Load S-parameters from a Lumerical .dat or .sparam file.

    Args:
        content_or_filename: Content as string (if contains newlines), file path,
            or file-like object with read() method.
        ports: port (or port@mode) labels to use.
            if not given, ports will be labeled as 'o1', 'o2', ...
        convert_to_wavelength: if True, convert frequency to wavelength.

    Returns:
        a pandas DataFrame with the S-parameters.

    Note:
        This function uses skrf.Network to parse the touchstone file.
    """
    temppath = None
    if isinstance(content_or_filename, str) and "\n" in content_or_filename:
        temppath = path = (
            Path(gettempdir()).resolve()
            / "sax"
            / f"touchstone_{md5(content_or_filename.encode()).hexdigest()}.dat"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content_or_filename)
    else:
        path = Path(content_or_filename).resolve()

    if not path.exists():
        msg = f"Touchstone file {path!r} not found."
        raise FileNotFoundError(msg)

    try:
        ntwk = rf.Network(str(path))
        ports = np.array(list(ports))
        if ports.shape[0] != ntwk.s.shape[-1] or ports.shape[0] != ntwk.s.shape[-2]:
            msg = (
                f"Length of ports list {ports} does not match "
                f"smatrix dimension [{ntwk.s.shape[-2]}x{ntwk.s.shape[-1]}]."
            )
            raise ValueError(msg)
        order = slice(None, None, -1) if convert_to_wavelength else slice(None)
        coords = {}
        if convert_to_wavelength:
            coords["wl"] = sax.C_UM_S / ntwk.f[order]
        coords["port_in"] = ports
        coords["port_out"] = ports
        xarr = xr.DataArray(ntwk.s[order], coords)
        df = sax.to_df(xarr, target_name="s")

        df["mode_in"] = [_get_mode(pm) for pm in df["port_in"].to_numpy()]
        df["mode_out"] = [_get_mode(pm) for pm in df["port_out"].to_numpy()]
        df["port_in"] = [_get_port(pm) for pm in df["port_in"].to_numpy()]
        df["port_out"] = [_get_port(pm) for pm in df["port_out"].to_numpy()]
        df["amp"] = np.abs(df["s"].to_numpy())
        df["phi"] = np.angle(df.pop("s").to_numpy())
    finally:
        if temppath is not None:
            temppath.unlink(missing_ok=True)
    return df


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
    nw.s = xarr.to_numpy()[:, :, :, 0]
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


def _get_port(pm: str) -> str:
    return pm.split("@")[0]


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
