"""Parser for Lumerical S-parameter files.

Based on the simphony lumerical parser.
"""

from __future__ import annotations

import re
import warnings
from hashlib import md5
from pathlib import Path
from tempfile import gettempdir
from textwrap import dedent
from typing import cast, overload

import numpy as np
import pandas as pd
from lark import Lark, Token, Transformer, Tree, v_args
from numpy.typing import NDArray
from typing_extensions import TypedDict

import sax

from .touchstone import _validate_columns


def parse_lumerical_dat(
    content_or_filename: str | Path | sax.IOLike,
    *,
    convert_f_to_wl: bool = False,
) -> pd.DataFrame:
    """Load S-parameters from a Lumerical .dat or .sparam file.

    Args:
        content_or_filename: Content as string (if contains newlines), file path,
            or file-like object with read() method.
        convert_f_to_wl: If True, convert frequency to wavelength.

    Returns:
        a pandas DataFrame with the S-parameters.

    Note:
        This function was adapted from the `simphony` load_sparams function.
    """
    msg = (
        "The `parse_lumerical_dat` function is experimental. "
        "If you encounter any issues, "
        "Please file a bug report here: https://github.com/flaport/sax/issues ."
    )
    warnings.warn(msg, stacklevel=2, category=sax.ExperimentalWarning)
    content = sax.read(content_or_filename)
    _tree, df = cast(tuple[Tree, pd.DataFrame], _parser.parse(content))
    df = df.rename(columns={"phase": "phi"})
    x = "f"
    if convert_f_to_wl:
        x = "wl"
        df["wl"] = sax.C_UM_S / df.pop("f")
    df["amp"] = df.pop("mag")
    return cast(
        pd.DataFrame,
        df[[x, "port_in", "port_out", "mode_in", "mode_out", "amp", "phi"]],
    )


@overload
def write_lumerical_dat(df: pd.DataFrame, path: None) -> str: ...


@overload
def write_lumerical_dat(df: pd.DataFrame, path: str | Path) -> Path: ...


def write_lumerical_dat(
    df: pd.DataFrame,
    path: str | Path | None = None,
) -> Path | str:
    """Write S-parameters to a Lumerical .dat or .sparam file.

    Args:
        df: DataFrame in tidy format containing S-parameters.
        path: Path to the output file. If None, the string content is returned

    Returns:
        Path to the written file or string content if path is None.
    """
    temppath = None
    if path is None:
        temppath = path = (
            Path(gettempdir()).resolve()
            / "sax"
            / f"write_lumerical_{md5(df.to_numpy().tobytes()).hexdigest()}.dat"
        )
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
    freq = xarr.coords["f"].to_numpy()
    sparams = xarr.to_numpy()[:, :, :, 0]
    buf = Path(path).open("a")  # noqa: SIM115
    try:
        d = sparams.shape[1]
        for in_ in range(d):
            for out in range(d):
                sp = sparams[:, in_, out]
                temp = np.vstack((freq, np.abs(sp), np.unwrap(np.angle(sp)))).T
                header = f'("port {out + 1}", "TE", 1, "port {in_ + 1}", 1, "transmission")\n'
                header += f"{temp.shape}"
                np.savetxt(buf, temp, header=header, comments="")
    finally:
        buf.close()

    if temppath:
        return temppath.read_text()
    return Path(path).resolve()


class _SparamsTransformer(Transformer):
    @v_args(inline=True)
    def start(self, header: str, *datablocks: pd.DataFrame) -> tuple[str, pd.DataFrame]:
        data = pd.concat(datablocks, ignore_index=True)
        return header, data

    @v_args(inline=True)
    def datablock(
        self, ports: _PortsDict, shape: tuple[int, int], values: NDArray[np.float64]
    ) -> pd.DataFrame:
        sweepparams = ports["sweepparams"] or []
        columns = np.array([*sweepparams, "f", "mag", "phase"])
        df = pd.DataFrame(values, columns=columns)
        rows, _ = shape
        if ports["groupdelay"] is not None:
            groupshift = (
                2
                * np.pi
                * float(ports["groupdelay"])
                * (df.loc[:, "f"] - df.loc[int(rows / 2), "f"])
            )
            df.loc[:, "phase"] += groupshift
        df.loc[:, "port_out"] = ports["port_out"]
        df.loc[:, "port_in"] = ports["port_in"]
        df.loc[:, "mode_out"] = ports["mode_out"]
        df.loc[:, "mode_in"] = ports["mode_in"]
        return df

    @v_args(inline=True)
    def ports(
        self,
        port_out: str,
        mode_type: str,
        mode_out: int,
        port_in: str,
        mode_in: int,
        valuetype: str,
        groupdelay: str | None = None,
        sweepparams: str | None = None,
    ) -> _PortsDict:
        return cast(
            _PortsDict,
            {
                "port_out": port_out,
                "mode_type": mode_type,
                "mode_out": mode_out,
                "port_in": port_in,
                "mode_in": mode_in,
                "valuetype": valuetype,
                "groupdelay": groupdelay,
                "sweepparams": sweepparams,
            },
        )

    def shape(self, args: list[Token]) -> tuple[int, int]:
        int_args = [int(arg) for arg in args]
        if len(int_args) != 2:
            msg = f"Expected 2 arguments for shape, got {len(int_args)}"
            raise ValueError(msg)
        return (int_args[0], int_args[1])

    def values(self, args: list[NDArray[np.float64]]) -> np.ndarray:
        return np.array(args)

    def row(self, args: list[Token]) -> np.ndarray:
        return np.array([float(arg) for arg in args])

    @v_args(inline=True)
    def port(self, port: Token) -> str:
        return _clean_string(_destring(port))

    @v_args(inline=True)
    def modeid(self, mid: Token) -> int:
        if isinstance(mid, int):
            return mid
        msg = f"Mode ID '{mid}' is not supported, contact the developers."
        raise ValueError(msg)

    @v_args(inline=True)
    def groupdelay(self, value: Token) -> float:
        return float(value)

    @v_args(inline=True)
    def sweepparams(self, params: Token) -> list[str]:
        return _destring(str(params)).split(";")

    def MODE(self, args: Token) -> str:
        return _destring(str(args))

    def VALUETYPE(self, args: Token) -> str:
        return _destring(str(args))

    def INT(self, args: Token) -> int:
        return int(args)

    def STRING(self, args: Token) -> str:
        return str(args)


def _clean_string(s: str, dot: str = "p", minus: str = "m", other: str = "_") -> str:
    s = s.strip()
    s = s.replace(".", dot)  # dot
    s = s.replace("-", minus)  # minus
    s = re.sub("[^0-9a-zA-Z_]", other, s)
    if s[0] in "0123456789":
        s = "_" + s
    if not s.isidentifier():
        msg = f"failed to clean string to a valid python identifier: {s}"
        raise ValueError(msg)
    return s


def _destring(string: str) -> str:
    return string.replace("'", "").replace('"', "")


_sparams_grammar = dedent(r"""
    ?start: _EOL* [header] datablock+ _EOL*

    header: option1 | option2
    option1: "[" INT "," INT "]" _EOL
    option2: ("[" port "," position "]" _EOL)+

    datablock: ports shape values

    ports: "(" port "," MODE "," modeid "," port "," modeid "," VALUETYPE ["," groupdelay] ["," sweepparams] ")" _EOL
    shape: "(" INT "," INT ")" _EOL
    values: row+

    row: (SIGNED_NUMBER)+ _EOL
    position: ("'" [SIDE] "'") | ("\"" [SIDE] "\"")
    port: STRING
    modeid: INT
    VALUETYPE: STRING
    groupdelay: SIGNED_NUMBER
    sweepparams: STRING

    SIDE: "TOP" | "BOTTOM" | "LEFT" | "RIGHT"
    MODE: ("'" [POL] "'") | ("\"" [POL] "\"") | STRING
    POL: "TE" | "TM"
    _EOL: NEWLINE
    STRING: ("'" /[^']*?/ _STRING_ESC_INNER "'") | ("\"" /[^\"]*?/ _STRING_ESC_INNER "\"")

    %import common._STRING_ESC_INNER
    %import common.SIGNED_NUMBER
    %import common.NUMBER
    %import common.INT
    %import common.WS_INLINE
    %import common.WS
    %import common.NEWLINE
    %ignore WS_INLINE
""")

_parser = Lark(
    _sparams_grammar, start="start", parser="lalr", transformer=_SparamsTransformer()
)


class _PortsDict(TypedDict):
    port_out: str
    mode_type: str
    mode_out: int
    port_in: str
    mode_in: int
    valuetype: str
    groupdelay: float | None
    sweepparams: str | None
