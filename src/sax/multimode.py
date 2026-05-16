"""SAX Multimode support."""

from __future__ import annotations

from collections.abc import Iterable
from functools import wraps
from typing import cast, overload

import jax.numpy as jnp
from natsort import natsorted

import sax

from .constants import DEFAULT_MODE, DEFAULT_MODES

__all__ = [
    "multimode",
    "singlemode",
]


@overload
def multimode(
    S: sax.SDictModel, modes: tuple[str, ...] = DEFAULT_MODES
) -> sax.SDictModel: ...


@overload
def multimode(
    S: sax.SCooModel, modes: tuple[str, ...] = DEFAULT_MODES
) -> sax.SCooModel: ...


@overload
def multimode(
    S: sax.SDenseModel, modes: tuple[str, ...] = DEFAULT_MODES
) -> sax.SDenseModel: ...


@overload
def multimode(S: sax.SDict, modes: tuple[str, ...] = DEFAULT_MODES) -> sax.SDictMM: ...


@overload
def multimode(S: sax.SCoo, modes: tuple[str, ...] = DEFAULT_MODES) -> sax.SCooMM: ...


@overload
def multimode(
    S: sax.SDense, modes: tuple[str, ...] = DEFAULT_MODES
) -> sax.SDenseMM: ...


def multimode(
    S: sax.SType | sax.Model,
    modes: tuple[str, ...] = DEFAULT_MODES,
) -> sax.STypeMM | sax.ModelMM:
    """Convert a single-mode S-matrix or model to multimode.

    Converts single-mode S-parameters to multimode by replicating the single-mode
    behavior across multiple optical modes (e.g., TE, TM). The resulting multimode
    S-matrix will have port@mode naming convention.

    Args:
        S: Single-mode S-matrix in any format or a model that returns such matrices.
        modes: Tuple of mode names to include. Defaults to ("TE", "TM").

    Returns:
        Multimode S-matrix with port@mode naming, or a model that returns such matrices.

    Raises:
        ValueError: If the input cannot be converted to multimode.

    Example:
        ```python
        # Convert single-mode S-matrix to multimode
        s_sm = {("in", "out"): 0.9 + 0.1j, ("out", "in"): 0.9 + 0.1j}
        s_mm = multimode(s_sm, modes=("TE", "TM"))
        # Result contains: ("in@TE", "out@TE"), ("in@TM", "out@TM"), etc.


        # Convert a model to multimode
        def single_mode_model(wl=1.55):
            return {("in", "out"): 0.9}


        mm_model = multimode(single_mode_model)
        ```
    """
    if (model := sax.try_into[sax.Model](S)) is not None:

        @wraps(model)
        def new_model(**params: sax.SettingsValue) -> sax.STypeMM:
            return multimode(model(**params), modes=modes)

        return cast(sax.ModelMM, new_model)

    s: sax.SType = sax.into[sax.SType](S)

    if (sdict_sm := sax.try_into[sax.SDictSM](s)) is not None:
        return _multimode_sdict(sdict_sm, modes=modes)

    if (scoo_sm := sax.try_into[sax.SCooSM](s)) is not None:
        return _multimode_scoo(scoo_sm, modes=modes)

    if (sdense_sm := sax.try_into[sax.SDenseSM](s)) is not None:
        return _multimode_sdense(sdense_sm)

    if (s_mm := sax.try_into[sax.STypeMM](s)) is not None:
        return s_mm

    msg = f"Cannot convert {S!r} to multimode. Unknown SType."
    raise ValueError(msg)


def _multimode_sdict(
    sdict: sax.SDict, modes: tuple[str, ...] = DEFAULT_MODES
) -> sax.SDict:
    multimode_sdict = {}
    mode_combinations = _mode_combinations(modes)
    for (p1, p2), value in sdict.items():
        for m1, m2 in mode_combinations:
            multimode_sdict[f"{p1}@{m1}", f"{p2}@{m2}"] = value
    return multimode_sdict


def _multimode_scoo(
    scoo: sax.SCoo, modes: tuple[sax.Mode, ...] = DEFAULT_MODES
) -> sax.SCoo:
    Si, Sj, Sx, port_map = scoo
    num_ports = len(port_map)
    mode_map = {mode: i for i, mode in enumerate(modes)}

    mode_combinations = _mode_combinations(modes)

    Si_m = jnp.concatenate(
        [Si + mode_map[m] * num_ports for _, m in mode_combinations],
        -1,
    )
    Sj_m = jnp.concatenate(
        [Sj + mode_map[m] * num_ports for m, _ in mode_combinations],
        -1,
    )
    Sx_m = jnp.concatenate([Sx for _ in mode_combinations], -1)
    port_map_m = {
        f"{port}@{mode}": idx + mode_map[mode] * num_ports
        for mode in modes
        for port, idx in port_map.items()
    }

    return Si_m, Sj_m, Sx_m, port_map_m


def _multimode_sdense(
    sdense: sax.SDenseSM, modes: tuple[sax.Mode, ...] = DEFAULT_MODES
) -> sax.SDenseMM:
    Sx, port_map = sdense
    num_ports = len(port_map)
    mode_map: dict[sax.Mode, int] = {mode: i for i, mode in enumerate(modes)}

    Sx_m = sax.block_diag(*(Sx for _ in modes))

    port_map_m = {
        f"{port}@{mode}": idx + mode_map[mode] * num_ports
        for mode in modes
        for port, idx in port_map.items()
    }

    return Sx_m, port_map_m


@overload
def singlemode(S: sax.SDictModel, mode: str = DEFAULT_MODE) -> sax.SDictModelSM: ...


@overload
def singlemode(S: sax.SCooModel, mode: str = DEFAULT_MODE) -> sax.SCooModelSM: ...


@overload
def singlemode(S: sax.SDenseModel, mode: str = DEFAULT_MODE) -> sax.SDenseModelSM: ...


@overload
def singlemode(S: sax.SDict, mode: str = DEFAULT_MODE) -> sax.SDictSM: ...


@overload
def singlemode(S: sax.SCoo, mode: str = DEFAULT_MODE) -> sax.SCooSM: ...


@overload
def singlemode(S: sax.SDense, mode: str = DEFAULT_MODE) -> sax.SDenseSM: ...


def singlemode(
    S: sax.SType | sax.Model, mode: sax.Mode = DEFAULT_MODE
) -> sax.STypeSM | sax.ModelSM:
    """Convert a multimode S-matrix or model to single-mode.

    Extracts a single optical mode from a multimode S-matrix, effectively
    filtering out all other modes. The resulting single-mode S-matrix will
    have standard port naming (without @mode suffix).

    Args:
        S: Multimode S-matrix in any format or a model that returns such matrices.
        mode: The optical mode to extract (e.g., "TE", "TM"). Defaults to "TE".

    Returns:
        Single-mode S-matrix with standard port naming, or a model that returns
        such matrices.

    Raises:
        ValueError: If the input cannot be converted to single-mode.

    Example:
        ```python
        # Extract TE mode from multimode S-matrix
        s_mm = {
            ("in@TE", "out@TE"): 0.9,
            ("in@TM", "out@TM"): 0.8,
            ("in@TE", "out@TM"): 0.1,
        }
        s_te = singlemode(s_mm, mode="TE")
        # Result: {("in", "out"): 0.9}


        # Convert a multimode model to single-mode
        def multimode_model(wl=1.55):
            return multimode_s_matrix


        te_model = singlemode(multimode_model, mode="TE")
        ```
    """
    if (model := sax.try_into[sax.Model](S)) is not None:

        @wraps(model)
        def new_model(**params: sax.SettingsValue) -> sax.STypeSM:
            return singlemode(model(**params), mode=mode)

        return cast(sax.ModelSM, new_model)

    if (sdict_mm := sax.try_into[sax.SDictMM](S)) is not None:
        return _singlemode_sdict(sdict_mm, mode=mode)

    if (scoo_mm := sax.try_into[sax.SCooMM](S)) is not None:
        return _singlemode_scoo(scoo_mm, mode=mode)

    if (sdense_mm := sax.try_into[sax.SDenseMM](S)) is not None:
        return _singlemode_sdense(sdense_mm, mode=mode)

    if (s_sm := sax.try_into[sax.STypeSM](S)) is not None:
        return s_sm

    msg = f"Cannot convert {S!r} to singlemode. Unknown SType."
    raise ValueError(msg)


def _singlemode_sdict(sdict: sax.SDictMM, mode: str = DEFAULT_MODE) -> sax.SDictSM:
    sdict_sm = {}
    for (p1, p2), value in sdict.items():
        if p1.endswith(f"@{mode}") and p2.endswith(f"@{mode}"):
            p1, _ = p1.split("@")
            p2, _ = p2.split("@")
            sdict_sm[p1, p2] = value
    return sdict_sm


def _singlemode_scoo(scoo: sax.SCooMM, mode: str = DEFAULT_MODE) -> sax.SCooSM:
    Si, Sj, Sx, port_map = scoo
    # no need to touch the data...
    # just removing some ports from the port map should be enough
    port_map = {
        port.split("@")[0]: idx
        for port, idx in port_map.items()
        if port.endswith(f"@{mode}")
    }
    return Si, Sj, Sx, port_map


def _singlemode_sdense(sdense: sax.SDenseMM, mode: str = DEFAULT_MODE) -> sax.SDenseSM:
    Sx, port_map = sdense
    port_map = {
        port.split("@")[0]: idx
        for port, idx in port_map.items()
        if port.endswith(f"@{mode}")
    }
    return _consolidate_sdense((Sx, port_map))


def _mode_combinations(
    modes: Iterable[str], *, cross: bool = False
) -> tuple[tuple[str, str], ...]:
    if cross:
        combinations = natsorted((m1, m2) for m1 in modes for m2 in modes)
    else:
        combinations = natsorted((m, m) for m in modes)
    return tuple(combinations)


def _consolidate_sdense(S: sax.SDense) -> sax.SDense:
    s, pm = S
    idxs = list(pm.values())
    s = s[..., idxs, :][..., :, idxs]
    pm = {p: i for i, p in enumerate(pm)}
    return s, pm
