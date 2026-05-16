"""Multi-dimensional interpolation utilitites."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import partial
from itertools import product
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr
from jaxtyping import Array
from natsort import natsorted
from numpy.typing import NDArray

import sax

__all__ = [
    "interpolate_xarray",
    "to_df",
    "to_xarray",
]


def interpolate_xarray(
    xarr: xr.DataArray,
    /,
    **kwargs: sax.FloatArray | str,
) -> dict[str, sax.FloatArray]:
    """Interpolate a multi-dimensional xarray with JAX.

    Args:
        xarr: the xarray to do a grid-interpolation over
        **kwargs: the other parameters to interpolate over
    """
    with jax.ensure_compile_time_eval():
        data = jnp.asarray(xarr)

        params: dict[str, NDArray] = {str(k): np.asarray(xarr[k]) for k in xarr.dims}
        new_params, strings = _extract_strings(params)
        target_name = str(xarr.dims[-1])
        targets = strings[target_name]
        if target_name in kwargs:
            msg = f"Cannot interpolate over target parameter {target_name}."
            raise ValueError(msg)
        string_kwargs = {k: v for k, v in kwargs.items() if k in strings}
        new_kwargs = {**kwargs, **{k: [str(v)] for k, v in string_kwargs.items()}}

    s, axs, pos = _evaluate_general_corner_model(
        data,
        new_params,
        strings,
        **new_kwargs,
    )

    axs_rev = dict(
        sorted([(v, k) for k, v in axs.items() if k != target_name], reverse=True)
    )

    ss = {k: s.take(pos[target_name][k], axs[target_name]) for k in targets}
    for ax, name in axs_rev.items():
        if name in string_kwargs:
            ss = {
                kk: s.take(pos[name][string_kwargs[name]], ax)  # type: ignore[reportArgumentType]
                for kk, s in ss.items()
            }

    return ss


def to_xarray(
    stacked_data: pd.DataFrame,
    *,
    target_names: Iterable[str] = ("amp", "phi"),
) -> xr.DataArray:
    """Converts a stacked dataframe into a multi-dimensional xarray.

    Args:
        stacked_data: the data to convert
        target_names: the names of the target columns.
    """
    target_names = list(target_names)
    df = stacked_data.copy()
    for name in target_names:
        if name not in df.columns:
            msg = (
                f"target name {name!r} not found in dataframe. "
                f"Got a dataframe with columns: {df.columns}."
            )
            raise ValueError(msg)
    param_names = [c for c in df.columns if c not in target_names]
    df = cast(pd.DataFrame, df[[*param_names, *target_names]])
    df = df.sort_values(by=param_names).reset_index(drop=True)
    params: dict[str, Array | NDArray] = {
        c: np.asarray(df[c].unique()) for c in param_names
    }
    params["targets"] = np.array(list(target_names), dtype=object)
    data = df[target_names].to_numpy()
    data = data.reshape(*(v.shape[0] for v in params.values()))
    data = jnp.asarray(data)
    return xr.DataArray(data=data, coords=params)


def to_df(
    obj: xr.DataArray | sax.SType,
    *,
    target_name: str = "target",
    **kwargs: sax.ArrayLike,
) -> pd.DataFrame:
    """Converts an object into a stacked dataframe.

    Args:
        obj: an xarray or a sax.SType object.
        target_name: the name of the target column
            in the dataframe (ignored when obj is an SType).
        kwargs: the coordinates of the SType values axes. You probably want to supply
            'wl' or 'f' here, i.e. the coordinates you supplied to generate the SType.
            The kwargs are ignored when the object is an xarray as the coordinates are
            already embedded in the xarray in that case.
    """
    if isinstance(obj, xr.DataArray):
        xarr = obj
        stacked = xarr.stack(__stacked_dim=xarr.dims)  # noqa: PD013
        df = (
            stacked.reset_index("__stacked_dim")
            .to_dataframe(name=target_name)
            .reset_index(drop=True)
        )
        return df

    stype = obj
    return _sdict_to_df(stype, **kwargs)


def _evaluate_general_corner_model(
    data: Array,
    params: dict[str, Array],
    strings: dict[str, dict[str, int]],
    /,
    **kwargs: Array,
) -> tuple[Array, dict[str, int], dict[str, dict[str, int]]]:
    """Hypercube representation interpolator.

    Args:
        data: the data to interpolate (output of to_hypercube)
        params: the parameter sample points (output of to_hypercube)
        strings: non-interpolatable parameters (like strings)
            with their values (output of to_hypercube)
        **kwargs: the values at which to interpolate the s-parameter.
            These values should be broadcastable into each other.

    Returns:
        the interpolated S-matrix at the values of **kwargs
    """
    given_params = {k: kwargs.get(k, jnp.asarray(v).mean()) for k, v in params.items()}
    given_strings = {
        p: (list(v.values()) if p not in kwargs else [v[vv] for vv in kwargs[p]])
        for p, v in strings.items()
    }
    string_locations = {k: i for i, k in enumerate(given_params) if k in given_strings}
    param_locations = {
        k: i for i, k in enumerate(given_params) if k not in given_strings
    }
    data = jnp.transpose(
        data, axes=[*param_locations.values(), *string_locations.values()]
    )
    given_params = {k: v for k, v in given_params.items() if k not in given_strings}
    num_params = len(given_params)
    data = _downselect(data, list(given_strings.values()))
    param_shape, string_shape = data.shape[:num_params], data.shape[num_params:]
    data = data.reshape(*param_shape, -1)
    given_param_values = [jnp.asarray(v) for v in given_params.values()]
    stacked_params = jnp.stack(jnp.broadcast_arrays(*given_param_values), 0)
    coords = _get_coordinates(
        [v for k, v in params.items() if k in given_params], stacked_params
    )
    result = _map_coordinates(data, coords)
    axs = {k: i for i, k in enumerate(given_strings)}
    rev_strings = {k: {vv: kk for kk, vv in v.items()} for k, v in strings.items()}
    pos = {
        k: {rev_strings[k][vv]: i for i, vv in enumerate(v)}
        for k, v in given_strings.items()
    }
    return result.reshape(*string_shape, *result.shape[1:]), axs, pos


def _downselect(data: Array, idxs_list: list[list[int]]) -> Array:
    for i, idxs in enumerate(reversed(idxs_list), start=1):
        data = data.take(np.asarray(idxs, dtype=int), axis=-i)
    return data


@partial(jax.vmap, in_axes=(-1, None), out_axes=0)
def _map_coordinates(input: Array, coordinates: Sequence[Array]) -> Array:  # noqa: A002
    return jax.scipy.ndimage.map_coordinates(input, coordinates, 1, mode="nearest")


def _get_coordinate(arr1d: Array, value: Array) -> Array:
    return jnp.interp(value, arr1d, jnp.arange(arr1d.shape[0]))


def _get_coordinates(arrs1d: Sequence[Array], values: jnp.ndarray) -> list[Array]:
    # don't use vmap as arrays in arrs1d could have different shapes...
    return [_get_coordinate(a, v) for a, v in zip(arrs1d, values, strict=True)]


def _extract_strings(
    params: dict[str, Array | NDArray] | dict[str, Array] | dict[str, NDArray],
) -> tuple[dict[str, Array], dict[str, dict[str, int]]]:
    string_map = {}  # jax does not like strings
    new_params = {}
    for k, v in list(params.items()):
        if v.dtype == object:  # probably string array!
            string_map[k] = {s: i for i, s in enumerate(v)}
            new_params[k] = np.array(list(string_map[k].values()), dtype=int)
        else:
            new_params[k] = np.array(v, dtype=float)
    return new_params, string_map


def _sdict_to_df(stype: sax.SType, **coords: sax.ArrayLike) -> pd.DataFrame:
    if not coords:
        msg = "The coords of at least one dimension should be given."
        raise ValueError(msg)
    coords = {k: np.atleast_1d(v) for k, v in coords.items()}
    sdict: sax.SDict = {k: jnp.atleast_1d(v) for k, v in sax.sdict(stype).items()}
    sdict = dict(zip(sdict, jnp.broadcast_arrays(*sdict.values()), strict=True))
    shape = jnp.asarray(next(iter(sdict.values()))).shape
    if len(shape) != len(coords):
        msg = "Specify at least one array of coordinates per dimension of the sdict."
        raise ValueError(msg)
    for i, (d, (k, a)) in enumerate(zip(shape, coords.items(), strict=True)):
        if a.ndim != 1:
            msg = f"Coords should be 1D arrays. {k}.ndim = {a.ndim}"
            raise ValueError(msg)
        if a.shape[0] != d:
            msg = (
                f"the length of coord {i} [len({k})={a.shape[0]}] should match "
                f"the size of axis {i} [{d}] of each element in the sdict."
            )
            raise ValueError(msg)
    ports, modes = set(), set()
    for p1, p2 in sdict:
        for p in [p1, p2]:
            p, *m = p.split("@")
            ports.add(p)
            modes.add("".join(m))
    ports = natsorted(ports)
    modes = natsorted(modes)

    sdict = {
        k: jnp.stack([jnp.abs(v), jnp.angle(v)], axis=-1) for k, v in sdict.items()
    }
    coords["amp_phi"] = np.array(["amp", "phi"])

    dfs = []
    port_mode = lambda p, m: f"{p}@{m}" if m else p  # noqa: E731
    zero = jnp.zeros_like(next(iter(sdict.values())))
    for p1, m1, p2, m2 in product(ports, modes, ports, modes):
        pm1, pm2 = port_mode(p1, m1), port_mode(p2, m2)
        values = sdict.get((pm1, pm2), zero)
        xarr = xr.DataArray(values, coords)
        df = sax.to_df(xarr, target_name="amp")
        phi = df["amp"].to_numpy()[1::2]
        df = df.loc[::2, :].drop(columns=["amp_phi"]).reset_index(drop=True)
        df["phi"] = phi
        df["port_in"] = p1
        df["port_out"] = p2
        df["mode_in"] = m1 or "1"
        df["mode_out"] = m2 or "1"
        dfs.append(df)

    df = pd.concat(dfs, axis=0)
    columns = [*[c for c in df.columns if c not in ["amp", "phi"]], *["amp", "phi"]]

    return cast(pd.DataFrame, df[columns].reset_index(drop=True))
