"""SAX S-Matrix utilities."""

from __future__ import annotations

from functools import wraps
from typing import cast, overload

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array
from natsort import natsorted
from pydantic import validate_call

import sax

from .multimode import _consolidate_sdense

__all__ = [
    "block_diag",
    "get_mode",
    "get_modes",
    "get_port_combinations",
    "get_ports",
    "reciprocal",
    "scoo",
    "sdense",
    "sdict",
]


@overload
def sdict(S: sax.Model) -> sax.SDictModel: ...


@overload
def sdict(S: sax.SType) -> sax.SDict: ...


def sdict(S: sax.Model | sax.SType) -> sax.SDictModel | sax.SDict:
    """Convert an S-matrix to SDict (dictionary) format.

    SDict format represents S-parameters as a dictionary with port pair tuples
    as keys and complex values as entries. This is the most flexible format for
    sparse matrices and custom port naming.

    Args:
        S: S-matrix in SCoo, SDense format, or a model that returns such matrices.

    Returns:
        S-matrix in SDict format, or a model that returns SDict format.

    Raises:
        ValueError: If the input cannot be converted to SDict format.

    Example:
        ```python
        # Convert from other formats
        scoo_matrix = (Si, Sj, Sx, port_map)
        sdict_matrix = sdict(scoo_matrix)


        # Convert a model
        def my_model(wl=1.55):
            return some_sdense_matrix


        sdict_model = sdict(my_model)
        ```
    """
    if callable(_model := S):

        @wraps(_model)
        def model(**kwargs: sax.SettingsValue) -> sax.SDict:
            return sdict(_model(**kwargs))

        return model

    if isinstance(_sdict := S, dict):
        return _sdict

    if len(_scoo := cast(tuple, S)) == 4:
        return _scoo_to_sdict(*_scoo)

    if len(_sdense := cast(tuple, S)) == 2:
        return _sdense_to_sdict(*_sdense)

    msg = f"Could not convert S-matrix to sdict. Got: {S!r}."
    raise ValueError(msg)


@overload
def scoo(S: sax.Model) -> sax.SCooModel: ...


@overload
def scoo(S: sax.SType) -> sax.SCoo: ...


def scoo(S: sax.Model | sax.SType) -> sax.SCooModel | sax.SCoo:
    """Convert an S-matrix to SCoo (coordinate) format.

    SCoo format represents S-parameters in coordinate (sparse) format with
    separate arrays for row indices (Si), column indices (Sj), values (Sx),
    and port mapping. This format is memory-efficient for sparse matrices.

    Args:
        S: S-matrix in SDict, SDense format, or a model that returns such matrices.

    Returns:
        S-matrix in SCoo format (Si, Sj, Sx, port_map), or a model that returns
        SCoo format.

    Raises:
        ValueError: If the input cannot be converted to SCoo format.

    Example:
        ```python
        # Convert from dict format
        sdict_matrix = {("in", "out"): 0.9 + 0.1j, ("out", "in"): 0.9 + 0.1j}
        Si, Sj, Sx, port_map = scoo(sdict_matrix)


        # Convert a model
        def my_model(wl=1.55):
            return some_sdict_matrix


        scoo_model = scoo(my_model)
        ```
    """
    if callable(_model := S):

        @wraps(_model)
        def model(**kwargs: sax.SettingsValue) -> sax.SCoo:
            return scoo(_model(**kwargs))

        return model

    if isinstance(_sdict := S, dict):
        return _sdict_to_scoo(_sdict)

    if len(_scoo := cast(tuple, S)) == 4:
        return _scoo

    if len(_sdense := cast(tuple, S)) == 2:
        return _sdense_to_scoo(*_sdense)

    msg = f"Could not convert S-matrix to scoo. Got: {S!r}."
    raise ValueError(msg)


@overload
def sdense(S: sax.Model) -> sax.SDenseModel: ...


@overload
def sdense(S: sax.SType) -> sax.SDense: ...


def sdense(S: sax.SType | sax.Model) -> sax.SDenseModel | sax.SDense:
    """Convert an S-matrix to SDense (dense matrix) format.

    SDense format represents S-parameters as a dense complex matrix with an
    associated port mapping. This format is efficient for dense matrices and
    enables fast linear algebra operations.

    Args:
        S: S-matrix in SDict, SCoo format, or a model that returns such matrices.

    Returns:
        S-matrix in SDense format (matrix, port_map), or a model that returns
        SDense format.

    Raises:
        ValueError: If the input cannot be converted to SDense format.

    Example:
        ```python
        # Convert from dict format
        sdict_matrix = {("in", "out"): 0.9 + 0.1j, ("out", "in"): 0.9 + 0.1j}
        matrix, port_map = sdense(sdict_matrix)


        # Convert a model
        def my_model(wl=1.55):
            return some_sdict_matrix


        sdense_model = sdense(my_model)
        ```
    """
    if callable(_model := S):

        @wraps(_model)
        def model(**kwargs: sax.SettingsValue) -> sax.SDense:
            return sdense(_model(**kwargs))

        return model

    if isinstance(_sdict := S, dict):
        return _sdict_to_sdense(_sdict)

    if len(_scoo := cast(tuple, S)) == 4:
        return _scoo_to_sdense(*_scoo)

    if len(_sdense := cast(tuple, S)) == 2:
        return _sdense

    msg = f"Could not convert S-matrix to sdense. Got: {S!r}."
    raise ValueError(msg)


def reciprocal(sdict: sax.SDict) -> sax.SDict:
    """Make an SDict S-matrix reciprocal by ensuring S[i,j] = S[j,i].

    Reciprocity is a fundamental property of passive optical devices where
    the S-parameter from port i to port j equals the S-parameter from port j
    to port i. This function enforces reciprocity by copying existing values.

    Args:
        sdict: S-matrix in SDict format.

    Returns:
        Reciprocal S-matrix in SDict format with symmetric port pairs.

    Example:
        ```python
        # Make a non-reciprocal matrix reciprocal
        s_matrix = {("in", "out"): 0.9 + 0.1j}
        s_reciprocal = reciprocal(s_matrix)
        # Result: {("in", "out"): 0.9+0.1j, ("out", "in"): 0.9+0.1j}
        ```
    """
    return {
        **{(p1, p2): v for (p1, p2), v in sdict.items()},
        **{(p2, p1): v for (p1, p2), v in sdict.items()},
    }


def block_diag(*arrs: Array) -> Array:
    """Create block diagonal matrix with arbitrary batch dimensions.

    Constructs a block diagonal matrix from square input matrices. All input
    arrays must have the same batch dimensions and be square in their last
    two dimensions.

    Args:
        *arrs: Square matrices to place on the block diagonal. All must have
            matching batch dimensions.

    Returns:
        Block diagonal matrix with batch dimensions preserved.

    Raises:
        ValueError: If batch dimensions don't match or matrices aren't square.

    Example:
        ```python
        import jax.numpy as jnp

        A = jnp.array([[1, 2], [3, 4]])
        B = jnp.array([[5, 6], [7, 8]])
        C = block_diag(A, B)
        # Result: [[1, 2, 0, 0],
        #          [3, 4, 0, 0],
        #          [0, 0, 5, 6],
        #          [0, 0, 7, 8]]
        ```
    """
    batch_shape = arrs[0].shape[:-2]

    N = 0
    for arr in arrs:
        if batch_shape != arr.shape[:-2]:
            msg = "Batch dimensions for given arrays don't match."
            raise ValueError(msg)
        m, n = arr.shape[-2:]
        if m != n:
            msg = "given arrays are not square."
            raise ValueError(msg)
        N += n

    arrs = tuple(arr.reshape(-1, arr.shape[-2], arr.shape[-1]) for arr in arrs)
    batch_block_diag = jax.vmap(jsp.linalg.block_diag, in_axes=0, out_axes=0)
    block_diag = batch_block_diag(*arrs)
    return block_diag.reshape(*batch_shape, N, N)


def get_ports(S: sax.SType) -> tuple[sax.Port, ...] | tuple[sax.PortMode]:
    """Extract port names from an S-matrix.

    Returns the port names present in the S-matrix in natural sorted order.
    For multimode S-matrices, returns port@mode combinations.

    Args:
        S: S-matrix in any format (SDict, SCoo, or SDense).

    Returns:
        Tuple of port names (for single-mode) or port@mode strings (for multimode).

    Raises:
        TypeError: If input is a model function (not evaluated) or invalid type.

    Example:
        ```python
        # Single-mode matrix
        s_matrix = {("in", "out"): 0.9, ("out", "in"): 0.9}
        ports = get_ports(s_matrix)
        # Result: ("in", "out")

        # Multimode matrix
        s_mm = {("in@TE", "out@TE"): 0.9, ("in@TM", "out@TM"): 0.8}
        ports_mm = get_ports(s_mm)
        # Result: ("in@TE", "in@TM", "out@TE", "out@TM")
        ```
    """
    if callable(S):
        msg = (
            "Getting the ports of a model is no longer supported. "
            "Please Evaluate the model first: Use get_ports(model()) in stead of "
            f"get_ports(model). Got: {S}"
        )
        raise TypeError(msg)
    if isinstance(sdict := S, dict):
        ports_set = {p1 for p1, _ in sdict} | {p2 for _, p2 in sdict}
        return tuple(natsorted(ports_set))

    if isinstance(S, tuple):
        *_, pm = cast(sax.SCoo | sax.SDense, S)
        return tuple(natsorted(pm.keys()))

    msg = f"Expected an SType. Got: {S!r} [{type(S)}]"
    raise TypeError(msg)


@validate_call
def get_modes(S: sax.STypeMM) -> tuple[sax.Mode, ...]:
    """Extract the optical modes from a multimode S-matrix.

    Returns all unique optical modes (e.g., "TE", "TM") present in the
    multimode S-matrix port names.

    Args:
        S: Multimode S-matrix in any format with port@mode naming convention.

    Returns:
        Tuple of unique mode names found in the S-matrix.

    Example:
        ```python
        # Multimode S-matrix with TE and TM modes
        s_mm = {
            ("in@TE", "out@TE"): 0.9,
            ("in@TM", "out@TM"): 0.8,
            ("in@TE", "out@TM"): 0.1,
        }
        modes = get_modes(s_mm)
        # Result: ("TE", "TM")
        ```
    """
    return tuple(get_mode(pm) for pm in get_ports(S))


@validate_call
def get_mode(pm: sax.PortMode) -> sax.Mode:
    """Extract the mode from a port@mode string.

    Parses the mode identifier from the standard port@mode naming convention
    used in multimode S-matrices.

    Args:
        pm: Port-mode string in the format "port@mode".

    Returns:
        The mode identifier (part after @).

    Example:
        ```python
        mode = get_mode("waveguide1@TE")
        # Result: "TE"

        mode = get_mode("input@TM")
        # Result: "TM"
        ```
    """
    return pm.split("@")[1]


def get_port_combinations(S: sax.Model | sax.SType) -> tuple[tuple[str, str], ...]:
    """Extract all port pair combinations from an S-matrix.

    Returns all (input_port, output_port) combinations present in the S-matrix.
    This is useful for understanding the connectivity structure of the device.

    Args:
        S: S-matrix in any format (SDict, SCoo, or SDense).

    Returns:
        Tuple of (port1, port2) combinations representing S-parameter entries.

    Raises:
        TypeError: If input is a model function (not evaluated).

    Example:
        ```python
        # S-matrix with cross-coupling
        s_matrix = {
            ("in1", "out1"): 0.9,
            ("in1", "out2"): 0.1,
            ("in2", "out1"): 0.1,
            ("in2", "out2"): 0.9,
        }
        combinations = get_port_combinations(s_matrix)
        # Result:
        # (("in1", "out1"), ("in1", "out2"), ("in2", "out1"), ("in2", "out2"))
        ```
    """
    if callable(S):
        msg = (
            "Getting the port combinations of a model is no longer supported. "
            "Please Evaluate the model first: Use get_ports(model()) in stead of "
            f"get_ports(model). Got: {S}"
        )
        raise TypeError(msg)
    if isinstance(sdict := S, dict):
        return tuple(sdict.keys())
    if len(scoo := cast(sax.SCoo, S)) == 4:
        Si, Sj, _, pm = scoo
        rpm = {int(i): str(p) for p, i in pm.items()}
        return tuple(
            natsorted((rpm[int(j)], rpm[int(i)]) for i, j in zip(Si, Sj, strict=False))
        )
    if len(sdense := cast(sax.SDense, S)) == 2:
        _, pm = sdense
        return tuple(natsorted((p1, p2) for p1 in pm for p2 in pm))
    msg = "Could not extract ports for given S"
    raise ValueError(msg)


def _scoo_to_sdict(
    Si: sax.IntArray1D,
    Sj: sax.IntArray1D,
    Sx: sax.ComplexArray,
    ports_map: sax.PortMap,
) -> sax.SDict:
    sdict = {}
    inverse_ports_map = {int(i): p for p, i in ports_map.items()}
    for i, (si, sj) in enumerate(zip(Si, Sj, strict=True)):
        output_port = inverse_ports_map.get(int(si), "")
        input_port = inverse_ports_map.get(int(sj), "")
        sdict[input_port, output_port] = Sx[..., i]
    return {(p1, p2): v for (p1, p2), v in sdict.items() if p1 and p2}


def _sdense_to_sdict(S: Array, ports_map: sax.PortMap) -> sax.SDict:
    sdict = {}
    for p1, i in ports_map.items():
        for p2, j in ports_map.items():
            sdict[p1, p2] = S[..., j, i]
    return sdict


def _sdict_to_scoo(sdict: sax.SDict) -> sax.SCoo:
    all_ports = {}
    for p1, p2 in sdict:
        all_ports[p1] = None
        all_ports[p2] = None
    ports_map = {p: int(i) for i, p in enumerate(all_ports)}
    Sx = jnp.stack(jnp.broadcast_arrays(*sdict.values()), -1)
    Si = jnp.array([ports_map[p] for _, p in sdict])
    Sj = jnp.array([ports_map[p] for p, _ in sdict])
    return Si, Sj, Sx, ports_map


def _sdense_to_scoo(S: sax.ComplexArray, ports_map: sax.PortMap) -> sax.SCoo:
    S, ports_map = _consolidate_sdense((S, ports_map))
    Sj, Si = jnp.meshgrid(jnp.arange(S.shape[-1]), jnp.arange(S.shape[-2]))
    return Si.ravel(), Sj.ravel(), S.reshape(*S.shape[:-2], -1), ports_map


def _scoo_to_sdense(
    Si: sax.IntArray1D,
    Sj: sax.IntArray1D,
    Sx: sax.ComplexArray,
    ports_map: dict[str, int],
) -> sax.SDense:
    n_col = len(ports_map)
    S = jnp.zeros((*Sx.shape[:-1], n_col, n_col), dtype=complex)
    S = S.at[..., Si, Sj].add(Sx)
    return S, ports_map


def _sdict_to_sdense(sdict: sax.SDict) -> sax.SDense:
    Si, Sj, Sx, ports_map = _sdict_to_scoo(sdict)
    return _scoo_to_sdense(Si, Sj, Sx, ports_map)
