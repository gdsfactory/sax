"""SAX CUDA Backend."""

from __future__ import annotations

from typing import Any

import cupy as cp  # type: ignore[import-not-found]
import jax.numpy as jnp
import numpy as np

import sax

__all__ = [
    "analyze_circuit_cuda",
    "analyze_instances_cuda",
    "evaluate_circuit_cuda",
]


def _scoo_cupy(S: sax.SType) -> sax.SCoo:
    """Convert an S-parameter to SCoo with CuPy arrays for values."""
    if isinstance(S, dict):
        all_ports: dict[str, None] = {}
        for p1, p2 in S:
            all_ports.setdefault(p1, None)
            all_ports.setdefault(p2, None)
        ports_map = {p: int(i) for i, p in enumerate(all_ports)}
        Si = np.array([ports_map[p] for _, p in S], dtype=np.int32)
        Sj = np.array([ports_map[p] for p, _ in S], dtype=np.int32)
        Sx = cp.stack([cp.asarray(v) for v in S.values()], -1)
        return Si, Sj, Sx, ports_map
    Si, Sj, Sx, ports_map = sax.scoo(S)
    return (
        np.asarray(Si, dtype=np.int32),
        np.asarray(Sj, dtype=np.int32),
        cp.asarray(Sx),
        ports_map,
    )


def _solve_cuda(
    Ai: np.ndarray, Aj: np.ndarray, Ax: cp.ndarray, B: cp.ndarray
) -> cp.ndarray:
    """Batched sparse solve using dense GPU operations.

    Builds dense matrices from the fixed sparsity pattern (Ai, Aj) with
    per-batch values (Ax), then solves all systems in parallel using
    cuBLAS/cuSOLVER batched routines.

    Args:
        Ai: Row indices of non-zero values (topology, shared across batch).
        Aj: Column indices of non-zero values (topology, shared across batch).
        Ax: Non-zero values, shape (batch, nnz).
        B: Right-hand side matrix, shape (n, n_rhs).

    Returns:
        Solution matrix, shape (batch, n, n_rhs).
    """
    n = int(B.shape[0])
    batch = Ax.shape[0]
    Ai_cp = cp.asarray(Ai)
    Aj_cp = cp.asarray(Aj)
    A_dense = cp.zeros((batch, n, n), dtype=Ax.dtype)
    A_dense[:, Ai_cp, Aj_cp] = Ax
    return cp.linalg.solve(A_dense, cp.broadcast_to(B, (batch, *B.shape)))


def _coo_mul_vec(
    Si: np.ndarray, Sj: np.ndarray, Sx: cp.ndarray, x: cp.ndarray
) -> cp.ndarray:
    """Batched sparse matrix-dense matrix multiply using dense GPU matmul.

    Builds dense matrices from the fixed sparsity pattern (Si, Sj) with
    per-batch values (Sx), then multiplies all in parallel.

    Args:
        Si: Row indices of non-zero values (topology, shared across batch).
        Sj: Column indices of non-zero values (topology, shared across batch).
        Sx: Non-zero values, shape (batch, nnz).
        x: Dense matrix to multiply, shape (batch, n, m).

    Returns:
        Result of S @ x, shape (batch, n, m).
    """
    n = x.shape[-2]
    batch = Sx.shape[0]
    Si_cp = cp.asarray(Si)
    Sj_cp = cp.asarray(Sj)
    S_dense = cp.zeros((batch, n, n), dtype=Sx.dtype)
    S_dense[:, Si_cp, Sj_cp] = Sx
    return S_dense @ x


def analyze_instances_cuda(
    instances: dict[sax.InstanceName, sax.Instance],
    models: dict[str, sax.Model],
) -> dict[str, sax.SCoo]:
    """Analyze circuit instances for the CUDA backend.

    Args:
        instances: Dictionary mapping instance names to instance definitions.
        models: Dictionary mapping component names to their model functions.

    Returns:
        Dictionary mapping instance names to their S-matrices in SCoo format.
    """
    instances = sax.into[sax.Instances](instances)
    model_names = set()
    for i in instances.values():
        model_names.add(i["component"])
    dummy_models = {k: _scoo_cupy(models[k]()) for k in model_names}
    dummy_instances = {}
    for k, i in instances.items():
        dummy_instances[k] = dummy_models[i["component"]]
    return dummy_instances


def analyze_circuit_cuda(
    analyzed_instances: dict[sax.InstanceName, sax.SCoo],
    nets: sax.Nets,
    ports: sax.Ports,
) -> Any:  # noqa: ANN401
    """Analyze circuit topology for the CUDA backend.

    Args:
        analyzed_instances: Instance S-matrices from analyze_instances_cuda.
        nets: List of net dictionaries with "p1" and "p2" keys.
        ports: Dictionary mapping external port names to instance ports.

    Returns:
        Tuple of pre-computed arrays for evaluate_circuit_cuda.
    """
    inverse_ports = {v: k for k, v in ports.items()}
    port_map = {k: i for i, k in enumerate(ports)}

    idx, Si, Sj, instance_ports = 0, [], [], {}
    for name, instance in analyzed_instances.items():
        si, sj, _, ports_map = sax.scoo(instance)
        Si.append(np.asarray(si) + idx)
        Sj.append(np.asarray(sj) + idx)
        instance_ports.update({f"{name},{p}": i + idx for p, i in ports_map.items()})
        idx += len(ports_map)

    n_col = idx
    n_rhs = len(port_map)

    Si = np.concatenate(Si, -1)
    Sj = np.concatenate(Sj, -1)

    pairs: set[tuple[int, int]] = set()
    for net in nets:
        p1_idx = int(instance_ports[net["p1"]])
        p2_idx = int(instance_ports[net["p2"]])
        pairs.add((p1_idx, p2_idx))
        pairs.add((p2_idx, p1_idx))
    sorted_pairs = sorted(pairs)
    Ci = np.array([p[0] for p in sorted_pairs], dtype=np.int32)
    Cj = np.array([p[1] for p in sorted_pairs], dtype=np.int32)

    Cextmap = {
        int(instance_ports[k]): int(port_map[v]) for k, v in inverse_ports.items()
    }
    Cexti = cp.asarray(list(Cextmap.keys()))
    Cextj = cp.asarray(list(Cextmap.values()))
    Cext = cp.zeros((n_col, n_rhs), dtype=complex)
    Cext[Cexti, Cextj] = 1.0

    match_2d = Cj[None, :] == Si[:, None]
    CSi = np.broadcast_to(Ci[None, :], match_2d.shape)[match_2d]
    s_idx_grid = np.broadcast_to(np.arange(len(Si))[:, None], match_2d.shape)
    cs_s_indices = s_idx_grid[match_2d]
    CSj = Sj[cs_s_indices]

    Ii = Ij = np.arange(n_col)
    I_CSi = np.concatenate([CSi, Ii], -1)
    I_CSj = np.concatenate([CSj, Ij], -1)

    return (
        n_col,
        cs_s_indices,
        Si,
        Sj,
        Cext,
        Cexti,
        Cextj,
        I_CSi,
        I_CSj,
        tuple((k, v[1]) for k, v in analyzed_instances.items()),
        tuple(port_map),
    )


def evaluate_circuit_cuda(
    analyzed: Any,  # noqa: ANN401
    instances: dict[sax.InstanceName, sax.SType],
) -> sax.SDense:
    """Evaluate circuit S-matrix using batched dense GPU operations.

    Uses CuPy batched dense linear algebra (cuBLAS/cuSOLVER) instead of
    sequential sparse solves, giving much higher GPU utilization for
    typical photonic circuit sizes.

    Args:
        analyzed: Pre-computed analysis from analyze_circuit_cuda.
        instances: Dictionary mapping instance names to evaluated S-matrices.

    Returns:
        Circuit S-matrix in SDense format.
    """
    (
        n_col,
        cs_s_indices,
        Si,
        Sj,
        Cext,
        Cexti,
        Cextj,
        I_CSi,
        I_CSj,
        dummy_pms,
        port_map,
    ) = analyzed

    idx = 0
    Sx = []
    batch_shape = ()
    for name, _ in dummy_pms:
        _, _, sx, ports_map = _scoo_cupy(instances[name])
        Sx.append(sx)
        if len(sx.shape[:-1]) > len(batch_shape):
            batch_shape = sx.shape[:-1]
        idx += len(ports_map)

    Sx = cp.concatenate(
        [cp.broadcast_to(sx, (*batch_shape, sx.shape[-1])) for sx in Sx], -1
    )
    CSx = Sx[..., cs_s_indices]
    Ix = cp.ones((*batch_shape, n_col))
    I_CSx = cp.concatenate([-CSx, Ix], -1)

    Sx = Sx.reshape(-1, Sx.shape[-1])  # n_lhs x N
    I_CSx = I_CSx.reshape(-1, I_CSx.shape[-1])  # n_lhs x M
    inv_I_CS_Cext = _solve_cuda(I_CSi, I_CSj, I_CSx, Cext)
    S_inv_I_CS_Cext = _coo_mul_vec(Si, Sj, Sx, inv_I_CS_Cext)

    CextT_S_inv_I_CS_Cext = S_inv_I_CS_Cext[..., Cexti, :][..., :, Cextj]

    _, n, _ = CextT_S_inv_I_CS_Cext.shape
    S = CextT_S_inv_I_CS_Cext.reshape(*batch_shape, n, n)

    return jnp.asarray(S), {p: i for i, p in enumerate(port_map)}
