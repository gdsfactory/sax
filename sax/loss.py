"""SAX Loss Functions"""

from __future__ import annotations

from typing import cast

import jax.numpy as jnp

from .saxtypes import ComplexArrayND


def mse(x: ComplexArrayND, y: ComplexArrayND) -> float:
    """Mean squared error"""
    diff = x - y
    # Use np.real to avoid unnecessary complex magnitude calculation if not needed
    sq = (diff.real**2 + diff.imag**2) if hasattr(diff, "imag") else diff**2
    return cast(float, sq.mean())


def huber_loss(x: ComplexArrayND, y: ComplexArrayND, delta: float = 0.5) -> float:
    """Huber loss"""
    loss = ((delta**2) * ((1.0 + (abs(x - y) / delta) ** 2) ** 0.5 - 1.0)).mean()
    return cast(float, loss)


def l2_reg(weights: dict[str, ComplexArrayND]) -> float:
    """L2 regularization loss"""
    numel = 0
    loss = 0.0
    for w in (v for k, v in weights.items() if k[0] in ("w", "b")):
        numel = numel + w.size
        loss = loss + (jnp.abs(w) ** 2).sum()
    return cast(float, loss / numel)
