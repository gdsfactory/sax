"""The sax *host*: sax's implementation of the multi-engine model contract.

A "host" is a module that exposes one small, structural (duck-typed) protocol so a
single model codebase -- cspdk is the motivating one -- can run on several
S-parameter engines without importing any of them directly. The engines deliberately
do NOT import each other, so there is no shared base class: conformance is by
attribute presence, and by the results agreeing.

The protocol
------------

``xp``
    The array module the host's models are written against: ``jax.numpy`` here.
``sm``
    The host's model library: ``sax.models`` here (``sox_py.models`` /
    ``socklu.models`` elsewhere).
``solve(connections, ports, instances)``
    A *nested* circuit solve, so a composed model calls its host instead of a
    specific backend. This wraps :func:`sax.backends.evaluate_circuit_fg`, the nested
    entry the library already uses (``sax/models/rf.py``), whose signature is
    ``evaluate_circuit_fg((connections, ports), instances)``.
``port_namer``
    The host's port-naming strategy, carried *on the host* so two engines can
    coexist in one process. sax's module-global ``set_port_naming_strategy`` /
    ``get_port_naming_strategy`` (:mod:`sax.ports`) are kept for existing callers and
    are never touched by this object: it names ports by passing its strategy to
    :class:`sax.ports.PortNamer` *explicitly*.
``differentiable``
    Whether the host traces differentiably. True for sax (jax).
``interpolate``
    The host's table-reading entry: :func:`sax.interpolation.interpolate_xarray`
    (also exported as ``sax.interpolate_xarray``).

This module is purely additive: it changes no existing behaviour and adds no
import-time work for existing callers (it is not imported by :mod:`sax`).
"""

from __future__ import annotations

from typing import cast

import jax.numpy as xp

from sax import models as sm
from sax.backends import evaluate_circuit_fg
from sax.interpolation import interpolate_xarray as interpolate
from sax.ports import PortNamer as _PortNamer
from sax.ports import PortNamingStrategy

__all__ = [
    "PortNamer",
    "PortNaming",
    "differentiable",
    "interpolate",
    "port_namer",
    "sm",
    "solve",
    "xp",
]

differentiable = True

PortNamer = _PortNamer


# --------------------------------------------------------------------------------------
# port_namer -- per-host naming, never a module global
# --------------------------------------------------------------------------------------
_PORT_STRATEGIES: tuple[PortNamingStrategy, ...] = ("inout", "optical")


def _check_strategy(strategy: str) -> PortNamingStrategy:
    s = str(strategy).lower()
    if s not in _PORT_STRATEGIES:
        msg = f"Invalid port naming strategy: {strategy}"
        raise ValueError(msg)
    return cast("PortNamingStrategy", s)


class PortNaming:
    """The host's port-naming strategy, carried on the host.

    Unlike :mod:`sax.ports`' module-global ``set_port_naming_strategy``, this holds
    its own strategy, so two hosts (sax and socklu, say) can carry different
    conventions in one process without clobbering each other. ``namer``/``names``
    delegate to :class:`sax.ports.PortNamer` with the strategy passed *explicitly*,
    so the global is never read or written.
    """

    def __init__(self, strategy: str = "inout") -> None:
        """Create a port-naming strategy holder with its own strategy."""
        self._strategy: PortNamingStrategy = _check_strategy(strategy)

    @property
    def strategy(self) -> PortNamingStrategy:
        """The strategy this host currently carries."""
        return self._strategy

    def get_strategy(self) -> PortNamingStrategy:
        """Return the strategy this host currently carries."""
        return self._strategy

    def set_strategy(self, strategy: str) -> PortNamingStrategy:
        """Set this host's strategy (does not touch any module global)."""
        self._strategy = _check_strategy(strategy)
        return self._strategy

    def namer(
        self,
        num_inputs: int,
        num_outputs: int,
        strategy: str | None = None,
    ) -> _PortNamer:
        """Return a :class:`sax.ports.PortNamer` bound to this host's strategy."""
        s = self._strategy if strategy is None else _check_strategy(strategy)
        return _PortNamer(num_inputs, num_outputs, strategy=s)

    def names(
        self,
        num_inputs: int,
        num_outputs: int,
        strategy: str | None = None,
    ) -> list[str]:
        """Return the port names, in index order, for the given port counts."""
        namer = self.namer(num_inputs, num_outputs, strategy)
        return [namer[i] for i in range(num_inputs + num_outputs)]

    def __repr__(self) -> str:
        return f"PortNaming({self._strategy!r})"


port_namer = PortNaming()


# --------------------------------------------------------------------------------------
# solve -- nested circuit solve
# --------------------------------------------------------------------------------------
def solve(connections, ports, instances):  # noqa: ANN001, ANN201
    """Nested circuit solve, host-style: ``solve(connections, ports, instances)``.

    A thin wrapper over :func:`sax.backends.evaluate_circuit_fg`, the nested entry the
    library already uses (``sax/models/rf.py``), whose ``analyzed`` argument is the
    ``(connections, ports)`` pair.
    """
    return evaluate_circuit_fg((connections, ports), instances)
