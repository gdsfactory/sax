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

The protocol also carries the *model-writing* surface a model module would otherwise
take from ``jax``/``sax`` directly, so a model module never imports either:

``jit``
    The engine's just-in-time decorator: the real :func:`jax.jit` for sax.
``ensure_compile_time_eval``
    The tracing guard the library uses to force table loads at compile time
    (:func:`jax.ensure_compile_time_eval`).
``reciprocal``
    :func:`sax.reciprocal`.
``set_port_naming_strategy(strategy)``
    Apply a port-naming strategy to the *engine's own models* -- for sax that is the
    module-global :func:`sax.ports.set_port_naming_strategy`, because ``sax.models``
    reads it. Distinct from ``port_namer``, which is per-host and never touches a
    global.
``SDict`` / ``SCoo`` / ``SType`` / ``FloatArrayLike`` / ``FloatArray`` / ``ArrayLike``
    The type aliases the models annotate with, re-exported from :mod:`sax` (or defined
    where the engine has no equivalent).

This module is purely additive: it changes no existing behaviour and adds no
import-time work for existing callers (it is not imported by :mod:`sax`).
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as xp

import sax
from sax import models as sm
from sax.backends import evaluate_circuit_fg
from sax.interpolation import interpolate_xarray as interpolate
from sax.ports import PortNamer as _PortNamer
from sax.ports import PortNamingStrategy

__all__ = [
    "ArrayLike",
    "FloatArray",
    "FloatArrayLike",
    "PortNamer",
    "PortNaming",
    "SCoo",
    "SDict",
    "SType",
    "differentiable",
    "ensure_compile_time_eval",
    "interpolate",
    "jit",
    "port_namer",
    "reciprocal",
    "set_port_naming_strategy",
    "sm",
    "solve",
    "xp",
]

differentiable = True

PortNamer = _PortNamer

# --------------------------------------------------------------------------------------
# jit / ensure_compile_time_eval -- the engine's tracing surface
# --------------------------------------------------------------------------------------
# The real jax decorators. ``@jit`` and ``@partial(jit, inline=True)`` behave exactly
# as on jax.jit.
jit = jax.jit
ensure_compile_time_eval = jax.ensure_compile_time_eval

# --------------------------------------------------------------------------------------
# Type aliases -- re-exported so a model module annotates without importing sax
# --------------------------------------------------------------------------------------
SDict = sax.SDict
SCoo = sax.SCoo
SType = sax.SType
FloatArrayLike = sax.FloatArrayLike
FloatArray = sax.FloatArray
try:  # jaxtyping.ArrayLike if present, else a structural union
    from jaxtyping import ArrayLike
except ModuleNotFoundError:  # pragma: no cover - jaxtyping is a sax dependency
    ArrayLike = sax.FloatArrayLike  # type: ignore[assignment,misc]

# --------------------------------------------------------------------------------------
# reciprocal -- re-exported so a model module does not import sax for it
# --------------------------------------------------------------------------------------
reciprocal = sax.reciprocal


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


def set_port_naming_strategy(strategy: str) -> PortNamingStrategy:
    """Apply a port-naming strategy to sax's own models and to this host.

    ``sax.models`` names its ports from the module-global strategy, so a model codebase
    that wants optical keys from ``sm.straight`` must set *that* global. This is the
    routing entry for ``sax.set_port_naming_strategy``: it sets the global (exactly as
    calling :func:`sax.set_port_naming_strategy` does) and keeps ``port_namer`` in sync.
    """
    s = _check_strategy(strategy)
    sax.ports.set_port_naming_strategy(s)
    port_namer.set_strategy(s)
    return s


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
