"""The sax host exposes the full multi-engine model contract.

The contract is structural: a host module must carry every protocol attribute, so a
single model codebase (cspdk) can be pointed at it. This pins that surface and the
property that matters for coexistence -- port naming must not mutate sax's global.
"""

from __future__ import annotations

import jax.numpy as jnp

import sax
import sax.ports
from sax import host

PROTOCOL = ("xp", "sm", "solve", "port_namer", "differentiable", "interpolate")


def test_host_exposes_every_protocol_attribute() -> None:
    for name in PROTOCOL:
        assert hasattr(host, name), f"sax.host is missing the attribute {name!r}"


def test_xp_is_jax_numpy() -> None:
    assert host.xp is jnp


def test_sm_is_sax_models_and_has_a_model() -> None:
    assert host.sm is sax.models
    assert callable(host.sm.straight)


def test_differentiable_is_true() -> None:
    assert host.differentiable is True


def test_interpolate_is_the_table_entry() -> None:
    assert host.interpolate is sax.interpolate_xarray


def test_solve_is_a_nested_circuit_solve() -> None:
    instances = {"wg": sax.models.straight(wl=1.55, length=10.0)}
    result = host.solve({}, {"o1": "wg,in0", "o2": "wg,out0"}, instances)
    assert ("o1", "o2") in result
    assert ("o2", "o1") in result


def test_port_namer_carries_strategy_without_touching_the_global() -> None:
    before = sax.get_port_naming_strategy()
    assert host.port_namer.get_strategy() == "inout"

    host.port_namer.set_strategy("optical")
    assert host.port_namer.get_strategy() == "optical"
    assert host.port_namer.names(1, 1) == ["o1", "o2"]

    # The module-global strategy is untouched.
    assert sax.get_port_naming_strategy() == before

    host.port_namer.set_strategy("inout")
    assert host.port_namer.names(1, 2) == ["in0", "out1", "out0"]


def test_port_namer_rejects_unknown_strategy() -> None:
    import pytest

    with pytest.raises(ValueError, match="Invalid port naming strategy"):
        host.port_namer.set_strategy("sideways")
