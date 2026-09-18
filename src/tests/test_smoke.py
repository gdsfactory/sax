"""Small numerical sanity suite; run with `just smoke`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sax


def _component(wl: sax.FloatArrayLike = 1.55, gain: float = 1.0) -> sax.SDict:
    return sax.reciprocal({("in0", "out0"): gain * jnp.exp(1j * jnp.asarray(wl))})


def test_asymmetric_representations() -> None:
    original: sax.SDict = {
        ("a", "b"): jnp.asarray(2 + 3j),
        ("b", "a"): jnp.asarray(4 - 1j),
    }
    dense, ports = sax.sdense(original)
    np.testing.assert_allclose(dense[ports["b"], ports["a"]], 2 + 3j)
    for converted in (sax.sdict(sax.scoo(original)), sax.sdict((dense, ports))):
        for key, value in original.items():
            np.testing.assert_allclose(converted[key], value)


@pytest.mark.parametrize("backend", ["klu", "fg"])
def test_circuit_settings_jit_gradient(backend: sax.BackendLike) -> None:
    net = {
        "instances": {
            "a": {"component": "component", "settings": {"gain": 2.0}},
            "b": "component",
        },
        "connections": {"a,out0": "b,in0"},
        "ports": {"in": "a,in0", "out": "b,out0"},
    }
    model, _ = sax.circuit(net, {"component": _component}, backend=backend)
    wl = jnp.array([1.5, 1.6])
    np.testing.assert_allclose(
        model(wl=wl, gain=2.0, b={"gain": 3.0})["in", "out"],
        6 * jnp.exp(2j * wl),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        jax.jit(model)(wl=wl)["in", "out"],
        2 * jnp.exp(2j * wl),
        rtol=1e-5,
    )
    gradient = jax.grad(lambda g: jnp.abs(model(a={"gain": g})["in", "out"]) ** 2)(2.0)
    np.testing.assert_allclose(gradient, 4.0, rtol=1e-5)


def test_optical_model_limit() -> None:
    result = sax.models.straight(length=0.0, wl=jnp.array([1.5, 1.6]))
    np.testing.assert_allclose(result["in0", "out0"], 1.0)
