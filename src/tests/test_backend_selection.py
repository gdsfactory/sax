import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sax


def _split() -> sax.SDict:
    return {("in0", "out0"): jnp.asarray(0.2), ("in0", "out1"): jnp.asarray(0.3)}


def _merge() -> sax.SDict:
    return {("in0", "out0"): jnp.asarray(0.4), ("in1", "out0"): jnp.asarray(0.5)}


def _waveguide(gain: sax.FloatArrayLike = 0.8) -> sax.SDict:
    return {("in0", "out0"): jnp.asarray(gain)}


def test_unequal_depth_reconvergence_matches_physical_solvers() -> None:
    net = {
        "instances": {"s": "split", "w": "waveguide", "m": "merge"},
        "connections": {"s,out0": "m,in0", "s,out1": "w,in0", "w,out0": "m,in1"},
        "ports": {"in0": "s,in0", "out0": "m,out0"},
    }
    models = {"split": _split, "merge": _merge, "waveguide": _waveguide}
    gain = jnp.array([0.5, 0.8])
    expected = 0.08 + 0.15 * gain
    for backend in ("klu", "fg"):
        circuit, _ = sax.circuit(net, models, backend=backend)
        np.testing.assert_allclose(circuit(w={"gain": gain})["in0", "out0"], expected)
        np.testing.assert_allclose(
            jax.jit(circuit)(w={"gain": gain})["in0", "out0"], expected
        )
        gradient = jax.grad(
            lambda g, circuit=circuit: jnp.real(
                circuit(w={"gain": g})["in0", "out0"]
            ).sum()
        )(0.8)
        np.testing.assert_allclose(gradient, 0.15)


@pytest.mark.parametrize("backend", ["forward", "FORWARD"])
def test_removed_forward_backend_rejected(backend: str) -> None:
    with pytest.raises(ValueError, match=r"Invalid backend.*forward"):
        sax.circuit({}, backend=backend)  # type: ignore[call-overload]
