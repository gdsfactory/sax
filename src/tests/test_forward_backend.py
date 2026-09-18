import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sax
from sax.backends.forward_only import evaluate_circuit_forward


def _split() -> sax.SDict:
    return {("in0", "out0"): 0.2, ("in0", "out1"): 0.3}


def _merge() -> sax.SDict:
    return {("in0", "out0"): 0.4, ("in1", "out0"): 0.5}


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
    for backend in ("forward", "klu", "fg"):
        circuit, _ = sax.circuit(net, models, backend=backend)
        np.testing.assert_allclose(circuit(w={"gain": gain})["in0", "out0"], expected)
        np.testing.assert_allclose(
            jax.jit(circuit)(w={"gain": gain})["in0", "out0"], expected
        )
        gradient = jax.grad(
            lambda g: jnp.real(circuit(w={"gain": g})["in0", "out0"]).sum()
        )(0.8)
        np.testing.assert_allclose(gradient, 0.15)


def test_directed_cycle_rejected() -> None:
    analyzed = (
        {"a,out0": "b,in0", "b,out0": "a,in0"},
        {"in0": "a,in0", "out0": "b,out0"},
    )
    with pytest.raises(ValueError, match="acyclic directed signal graph"):
        evaluate_circuit_forward(analyzed, {"a": _waveguide(), "b": _waveguide()})
