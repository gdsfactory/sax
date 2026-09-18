import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sax


def _model(gain: sax.FloatArrayLike = 1.0) -> sax.SDict:
    return sax.reciprocal({("in0", "out0"): jnp.asarray(gain)})


def _circuit(backend: sax.BackendLike) -> sax.Model:
    circuit, _ = sax.circuit(
        {
            "instances": {"a": "wg", "b": "wg"},
            "connections": {"a,out0": "b,in0"},
            "ports": {"in": "a,in0", "out": "b,out0"},
        },
        {"wg": _model},
        backend=backend,
    )
    return circuit


@pytest.mark.parametrize(
    "shapes", [((3, 1), (1, 4)), ((), (2, 3)), ((2, 1, 3), (4, 1)), ((), ())]
)
@pytest.mark.parametrize("backend", ["klu", "fg"])
def test_joint_broadcast(
    shapes: tuple[tuple[int, ...], tuple[int, ...]], backend: sax.BackendLike
) -> None:
    a = jnp.ones(shapes[0]) * 0.8
    b = jnp.ones(shapes[1]) * 0.5
    model = _circuit(backend)
    for evaluate in (model, jax.jit(model)):
        result = sax.sdict(evaluate(a={"gain": a}, b={"gain": b}))["in", "out"]
        assert result.shape == jnp.broadcast_shapes(a.shape, b.shape)
        np.testing.assert_allclose(result, a * b)
    derivative = jax.grad(
        lambda g: jnp.real(
            sax.sdict(model(a={"gain": a * g}, b={"gain": b}))["in", "out"]
        ).sum()
    )(1.0)
    np.testing.assert_allclose(derivative, jnp.sum(a * b))


def test_incompatible_shapes_rejected() -> None:
    model = _circuit("klu")
    with pytest.raises(ValueError, match="[Ii]ncompatible shapes"):
        model(a={"gain": jnp.ones((2,))}, b={"gain": jnp.ones((3,))})
