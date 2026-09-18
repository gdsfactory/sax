import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sax


@pytest.mark.parametrize("batched", [False, True])
def test_duplicate_coo_sums_consistently(batched: bool) -> None:
    values = jnp.array([2 + 1j, 3 - 1j, 4j, -4j])
    if batched:
        values = jnp.stack([values, values * 2])
    coo: sax.SCoo = (
        np.array([1, 1, 0, 0]),
        np.array([0, 0, 1, 1]),
        values,
        {"a": 0, "b": 1},
    )
    direct = sax.sdict(coo)
    dense = sax.sdict(sax.sdense(coo))
    for key, value in direct.items():
        np.testing.assert_allclose(value, dense[key])
    np.testing.assert_allclose(direct["a", "b"], [5, 10] if batched else 5)
    np.testing.assert_allclose(direct["b", "a"], 0)
    np.testing.assert_allclose(sax.sdense(sax.scoo(direct))[0], sax.sdense(coo)[0])


def test_duplicate_coo_differentiation() -> None:
    def total(values: jax.Array) -> jax.Array:
        coo: sax.SCoo = (np.array([1, 1]), np.array([0, 0]), values, {"a": 0, "b": 1})
        return jnp.real(sax.sdict(coo)["a", "b"])

    np.testing.assert_allclose(jax.jit(total)(jnp.array([2.0, 3.0])), 5)
    np.testing.assert_allclose(jax.grad(total)(jnp.array([2.0, 3.0])), [1, 1])
