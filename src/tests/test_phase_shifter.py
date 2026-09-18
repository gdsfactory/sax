import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sax


@pytest.mark.parametrize("length", [0.0, 1.0, 10.0])
def test_phase_shifter_loss_is_db_per_micrometer(length: float) -> None:
    result = sax.models.phase_shifter(wl=jnp.array([1.5, 1.6]), length=length, loss=0.2)
    for value in result.values():
        np.testing.assert_allclose(jnp.abs(value), 10 ** (-0.2 * length / 20))


def test_phase_shifter_voltage_phase_and_gradient() -> None:
    def transmission(voltage: float) -> jax.Array:
        return next(
            iter(sax.models.phase_shifter(length=0.0, voltage=voltage).values())
        )

    np.testing.assert_allclose(transmission(0.5), 1j, atol=1e-12)
    derivative = jax.grad(lambda v: jnp.imag(transmission(v)))(0.0)
    np.testing.assert_allclose(derivative, jnp.pi)
