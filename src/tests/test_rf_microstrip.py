"""Tests for microstrip electromagnetic analysis functions."""

import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from sax.models.rf import (
    microstrip as straight_microstrip,
)
from sax.models.rf import (
    microstrip_epsilon_eff,
    microstrip_thickness_correction,
    microstrip_z0,
)


class TestMicrostripEpsilonEff:
    """Tests for microstrip effective permittivity."""

    def test_vacuum_substrate(self) -> None:
        """With ep_r=1, ep_eff should be 1."""
        ep = microstrip_epsilon_eff(10e-6, 500e-6, 1.0)
        assert_allclose(float(ep), 1.0, atol=1e-10)

    def test_bounded(self) -> None:
        """1 < ep_eff < ep_r for any substrate."""
        ep_r = 11.45
        ep = microstrip_epsilon_eff(10e-6, 500e-6, ep_r)
        assert 1.0 < float(ep) < ep_r

    def test_wide_strip_approaches_ep_r(self) -> None:
        """For very wide strips (w/h >> 1), ep_eff → ep_r."""
        ep_r = 11.45
        ep = microstrip_epsilon_eff(1e-3, 1e-6, ep_r)  # w/h = 1000
        assert float(ep) > 0.9 * ep_r

    def test_narrow_strip_approaches_average(self) -> None:
        """For very narrow strips (w/h << 1), ep_eff → (ep_r+1)/2."""
        ep_r = 11.45
        ep = microstrip_epsilon_eff(1e-9, 1e-3, ep_r)  # w/h = 1e-6
        assert_allclose(float(ep), (ep_r + 1) / 2, rtol=0.1)

    def test_increases_with_width(self) -> None:
        """ep_eff increases as strip gets wider (more field in substrate)."""
        ep1 = microstrip_epsilon_eff(5e-6, 500e-6, 11.45)
        ep2 = microstrip_epsilon_eff(100e-6, 500e-6, 11.45)
        assert float(ep2) > float(ep1)


class TestMicrostripZ0:
    """Tests for microstrip characteristic impedance."""

    def test_narrow_strip_high_impedance(self) -> None:
        """Narrow strip (w/h < 1) → high impedance."""
        ep = microstrip_epsilon_eff(1e-6, 500e-6, 11.45)
        z0 = microstrip_z0(1e-6, 500e-6, ep)
        assert float(z0) > 100.0

    def test_wide_strip_low_impedance(self) -> None:
        """Wide strip (w/h >> 1) → low impedance."""
        ep = microstrip_epsilon_eff(1e-3, 500e-6, 11.45)
        z0 = microstrip_z0(1e-3, 500e-6, ep)
        assert float(z0) < 35.0

    def test_typical_50_ohm(self) -> None:
        r"""A common 50 Ω microstrip on alumina (ep_r=9.8, h=0.635mm) has w ≈ 0.6mm."""
        ep = microstrip_epsilon_eff(0.6e-3, 0.635e-3, 9.8)
        z0 = microstrip_z0(0.6e-3, 0.635e-3, ep)
        assert_allclose(float(z0), 50.0, atol=5.0)

    def test_jit_compatible(self) -> None:
        """Function can be JIT-compiled."""
        jitted = jax.jit(microstrip_z0)
        result = jitted(10e-6, 500e-6, 6.2)
        assert jnp.isfinite(result)


class TestMicrostripThicknessCorrection:
    """Tests for microstrip conductor thickness correction."""

    def test_reduces_impedance(self) -> None:
        """Thickness correction should reduce Z0 (wider effective strip)."""
        ep0 = microstrip_epsilon_eff(10e-6, 500e-6, 11.45)
        z0_0 = microstrip_z0(10e-6, 500e-6, ep0)
        _, _ep_t, z0_t = microstrip_thickness_correction(
            10e-6, 500e-6, 0.2e-6, 11.45, ep0
        )
        assert float(z0_t) < float(z0_0)

    def test_thickness_zero(self) -> None:
        """With thickness=0, results should match the uncorrected case."""
        ep0 = microstrip_epsilon_eff(10e-6, 500e-6, 11.45)
        _, ep_t, z0_t = microstrip_thickness_correction(10e-6, 500e-6, 0.0, 11.45, ep0)
        z0_0 = microstrip_z0(10e-6, 500e-6, ep0)
        assert_allclose(float(ep_t), float(ep0), rtol=1e-10)
        assert_allclose(float(z0_t), float(z0_0), rtol=1e-10)


class TestStraightMicrostrip:
    """Tests for the straight_microstrip model."""

    def test_zero_length_transmission(self) -> None:
        """Zero-length microstrip → near-unity transmission."""
        f = jnp.array([5e9])
        result = straight_microstrip(f=f, length=0.0)
        s21 = jnp.abs(result[("o1", "o2")])
        assert_allclose(float(s21.squeeze()), 1.0, atol=1e-6)

    def test_phase_shift_increases_with_length(self) -> None:
        """Longer lines should have more phase shift."""
        f = jnp.array([5e9])
        r1 = straight_microstrip(f=f, length=1000)
        r2 = straight_microstrip(f=f, length=2000)
        phase1 = jnp.angle(r1[("o1", "o2")]).squeeze()
        phase2 = jnp.angle(r2[("o1", "o2")]).squeeze()
        # Phase shift roughly doubles (unwrapped)
        assert jnp.abs(phase2) > jnp.abs(phase1) - 0.01

    def test_lossy_attenuates(self) -> None:
        """With tand > 0, transmission should be reduced."""
        f = jnp.array([5e9])
        r_lossless = straight_microstrip(f=f, length=10000, tand=0.0)
        r_lossy = straight_microstrip(f=f, length=10000, tand=0.01)
        s21_ll = jnp.abs(r_lossless[("o1", "o2")]).squeeze()
        s21_ly = jnp.abs(r_lossy[("o1", "o2")]).squeeze()
        assert float(s21_ly) < float(s21_ll)

    def test_jit_compatible(self) -> None:
        """Microstrip straight model can be JIT-compiled."""
        jitted = jax.jit(lambda f, length: straight_microstrip(f=f, length=length))
        f = jnp.array([5e9])
        result = jitted(f, 1000.0)
        assert jnp.isfinite(result[("o1", "o2")])
