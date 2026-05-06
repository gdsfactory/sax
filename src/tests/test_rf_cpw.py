"""Tests for CPW electromagnetic analysis functions."""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from sax import C_M_S

pytest.importorskip("jaxellip")

from hypothesis import given, settings
from hypothesis import strategies as st

from sax.models.rf import (
    coplanar_waveguide,
    cpw_epsilon_eff,
    cpw_thickness_correction,
    cpw_z0,
    ellipk_ratio,
    propagation_constant,
    transmission_line_s_params,
)


class TestEllipkRatio:
    """Tests for the elliptic integral ratio K(m)/K(1-m)."""

    def test_symmetry(self) -> None:
        """K(m)/K(1-m) = 1 / (K(1-m)/K(m)) → ratio(m) * ratio(1-m) == 1."""
        m = 0.3
        assert_allclose(
            float(ellipk_ratio(m) * ellipk_ratio(1.0 - m)),
            1.0,
            atol=1e-10,
        )

    def test_at_half(self) -> None:
        """K(0.5) / K(0.5) == 1."""
        assert_allclose(float(ellipk_ratio(0.5)), 1.0, atol=1e-10)

    @given(m=st.floats(min_value=0.01, max_value=0.99))
    @settings(deadline=None)
    def test_positive(self, m: float) -> None:
        """Ratio should always be positive for 0 < m < 1."""
        assert float(ellipk_ratio(m)) > 0


class TestCPWEpsilonEff:
    """Tests for CPW effective permittivity."""

    def test_vacuum_substrate(self) -> None:
        """With ep_r=1 (vacuum), ep_eff should be 1."""
        ep = cpw_epsilon_eff(10e-6, 6e-6, 500e-6, 1.0)
        assert_allclose(float(ep), 1.0, atol=1e-6)

    def test_infinite_substrate_limit(self) -> None:
        """For very thick substrate, ep_eff → (ep_r+1)/2."""
        ep_r = 11.45
        # Use very thick substrate (h >> w)
        ep = cpw_epsilon_eff(10e-6, 6e-6, 100e-3, ep_r)
        assert_allclose(float(ep), (ep_r + 1) / 2, rtol=1e-4)

    def test_bounded(self) -> None:
        """1 < ep_eff < ep_r for any substrate."""
        ep_r = 11.45
        ep = cpw_epsilon_eff(10e-6, 6e-6, 500e-6, ep_r)
        assert 1.0 < float(ep) < ep_r

    def test_increases_with_ep_r(self) -> None:
        """ep_eff should increase with substrate permittivity."""
        ep1 = cpw_epsilon_eff(10e-6, 6e-6, 500e-6, 4.0)
        ep2 = cpw_epsilon_eff(10e-6, 6e-6, 500e-6, 11.45)
        assert float(ep2) > float(ep1)

    def test_jit_compatible(self) -> None:
        """Function can be JIT-compiled."""
        jitted = jax.jit(cpw_epsilon_eff)
        result = jitted(10e-6, 6e-6, 500e-6, 11.45)
        assert jnp.isfinite(result)


class TestCPWZ0:
    """Tests for CPW characteristic impedance."""

    def test_default_cpw_approx_50_ohm(self) -> None:
        """Default CPW dimensions (w=10, s=6) should give ~50 Ω."""
        ep = cpw_epsilon_eff(10e-6, 6e-6, 500e-6, 11.45)
        z0 = cpw_z0(10e-6, 6e-6, ep)
        assert_allclose(float(z0), 50.0, atol=2.0)  # within 2 Ω

    def test_narrow_conductor_high_impedance(self) -> None:
        """Narrow conductor (small w) → high impedance."""
        ep = cpw_epsilon_eff(1e-6, 20e-6, 500e-6, 11.45)
        z0 = cpw_z0(1e-6, 20e-6, ep)
        assert float(z0) > 100.0

    def test_wide_conductor_low_impedance(self) -> None:
        """Wide conductor (large w) → low impedance."""
        ep = cpw_epsilon_eff(100e-6, 2e-6, 500e-6, 11.45)
        z0 = cpw_z0(100e-6, 2e-6, ep)
        assert float(z0) < 25.0

    def test_jit_compatible(self) -> None:
        """Function can be JIT-compiled."""
        jitted = jax.jit(cpw_z0)
        result = jitted(10e-6, 6e-6, 6.2)
        assert jnp.isfinite(result)


class TestCPWThicknessCorrection:
    """Tests for GGBB96 conductor thickness correction."""

    def test_thin_conductor_small_correction(self) -> None:
        """Very thin conductor should produce small corrections."""
        ep0 = cpw_epsilon_eff(10e-6, 6e-6, 500e-6, 11.45)
        z0_0 = cpw_z0(10e-6, 6e-6, ep0)
        ep_t, z0_t = cpw_thickness_correction(10e-6, 6e-6, 1e-9, ep0)
        # Correction should be small for t = 1 nm
        assert_allclose(float(ep_t), float(ep0), rtol=0.01)
        assert_allclose(float(z0_t), float(z0_0), rtol=0.01)

    def test_reduces_impedance(self) -> None:
        """Thickness correction should reduce Z0 (wider effective conductor)."""
        ep0 = cpw_epsilon_eff(10e-6, 6e-6, 500e-6, 11.45)
        z0_0 = cpw_z0(10e-6, 6e-6, ep0)
        _ep_t, z0_t = cpw_thickness_correction(10e-6, 6e-6, 0.2e-6, ep0)
        assert float(z0_t) < float(z0_0)

    def test_thickness_zero(self) -> None:
        """With thickness=0, results should match the uncorrected case."""
        ep0 = cpw_epsilon_eff(10e-6, 6e-6, 500e-6, 11.45)
        ep_t, z0_t = cpw_thickness_correction(10e-6, 6e-6, 0.0, ep0)
        z0_0 = cpw_z0(10e-6, 6e-6, ep0)
        assert_allclose(float(ep_t), float(ep0), rtol=1e-10)
        assert_allclose(float(z0_t), float(z0_0), rtol=1e-10)


class TestPropagationConstant:
    """Tests for the complex propagation constant."""

    def test_lossless_purely_imaginary(self) -> None:
        """For tand=0, gamma should be purely imaginary."""
        gamma = propagation_constant(5e9, 6.225, tand=0.0)
        assert_allclose(float(jnp.real(gamma)), 0.0, atol=1e-20)
        assert float(jnp.imag(gamma)) > 0

    def test_phase_velocity(self) -> None:
        """beta = ω√ep_eff/c_0, so v_p = ω/beta = c_0/√ep_eff."""
        ep_eff = 6.225
        f = 5e9
        gamma = propagation_constant(f, ep_eff)
        beta = float(jnp.imag(gamma))
        v_p = 2 * jnp.pi * f / beta
        assert_allclose(float(v_p), C_M_S / jnp.sqrt(ep_eff), rtol=1e-8)

    def test_lossy_has_real_part(self) -> None:
        """For tand > 0, gamma should have a positive real part (attenuation)."""
        gamma = propagation_constant(5e9, 6.225, tand=0.01, ep_r=11.45)
        assert float(jnp.real(gamma)) > 0

    def test_scales_with_frequency(self) -> None:
        """beta should scale linearly with frequency."""
        g1 = propagation_constant(5e9, 6.225)
        g2 = propagation_constant(10e9, 6.225)
        assert_allclose(
            float(jnp.imag(g2)),
            2.0 * float(jnp.imag(g1)),
            rtol=1e-8,
        )


class TestTransmissionLineSParams:
    """Tests for ABCD→S-parameter conversion."""

    def test_zero_length_identity(self) -> None:
        """Zero-length line → S11=0, S21=1."""
        gamma = 1j * 100.0
        s11, s21 = transmission_line_s_params(gamma, 50.0, 0.0)
        assert_allclose(float(jnp.abs(s11)), 0.0, atol=1e-10)
        assert_allclose(float(jnp.abs(s21)), 1.0, atol=1e-10)

    def test_matched_impedance_no_reflection(self) -> None:
        """With z_ref = z0, S11 should be zero."""
        gamma = jnp.array([1j * 100.0])
        s11, _ = transmission_line_s_params(gamma, 50.0, 0.001)
        assert_allclose(float(jnp.abs(s11[0])), 0.0, atol=1e-10)

    def test_mismatched_impedance_reflection(self) -> None:
        """With z_ref ≠ z0, S11 should be non-zero."""
        gamma = jnp.array([1j * 100.0])
        s11, _ = transmission_line_s_params(gamma, 50.0, 0.1, z_ref=75.0)
        assert float(jnp.abs(s11[0])) > 0.01

    def test_lossless_passivity(self) -> None:
        """For lossless line: |S11|² + |S21|² = 1."""
        gamma = jnp.array([1j * 200.0])
        s11, s21 = transmission_line_s_params(gamma, 50.0, 0.01, z_ref=75.0)
        power = jnp.abs(s11) ** 2 + jnp.abs(s21) ** 2
        assert_allclose(float(power[0]), 1.0, atol=1e-10)

    def test_reciprocal(self) -> None:
        """S21 = S12 for a reciprocal 2-port network."""
        gamma = jnp.array([1j * 150.0])
        z0 = 50.0
        length = 0.02
        z_ref = 75.0
        _s11, s21 = transmission_line_s_params(gamma, z0, length, z_ref=z_ref)

        theta = gamma * length
        a = jnp.cosh(theta)
        b = z0 * jnp.sinh(theta)
        c = jnp.sinh(theta) / z0
        d = a

        denom = a + b / z_ref + c * z_ref + d
        s12_expected = 2.0 / denom

        assert_allclose(s21, s12_expected, atol=1e-10)
        assert jnp.isfinite(s21[0])


class TestStraightJIT:
    """Tests for JIT compilation of the straight CPW model."""

    def test_zero_length(self) -> None:
        """Zero-length CPW → near-unity transmission."""
        f = jnp.array([5e9])
        result = coplanar_waveguide(f=f, length=0.0)
        s21 = jnp.abs(result[("o1", "o2")])
        assert_allclose(float(s21.squeeze()), 1.0, atol=1e-6)

    def test_straight_matches_non_jit(self) -> None:
        """JIT-compiled straight should give same results as non-JIT."""
        f = jnp.linspace(4e9, 8e9, 10)

        result_nojit = coplanar_waveguide(f=f, length=1000)

        jitted_inner = jax.jit(lambda f, length: coplanar_waveguide(f=f, length=length))
        result_jit = jitted_inner(f, 1000.0)

        for key in result_nojit:
            assert_allclose(
                result_nojit[key],
                result_jit[key],
                atol=1e-10,
                err_msg=f"Mismatch for {key}",
            )
