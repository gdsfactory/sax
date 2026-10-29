from itertools import product

import jax
import jax.numpy as jnp
import pytest

from sax.models import rf


class TestRFModels:
    """Test suite for RF model components."""

    @pytest.fixture
    def freq_array(self) -> jnp.ndarray:
        """Frequency array fixture."""
        return jnp.linspace(1e9, 10e9, 10)

    @pytest.fixture
    def freq_single(self) -> float:
        """Single frequency fixture."""
        return 1e9

    @staticmethod
    def _assert_s_params_dict(s: dict, expected_shape: tuple | None = None) -> None:
        """Helper method to assert common S-parameter dictionary properties."""
        assert isinstance(s, dict)
        if expected_shape is not None:
            assert s[("o1", "o1")].shape == expected_shape

    @staticmethod
    def _assert_s_param(s: dict, port_pair: tuple, expected_value: complex) -> None:
        """Helper method to assert a specific S-parameter value."""
        assert jnp.allclose(s[port_pair], expected_value)

    def test_gamma_0_load(self, freq_array: jnp.ndarray) -> None:
        """Test gamma_0_load with frequency array."""
        s = rf.gamma_0_load(f=freq_array, gamma_0=0.5, n_ports=2)

        self._assert_s_params_dict(s, expected_shape=(len(freq_array),))
        self._assert_s_param(s, ("o1", "o1"), 0.5)
        self._assert_s_param(s, ("o1", "o2"), 0)

    def test_tee(self, freq_array: jnp.ndarray) -> None:
        """Test tee splitter."""
        s = rf.tee(f=freq_array)

        self._assert_s_params_dict(s, expected_shape=(len(freq_array),))
        self._assert_s_param(s, ("o1", "o1"), -1 / 3)
        self._assert_s_param(s, ("o1", "o2"), 2 / 3)

    def test_impedance(self, freq_single: float) -> None:
        """Test impedance element."""
        s = rf.impedance(f=freq_single, z=75, z0=50)

        self._assert_s_params_dict(s, expected_shape=())
        self._assert_s_param(s, ("o1", "o1"), 75 / (75 + 100))

    def test_admittance(self, freq_single: float) -> None:
        """Test admittance element."""
        s = rf.admittance(f=freq_single, y=1 / 75)

        self._assert_s_params_dict(s, expected_shape=())
        self._assert_s_param(s, ("o1", "o1"), 75 / (75 + 100))
        self._assert_s_param(s, ("o1", "o2"), 100 / (75 + 100))

    @pytest.mark.parametrize("z0", [25.0, 50.0, 75.0])
    @pytest.mark.parametrize("y", [1 / 75, 0.02j, 0.01 - 0.03j])
    def test_admittance_matches_impedance(self, y: complex, z0: float) -> None:
        """Reciprocal impedance and admittance describe the same series element."""
        f = jnp.linspace(1e9, 10e9, 6).reshape(2, 3)
        actual = rf.admittance(f=f, y=y, z0=z0)
        expected = rf.impedance(f=f, z=1 / y, z0=z0)

        assert actual.keys() == expected.keys()
        for port_pair in expected:
            assert actual[port_pair].shape == f.shape
            assert jnp.allclose(actual[port_pair], expected[port_pair])

    def test_admittance_broadcasting(self) -> None:
        """Frequency, admittance, and reference impedance broadcast together."""
        f = jnp.array([[1e9], [2e9], [3e9]])
        y = jnp.array([[0.01 + 0.02j, 0.02 - 0.03j]])
        z0 = jnp.array([[25.0], [50.0], [75.0]])
        actual = rf.admittance(f=f, y=y, z0=z0)
        expected = rf.impedance(f=f, z=1 / y, z0=z0)

        for port_pair in expected:
            assert actual[port_pair].shape == (3, 2)
            assert jnp.allclose(actual[port_pair], expected[port_pair])

    def test_admittance_open_circuit(self) -> None:
        """Zero series admittance is an open circuit, including under JIT and grad."""
        s = jax.jit(rf.admittance)(f=jnp.array([1e9, 2e9]), y=0.0, z0=75.0)
        for port_pair, value in s.items():
            assert value.shape == (2,)
            assert jnp.allclose(value, float(port_pair[0] == port_pair[1]))

        derivative = jax.jit(
            jax.grad(lambda y: rf.admittance(f=1e9, y=y, z0=75.0)["o1", "o1"])
        )(0.0)
        assert jnp.allclose(derivative, -150.0)

    def test_admittance_lossless(self) -> None:
        """A reactive series element has a unitary scattering matrix."""
        s = rf.admittance(f=1e9, y=0.02j)
        matrix = jnp.array(
            [[s["o1", "o1"], s["o1", "o2"]], [s["o2", "o1"], s["o2", "o2"]]]
        )
        assert jnp.allclose(matrix.conj().T @ matrix, jnp.eye(2), atol=1e-12)

    def test_capacitor(self, freq_single: float) -> None:
        """Test capacitor element."""
        s = rf.capacitor(f=freq_single, capacitance=1e-12, z0=50)

        self._assert_s_params_dict(s, expected_shape=())

        angular_frequency = 2 * jnp.pi * freq_single
        capacitor_impedance = 1 / (1j * angular_frequency * 1e-12)
        expected_s11 = capacitor_impedance / (capacitor_impedance + 100)
        self._assert_s_param(s, ("o1", "o1"), expected_s11)

    def test_inductor(self, freq_single: float) -> None:
        """Test inductor element."""
        s = rf.inductor(f=freq_single, inductance=1e-9, z0=50)

        self._assert_s_params_dict(s)

        angular_frequency = 2 * jnp.pi * freq_single
        inductor_impedance = 1j * angular_frequency * 1e-9
        expected_s11 = inductor_impedance / (inductor_impedance + 100)
        self._assert_s_param(s, ("o1", "o1"), expected_s11)

    @pytest.mark.parametrize("n_ports", [1, 2, 3])
    def test_electrical_short(self, freq_array: jnp.ndarray, n_ports: int) -> None:
        """Test electrical_short with frequency array."""
        s = rf.electrical_short(f=freq_array, n_ports=n_ports)

        self._assert_s_params_dict(s, expected_shape=(len(freq_array),))
        for i, j in product(range(1, n_ports + 1), repeat=2):
            expected_value = -1 if i == j else 0
            self._assert_s_param(s, (f"o{i}", f"o{j}"), expected_value)

    @pytest.mark.parametrize("n_ports", [1, 2, 3])
    def test_electrical_open(self, freq_array: jnp.ndarray, n_ports: int) -> None:
        """Test electrical_open with frequency array."""
        s = rf.electrical_open(f=freq_array, n_ports=n_ports)

        self._assert_s_params_dict(s, expected_shape=(len(freq_array),))
        for i, j in product(range(1, n_ports + 1), repeat=2):
            expected_value = 1 if i == j else 0
            self._assert_s_param(s, (f"o{i}", f"o{j}"), expected_value)

    def test_resistor(self, freq_single: float) -> None:
        """Test resistor element."""
        s = rf.resistor(f=freq_single, resistance=100, z0=50)

        self._assert_s_params_dict(s, expected_shape=())
        # S11 = R / (R + 2*Z0) = 100 / (100 + 100) = 0.5
        self._assert_s_param(s, ("o1", "o1"), 0.5)

    def test_lc_shunt_component(self) -> None:
        """Test LC shunt component circuit resonance."""
        L = 1e-9
        C = 1e-12
        f0 = 1 / (2 * jnp.pi * jnp.sqrt(L * C))
        f = jnp.array([f0 * 0.5, f0, f0 * 2.0])

        s = rf.lc_shunt_component(
            f=f,
            inductance=L,
            capacitance=C,
            z0=50,
        )

        self._assert_s_params_dict(s, expected_shape=(3,))
        # At resonance, S11 should be 1.0 (open circuit for shunted parallel LC)
        assert jnp.abs(s[("o1", "o1")][1]) > 0.99

    def test_multidimensional_frequency(self) -> None:
        """Test models with multidimensional frequency arrays."""
        f = jnp.linspace(1e9, 10e9, 12).reshape(3, 4)
        s = rf.gamma_0_load(f=f, gamma_0=0.5, n_ports=1)
        assert s[("o1", "o1")].shape == (3, 4)
        assert jnp.allclose(s[("o1", "o1")], 0.5)
