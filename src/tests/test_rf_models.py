from itertools import product

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
        self._assert_s_param(s, ("o1", "o1"), 1 / (1 + 1 / 75))

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
