import jax
import jax.numpy as jnp
import numpy as np
import pytest
from kfnetlist import Netlist, NetlistPort, PortRef

import sax


def _one_waveguide() -> Netlist:
    netlist = Netlist()
    netlist.create_inst("w", "pdk", "wg")
    for name, port in (("in", "in0"), ("out", "out0")):
        netlist.create_port(name)
        netlist.create_net(NetlistPort(name), PortRef("w", port))
    return netlist


def test_get_modes_unique_natural_order_all_formats() -> None:
    data: sax.SDict = {("a@mode10", "b@mode10"): 1.0, ("a@mode2", "b@mode2"): 2.0}
    for matrix in (data, sax.scoo(data), sax.sdense(data)):
        assert sax.get_modes(matrix) == ("mode2", "mode10")


def test_reciprocal_first_direction_wins_without_mutation() -> None:
    data: sax.SDict = {
        ("a", "b"): jnp.array([1j, 2j]),
        ("b", "a"): jnp.array([3j, 4j]),
        ("a", "a"): jnp.array([0.1, 0.2]),
    }
    result = sax.reciprocal(data)
    np.testing.assert_allclose(result["a", "b"], [1j, 2j])
    np.testing.assert_allclose(result["b", "a"], [1j, 2j])
    np.testing.assert_allclose(data["b", "a"], [3j, 4j])
    np.testing.assert_allclose(result["a", "a"], [0.1, 0.2])
    reverse = sax.reciprocal(dict(reversed(list(data.items()))))
    np.testing.assert_allclose(reverse["a", "b"], [3j, 4j])


def test_reciprocal_jit_gradient() -> None:
    def response(value: float) -> jax.Array:
        return sax.reciprocal(
            {("a", "b"): jnp.asarray(value), ("b", "a"): jnp.asarray(9.0)}
        )["b", "a"]

    np.testing.assert_allclose(jax.jit(response)(2.0), 2.0)
    np.testing.assert_allclose(jax.grad(response)(2.0), 1.0)


def test_invalid_return_type_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid return_type"):
        sax.circuit(_one_waveguide(), {"wg": sax.models.straight}, return_type="typo")  # type: ignore[call-overload]


@pytest.mark.parametrize("return_type", ["SDict", "SCoo", "SDense"])
def test_documented_return_types(return_type: str) -> None:
    model, _ = sax.circuit(
        _one_waveguide(), {"wg": sax.models.straight}, return_type=return_type
    )  # type: ignore[call-overload]
    response = model()
    if return_type == "SDict":
        assert isinstance(response, dict)
    else:
        assert len(response) == (4 if return_type == "SCoo" else 2)
