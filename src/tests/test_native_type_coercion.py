import jax.numpy as jnp
import pytest
from pydantic import TypeAdapter, ValidationError

import sax


@pytest.mark.parametrize("target", [float, complex])
@pytest.mark.parametrize(
    "value",
    [0.0, 3, 3 + 2j, jnp.array(3.0, dtype=complex), jnp.array(3, dtype=int), object()],
)
def test_native_coercion_delegates_to_pydantic(target: type, value: object) -> None:
    try:
        expected = TypeAdapter(target).validate_python(value, strict=False)
    except ValidationError:
        assert sax.try_into[target](value) is None
        with pytest.raises(TypeError):
            sax.into[target](value)
    else:
        assert sax.into[target](value) == expected
        assert sax.try_into[target](value) == expected
