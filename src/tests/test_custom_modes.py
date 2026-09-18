import numpy as np
import pytest

import sax


def _model() -> sax.SDict:
    return sax.reciprocal({("a", "b"): 0.75})


@pytest.mark.parametrize("format_name", ["sdict", "scoo", "sdense"])
@pytest.mark.parametrize("modes", [("X",), ("mode2", "mode10")])
@pytest.mark.parametrize("wrapped", [False, True])
def test_custom_modes_all_formats(
    format_name: str, modes: tuple[str, ...], wrapped: bool
) -> None:
    convert = getattr(sax, format_name)
    if wrapped:
        model = sax.multimode(convert(_model), modes=modes)
        result = model()
    else:
        result = sax.multimode(convert(_model()), modes=modes)
    assert set(sax.get_ports(result)) == {f"{p}@{m}" for p in ("a", "b") for m in modes}
    values = sax.sdict(result)
    for mode in modes:
        np.testing.assert_allclose(values[f"a@{mode}", f"b@{mode}"], 0.75)
        selected = sax.sdict(sax.singlemode(result, mode=mode))
        np.testing.assert_allclose(selected["a", "b"], 0.75)
    for first in modes:
        for second in modes:
            if first != second:
                np.testing.assert_allclose(
                    values.get((f"a@{first}", f"b@{second}"), 0), 0
                )
