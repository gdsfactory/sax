import numpy as np
import pytest

import sax


@pytest.mark.parametrize(
    "components",
    [("top_level",), ("top_level", "child"), ("top_level", "child", "grandchild")],
)
def test_dependency_cycles_have_explicit_diagnostic(
    components: tuple[str, ...],
) -> None:
    rec = {
        name: {
            "instances": {"sub": {"component": components[(i + 1) % len(components)]}},
            "ports": {"in": "sub,in", "out": "sub,out"},
        }
        for i, name in enumerate(components)
    }
    with pytest.raises(ValueError, match="dependency cycles"):
        sax.circuit(rec)
    with pytest.raises(ValueError, match="dependency cycles"):
        sax.get_required_circuit_models(rec)


def test_optical_feedback_is_not_a_hierarchy_cycle() -> None:
    net = {
        "instances": {"c": "coupler", "w": "waveguide"},
        "connections": {"c,out1": "w,in0", "w,out0": "c,in1"},
        "ports": {"in": "c,in0", "out": "c,out0"},
    }
    models = {"coupler": sax.models.coupler_ideal, "waveguide": sax.models.straight}
    klu, _ = sax.circuit(net, models, backend="klu")
    fg, _ = sax.circuit(net, models, backend="fg")
    for key, value in klu().items():
        assert np.isfinite(value).all()
        np.testing.assert_allclose(value, fg()[key], atol=1e-10)
