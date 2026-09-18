import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from sax.fit import eval_neural_fit, neural_fit


@pytest.mark.parametrize("constant_feature", [True, False])
@pytest.mark.parametrize("constant_target", [True, False])
def test_constant_columns_finite_and_export_consistent(
    constant_feature: bool, constant_target: bool
) -> None:
    frame = pd.DataFrame(
        {
            "x": [2.0, 2.0, 2.0] if constant_feature else [1.0, 2.0, 3.0],
            "y": [7.0, 7.0, 7.0] if constant_target else [3.0, 5.0, 7.0],
        },
        index=[10, 20, 30],
    )
    result = neural_fit(
        frame, targets=["y"], hidden_dims=(2,), num_epochs=5, progress_bar=False
    )
    assert np.isfinite(result["final_loss"])
    assert np.all(np.asarray(result["X_norm"].std) > 0)
    assert np.all(np.asarray(result["Y_norm"].std) > 0)
    if constant_feature:
        np.testing.assert_array_equal(result["X_norm"].std, [[1.0]])
    if constant_target:
        np.testing.assert_array_equal(result["Y_norm"].std, [[1.0]])
    prediction = result["predict_fn"](frame)
    assert prediction.index.equals(frame.index)
    assert len(prediction) == len(frame)
    assert np.isfinite(prediction["y_pred"]).all()
    direct = result["predict_fn"](jnp.asarray(frame[["x"]].values))
    exported = eval_neural_fit(result, "y", x=jnp.asarray(frame["x"].values))
    np.testing.assert_allclose(exported, direct[:, 0], rtol=1e-5, atol=1e-5)
    if constant_feature and constant_target:
        np.testing.assert_allclose(prediction["y_pred"], 7.0)
