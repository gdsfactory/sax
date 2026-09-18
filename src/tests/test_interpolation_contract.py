import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import sax


def test_grid_interpolation_and_boundary_clamping() -> None:
    grid = sax.to_xarray(
        pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 2.0, 4.0]}), target_names=["y"]
    )
    evaluate = jax.jit(lambda x: sax.interpolate_xarray(grid, x=x)["y"])
    np.testing.assert_allclose(
        evaluate(jnp.array([-0.5, 0.5, 1.5, 3.0])), [0.0, 1.0, 3.0, 4.0]
    )
    np.testing.assert_allclose(jax.grad(lambda x: evaluate(x).sum())(0.5), 2.0)
