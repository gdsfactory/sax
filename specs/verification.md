# Verification and evidence

## Baseline runs

Source revision: `9ebd77d0844b48dd1268066372db929946069f34` (SAX 0.18.2).
Environment: macOS arm64, Python 3.12.14, JAX 0.9.2. In this environment importing
SAX changed JAX x64 configuration from false to true; results below are not evidence
for every precision/device combination.

| Check | Result |
| --- | --- |
| Full `src/tests` suite, including notebook execution | **266 passed, 1 skipped** |
| Non-notebook subset (also run separately) | **262 passed, 1 skipped** |
| Four `src/tests/nbs/` notebooks through pytest/Papermill (also run separately) | **4 passed** |
| Two-component KLU and FG settings/JIT/gradient smoke checks | **Passed** |
| Selected mode, duplicate-COO, and Touchstone discrepancy reproductions | Recorded in [open questions](open-questions.md), not counted as conformance passes |
| Optional kfnetlist tests | **Skipped**: `kfnetlist` not installed |
| Example notebooks under root `nbs/`, documentation build, full lint/type checks | **Not run** for this documentation-only baseline |
| Other OSes, accelerators, supported dependency combinations | **Not run** |

The pytest run emitted a Hypothesis collection warning about the configured
`norecursedirs`; it did not fail tests. No runtime source or lockfile was changed.
An ignored `.venv` was created. The notebook kernel was installed under a temporary
prefix rather than replacing the user's global `sax` kernel.

## Reproduce targeted tests

Preferred environment setup respects the checked-in metadata and lockfile:

```sh
uv run --locked --python 3.12 --no-default-groups --group test \
  pytest src/tests --ignore=src/tests/test_nbs.py -q -rs
```

At baseline this failed before testing because the lockfile was stale. The actual
successful run used `--frozen` in place of `--locked` (use the existing lock without
resolving updates). Report that distinction; it does not prove the current declared
dependency set resolves. Once installed, `.venv/bin/python -m pytest ...` avoids
implicit environment synchronization.

For a focused change, substitute the relevant file/node IDs from the specs. The
pytest configuration in [`pyproject.toml`](../pyproject.toml) collects `src/tests`.

Notebook tests require Papermill and an IPython kernel named `sax`. The baseline
used the following isolated kernel setup after installing the test group:

```sh
uv pip install --python .venv/bin/python ipykernel
PREFIX=$(mktemp -d)
.venv/bin/python -m ipykernel install --prefix "$PREFIX" --name sax
JUPYTER_PATH="$PREFIX/share/jupyter" \
  .venv/bin/python -m pytest src/tests/test_nbs.py -q
# For the combined suite, use src/tests instead of src/tests/test_nbs.py.
```

`test_nbs.py` executes the four notebooks in `src/tests/nbs/`; it does not execute
all user examples. It recreates `src/tests/failed/` for failure output. `just test`
is the repository's broader standard entry point and installs a user kernel;
inspect [`justfile`](../justfile) before running setup commands. In particular,
`just dev` clears `.venv` and `just clean` removes generated/local files.

## Representative circuit acceptance check

Run in the installed environment. This checks a narrow numerical path, not blanket
JAX compatibility:

```python
import jax
import jax.numpy as jnp
import numpy as np
import sax


def component(wl=1.55, gain=1.0):
    return sax.reciprocal({("in0", "out0"): gain * jnp.exp(1j * wl)})


net = {
    "instances": {"a": {"component": "component", "settings": {"gain": 2.0}},
                  "b": "component"},
    "connections": {"a,out0": "b,in0"},
    "ports": {"in": "a,in0", "out": "b,out0"},
}
wl = jnp.array([1.5, 1.6])
for backend in ("klu", "fg"):
    model, _ = sax.circuit(net, {"component": component}, backend=backend)
    np.testing.assert_allclose(
        model(wl=wl, gain=2.0, b={"gain": 3.0})["in", "out"],
        6 * jnp.exp(2j * wl), rtol=1e-5,
    )
    np.testing.assert_allclose(
        jax.jit(model)(wl=wl)["in", "out"],
        2 * jnp.exp(2j * wl), rtol=1e-5,
    )
    derivative = jax.grad(
        lambda g: jnp.abs(model(a={"gain": g})["in", "out"]) ** 2
    )(2.0)
    np.testing.assert_allclose(derivative, 4.0, rtol=1e-5)
```

## Selecting evidence for future changes

- Representation changes: asymmetric coefficients, all three formats, sparse zeros,
  batch shapes, and any newly supported duplicate/subset semantics.
- Circuit changes: settings precedence, hierarchy, missing ports/models, one-port
  circuits, probes, and intended backend support.
- Numerical changes: analytic limits, explicit units/signs, selected backend
  comparisons, JIT and real-objective gradients, with justified tolerances.
- Parser changes: external known-good asymmetric fixtures, not just self-round-trips.
- Fitting/interpolation: grid shape/boundaries, degenerate data, and numeric/export
  equivalence as appropriate; import success is insufficient.
- Spec-only changes: inspect source/test evidence, check local links and skill
  references, and `git diff --check`. No numerical run is required merely for wording.

Do not silently expand scope to fix every baseline discrepancy. Record failures,
separate pre-existing failures from regressions, and request input when intent or a
required verification environment is missing.
