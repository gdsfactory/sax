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

## Final TODO remediation verification

After the per-item fixes, the full command
`JUPYTER_PATH="$PREFIX/share/jupyter" .venv/bin/python -m pytest src/tests -q -rs`
passed **388 tests, no skips, in 29.50 seconds**. This includes all four notebook
tests and all 33 kfnetlist tests (kfnetlist 0.3.0, already installed in the development
environment). An isolated temporary `sax` kernel was installed using the method below.
The final `just smoke` run passed 4 tests in **4.07 seconds wall-clock** (3.03s pytest).
`uv lock --check` also passed. The [completion audit](changes/todo-remediation.md)
maps each TODO to commits and actual regression surfaces.

Environment: Python 3.12.14 on macOS arm64; JAX 0.9.2, NumPy 2.5.3,
Pydantic 2.13.5, scikit-rf 1.13.0, pytest 9.1.1. The 34 warnings were Hypothesis
collection configuration and NumPy/xarray/SAX dtype deprecations, not skipped tests.
No full lint/type/pre-commit or public-example notebook run is claimed. Commits after
the user's instruction used `-n`.

## Netlist identity investigation

The native kfnetlist path adds `src/sax/native.py`, `_circuit_native`, and
`src/tests/test_native_kfnetlist.py`. Circuit construction now runs entirely on
native kfnetlist objects; legacy dictionaries and `.pic.yml` are adapted into
native first, and `get_required_circuit_models` uses the native resolver.
Non-notebook suite: **406 passed, 1 xfailed** (native tests plus all legacy
suites). `just smoke`: **6 passed**, ~4.4s wall-clock, existing environment.
Native input supports factory model substitution, distinct-cell fallback,
cell-specific overrides, missing-model diagnostics, JSON, legacy adaptation,
arrays, flat/hierarchical probes, internal-port policy, settings, JIT, and
gradients. The one xfail records that native input without direction cannot feed
the `forward` backend; legacy direction is preserved and legacy forward tests
pass. Full `src/tests` including notebooks, `uv lock --check`, and
cross-platform installs were not rerun for this slice. See the
[investigation](changes/kfnetlist-canonical.md).

## Fast checks during TODO remediation

Run `just smoke` (or `.venv/bin/python -m pytest src/tests/test_smoke.py -q`) in the
installed environment. Four tests cover asymmetric representation conversion,
KLU/FG settings/JIT/gradients, and zero-length optical transmission. Initial measured
wall-clock time: **7.13 seconds** including startup (pytest: 5.58 seconds). The
under-10-second target is local, not a guarantee on every machine or cold install.
No kernel setup or dependency synchronization occurs in this command.

## Reproduce targeted tests

Preferred environment setup respects the checked-in metadata and lockfile:

```sh
uv run --locked --python 3.12 --no-default-groups --group test \
  pytest src/tests --ignore=src/tests/test_nbs.py -q -rs
```

At baseline this failed before testing because the lockfile was stale. The actual
successful run used `--frozen` in place of `--locked` (use the existing lock without
resolving updates). Report that distinction; it does not prove the current declared
dependency set resolves. The subsequently tracked lockfile (`9e1e73b`) now passes
`uv lock --check`; remediation also ran
`uv run --locked --inexact --no-default-groups --group test pytest src/tests/test_smoke.py -q`
(**4 passed**). `--inexact` retained already installed optional fixture/kernel tools.
No lock changes were necessary in remediation. Once installed,
`.venv/bin/python -m pytest ...` avoids implicit environment synchronization.

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
