# Codebase TODOs

Findings from the baseline review. **Reproduced** means observed at runtime;
**inspected** means identified from source and still needs a focused regression
check. Establish intended behavior before changing compatibility-sensitive APIs.
See [specs/open-questions.md](specs/open-questions.md) for supporting details.

## Start with this one:

- [x] Create a smaller smoke-test test-suite that completes in under 10 seconds.
  `just smoke`: 4 tests covering representations, settings, JIT/gradients, and an
  optical limit; measured 7.13 seconds wall-clock including Python/pytest startup.

## Highest priority

- [x] **Correct Touchstone input/output direction mapping.** Reproduced: importing
  an asymmetric fixture reverses `S21` and `S12`. Check the writer independently
  against an external fixture; a self-round-trip could conceal matching errors.
  Source: `src/sax/parsers/touchstone.py`.
- [x] **Make netlist transformations handle `nets` consistently.** Inspected:
  hierarchical flattening handles `connections` but not equivalent `nets`, and
  instance renaming leaves references inside `nets` unchanged. Add equivalence and
  renamed-reference tests. Source: `src/sax/netlists.py`.
  Regression: `test_netlist_transforms.py`; focused suite plus smoke: **12 passed**.
  Circuit equivalence uses identifier-safe `sep="__"`; legacy `~` names are export-only.
- [x] **Validate and correct forward-only propagation on reconvergent paths.**
  Inspected: BFS-layer propagation may miss contributions arriving along longer
  paths after a node has propagated. Compare unequal-depth feed-forward fixtures
  against KLU/FG; implement correct accumulation or explicitly reject unsupported
  topologies. Source: `src/sax/backends/forward_only.py`.
  Resolved with topological accumulation and explicit cycle rejection;
  `test_forward_backend.py` plus smoke: **6 passed**.
- [x] **Define and implement circuit-wide batch broadcasting.** Inspected: KLU
  chooses a highest-rank instance shape rather than a joint broadcast shape.
  Test `(N, 1)` with `(1, M)`, scalar/array mixtures, and incompatible shapes.
  Source: `src/sax/backends/klu.py`.
  Joint broadcast shape implemented; `test_backend_broadcasting.py` plus smoke:
  **13 passed**, including JIT and gradient checks.

Touchstone resolution: `src/tests/test_touchstone.py` plus smoke: **13 passed**.
Reader/writer directions are tested independently; related input fixes share a commit.

## Other correctness and API issues

- [x] **Honor custom modes in dense multimode conversion.** Reproduced:
  `multimode(sdense, modes=("X",))` produces TE/TM instead of X. Verify custom
  modes across all three representations. Source: `src/sax/multimode.py`.
  Fixed dispatch; `test_custom_modes.py` plus smoke: **16 passed**.
- [ ] **Resolve duplicate-COO conversion semantics.** Reproduced: conversion to
  dense sums duplicate coordinates, while conversion to dictionary keeps the last
  value. Decide whether to reject duplicates or reduce them consistently; add
  numerical conversion tests. Source: `src/sax/s.py`.
- [x] **Repair remaining Touchstone input paths.** Source:
  `src/sax/parsers/touchstone.py`.
  - [x] Implement documented default port labels or explicitly require labels.
    Reproduced: omitted labels currently raise an error.
  - [x] Support raw Touchstone v1 text without losing its port-count information.
    Reproduced: the temporary `.dat` extension causes scikit-rf to reject it.
  - [x] Test and repair `convert_to_wavelength=False`. Inspected: the frequency
    coordinate is omitted during xarray construction.
- [ ] **Handle zero-variance neural-fit columns.** Inspected: feature/target
  normalization divides by zero for constant columns. Decide rejection versus
  supported constant-column handling and test finite results or clear errors.
  Source: `src/sax/fit.py`.
- [ ] **Define safe Lumerical writer file semantics.** Inspected: the writer opens
  in append mode, including deterministic temporary paths. Test repeated writes,
  choose/document overwrite versus append behavior, and clean up temporary output.
  Source: `src/sax/parsers/lumerical.py`.
- [ ] **Fix or clarify recursive YAML discovery.** Inspected: the default search
  is `rglob(".pic.yml")`, not `rglob("*.pic.yml")`. Add a multi-file fixture and
  verify intended naming/discovery rules. Source: `src/sax/utils.py`.
- [ ] **Resolve phase-shifter loss units.** Inspected: attenuation uses
  `loss * length`, but documentation suggests lumped loss. Establish intended
  units and compatibility policy, then align formula, docs, and tests.
  Source: `src/sax/models/straight.py`.

## Validation and maintainability

- [ ] **Validate hierarchy acyclicity explicitly.** Inspected: `_validate_dag`
  checks `is_directed()` rather than acyclicity. Add cyclic hierarchy fixtures and
  stable diagnostics without rejecting legitimate optical feedback wiring.
  Source: `src/sax/circuits.py`.
- [ ] **Resolve the missing-KLU fallback.** Inspected: unconditional imports can
  fail before the fallback handler runs. Decide whether KLU is mandatory or truly
  optional; align imports, metadata, documentation, and isolated import tests.
  Sources: `src/sax/backends/__init__.py`, `src/sax/backends/klu.py`.
- [ ] **Align smaller API behaviors and documentation.**
  - [ ] Decide unique-mode ordering for `get_modes` and test multiple ports.
    Reproduced: it repeats modes despite documenting uniqueness.
    Source: `src/sax/s.py`.
  - [ ] Correct unconnected-probe documentation to match the tested two-tap
    behavior, unless an intentional behavior change is requested.
    Source: `src/sax/netlists.py`.
  - [ ] Decide whether unsupported circuit `return_type` values should raise.
    Inspected: they silently leave the output unwrapped.
    Source: `src/sax/circuits.py`.
  - [ ] Define `reciprocal` behavior for conflicting directional entries.
    Inspected: it swaps conflicting values instead of enforcing equality.
    Source: `src/sax/s.py`.
- [ ] **Reconcile dependency declarations and the lockfile.** Reproduced:
  `uv run --locked` refuses the current metadata/lock combination. Regenerate only
  as an explicit dependency change, inspect the resolution diff, and rerun checks.
  Align README installation guidance with Python requirements/dependency groups.
  Sources: `pyproject.toml`, `uv.lock`, `README.md`.
- [ ] **Close verification gaps.** Add focused tests for parser directionality,
  fitting degeneracies, batch broadcasting, and backend restrictions as the above
  issues are addressed. Run the optional kfnetlist suite in an environment with
  its fixture dependency installed. Do not treat import tests as numerical tests.

For each completed item, update affected specs/tests and record verification.
The baseline suite passed **266 tests**, with **1 skipped** for missing `kfnetlist`;
that result does not establish correctness of the untested cases above.
