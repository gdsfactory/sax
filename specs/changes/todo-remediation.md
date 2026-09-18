# Codebase TODO remediation

Status: implemented and verified. Scope: all items in root `todo.md`; one commit
per item except the two related Touchstone items, which share implementation/tests.
The smaller-API parent item groups its four subitems. No unrelated refactors.

## Contract and decisions

- Preserve dictionary (input, output) versus matrix (output, input) direction.
- Fix netlist transforms and numerical paths with explicit regression fixtures.
- Keep phase-shifter numerical behavior for compatibility; document its actual
  per-micrometer loss unit rather than silently changing existing simulations.
- KLU remains mandatory, consistent with package metadata. Remove the misleading
  optional fallback instead of introducing a new optional installation mode.
- Conflicting reciprocal dictionary pairs use the first supplied direction for
  both entries. This avoids data-dependent validation inside JAX tracing.
- Constant neural-fit columns use unit normalization scale for training and export.
- Preserve the legacy flattening separator; use `sep="__"` for executable flattened
  circuits because `~` cannot be used in their synthetic Python signatures.
- The lockfile was already reconciled in `9e1e73b`; verification and installation
  guidance were updated without an unnecessary dependency resolution change.

## Completion audit

Deliverables: every `todo.md` item addressed, focused pytest regressions where
applicable, current specs updated, per-item commits, a measured <10-second smoke
suite, and a final full run including notebooks and optional kfnetlist tests.

| TODO / requirement | Commit | Concrete verification surface |
| --- | --- | --- |
| Smoke suite <10 seconds | `8893748` | `test_smoke.py`: 4 tests; final `just smoke` 4.07s wall-clock |
| Touchstone direction | `e1bddff` | `test_touchstone.py`: asymmetric external reader fixture and independent scikit-rf writer check |
| Touchstone defaults/raw v1/frequency subitems | `e1bddff` | Same suite: default/invalid labels, raw v1 multiline/v2, frequency and wavelength |
| Netlist `nets` transforms | `17d9718` | `test_netlist_transforms.py`: KLU/FG equivalence, renaming, metadata, same-child endpoints, no mutation |
| Forward reconvergence | `f0eb5b4` | `test_forward_backend.py`: unequal-depth paths vs KLU/FG, JIT/gradient, cycle rejection |
| Joint batch broadcasting | `149499f` | `test_backend_broadcasting.py`: scalar, complementary/multiaxis, invalid shapes, JIT/gradient |
| Dense custom modes | `005b0fe` | `test_custom_modes.py`: 3 formats, wrappers, custom modes and extraction |
| COO duplicates | `955a508` | `test_sparse_duplicates.py`: sum/cancellation, batch, round-trip and differentiation |
| Constant fit columns | `04879ca` | `test_fit_degenerate.py`: all constant/nonconstant combinations, finite values, export agreement, index preservation |
| Safe Lumerical writes | `649d4dd` | `test_lumerical_writer.py`: repeated overwrite, in-memory output, no mutation, parsed amplitudes |
| Recursive YAML discovery | `aedb53a` | `test_recursive_yaml.py`: suffixes, nested discovery/order, duplicates |
| Phase-shifter loss units | `461f51a` | `test_phase_shifter.py`: length scaling, phase and gradient; corrected API docstring |
| Hierarchy cycles | `016d894` | `test_hierarchy_validation.py`: self/multilevel cycles, valid optical feedback |
| KLU dependency policy | `68adbd3` | `test_backend_dependency.py`: default selection and subprocess blocked-import check; README policy |
| Mode uniqueness, reciprocal conflicts, return type, probe docs | `9f72350` | `test_api_contracts.py` and existing `test_probes.py` rerun; corresponding docstrings/specs |
| Dependency/README alignment | `112d3a4` | `uv lock --check`; locked smoke run; Python/group installation instructions |
| Remaining verification gaps / final suite | Final coverage commit | `test_backend_restrictions.py`, `test_interpolation_contract.py`, `test_native_type_coercion.py`, notebook correction, all 33 kfnetlist tests |
| Commit isolation | Git log from `9e1e73b` | 16 commits, with related Touchstone items combined; later commits use requested `-n` |
| Full suite after implementation | Final coverage commit | **388 passed, no skips**, including 4 notebook tests; 29.50s |

The full run first exposed a stale notebook expectation: Pydantic 2.13 accepts
zero-dimensional arrays as native `complex`. Native conversions delegate to Pydantic,
so the notebook and new pytest cases now check that delegation, accepted values,
and failure translation rather than freezing an older dependency's rejection.
A global filesystem mock also exposed test-order sensitivity during lazy imports;
it was narrowed to the Lumerical writer's own filesystem boundary. Both were
corrected and the whole suite rerun, not merely skipped.

Final commands/environment and warnings are in [verification](../verification.md).
The run proves the checked contract surfaces, not universal physical validity,
all dependency/platform combinations, or every user example notebook. Those limits
are not unfinished TODO items. No pre-commit compliance claim is made.
