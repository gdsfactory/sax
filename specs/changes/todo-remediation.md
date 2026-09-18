# Codebase TODO remediation

Status: in progress. Scope: all items in root `todo.md`; one commit per item unless
closely related fixes share implementation and tests. No unrelated refactors.

## Contract and decisions

- Preserve dictionary (input, output) versus matrix (output, input) direction.
- Fix netlist transforms and numerical paths with explicit regression fixtures.
- Keep phase-shifter numerical behavior for compatibility; document its actual
  per-micrometer loss unit rather than silently changing existing simulations.
- KLU remains a mandatory dependency, consistent with package metadata. Remove the
  misleading optional fallback instead of introducing a new optional installation mode.
- Conflicting reciprocal dictionary pairs will use the first supplied direction
  for both entries. This avoids data-dependent validation inside JAX tracing.
- Constant neural-fit columns will use unit normalization scale, so constant features
  normalize to zero and constant targets remain well-defined.
- Touchstone input direction and remaining input-path issues are one coherent change.

## Verification

`just smoke` runs only `src/tests/test_smoke.py` in the installed environment. Target:
under 10 seconds wall-clock locally, without notebook kernels or dependency resolution.
Each fix also gets focused tests as applicable. Run the complete pytest suite, including
notebooks and optional kfnetlist fixtures, after all changes. Update current specs and
track resolution evidence in `todo.md`; historical baseline results remain historical.

## Progress

- Smoke suite: asymmetric representations, two-backend settings/JIT/gradient checks,
  and optical zero-length behavior. Timing recorded in `specs/verification.md`.
