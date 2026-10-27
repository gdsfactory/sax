# RF admittance reference impedance

Implemented locally, pending upstream review.

## Intent and scope

`admittance` documented physical siemens but omitted the reference impedance from its conversion to S-parameters. Correct the series element while preserving its topology and port names. Shunt models and other RF models are outside this change.

## Contract and compatibility

Add keyword-only `z0=50`, matching `impedance`. Use `S11=1/(1+2*z0*y)` and `S21=2*z0*y/(1+2*z0*y)` directly so zero admittance remains an open circuit with a finite derivative. See [RF contracts](../models.md#rf-models).

This intentionally changes results for callers using the default normalization. `z0=0.5` reproduces the old formula. The release policy for that correction remains an upstream decision.

## Verification

The corrected physical-value assertion and new reference-impedance tests fail before the implementation change. After the fix, all 216 RF tests pass using Python 3.12 with both JAX 0.8.3 from an existing QPDK environment and JAX 0.11.2 from the current locked SAX environment. Changed Python files pass the repository's pre-commit hooks, including Pyright. Tests cover equivalence with reciprocal impedance, scalar and broadcast frequency shapes, reactive power conservation, and JIT and differentiation at zero admittance. This is not a full-suite or all-dependency-version claim.
