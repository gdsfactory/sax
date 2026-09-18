# Architecture

## Purpose and boundaries

SAX is a frequency-domain scattering-parameter circuit simulator built around JAX.
A component is a Python callable returning an S-matrix; a composed circuit is
another callable with the same role. Dictionaries, tuples, and arrays are the
runtime interchange structures, with annotated types and Pydantic validators at
selected boundaries rather than a mandatory component class hierarchy.

SAX provides differentiation-friendly numerical models and circuit evaluation,
not a time-domain solver, electromagnetic field solver, layout engine, or universal
physical-validity checker. Nonreciprocal and deliberately nonphysical models are
supported. Optimization is performed with JAX-compatible objectives and tools;
`sax.fit` additionally supplies neural regression.

## Subsystems

| Source | Responsibility |
| --- | --- |
| [`saxtypes/`](../src/sax/saxtypes/) | Annotated types, coercion, model/netlist validation |
| [`s.py`](../src/sax/s.py), [`multimode.py`](../src/sax/multimode.py), [`ports.py`](../src/sax/ports.py) | Representations, modes, port names |
| [`netlists.py`](../src/sax/netlists.py), [`circuits.py`](../src/sax/circuits.py) | Topology preparation, hierarchy, model composition |
| [`backends/`](../src/sax/backends/) | Analysis/evaluation implementations |
| [`models/`](../src/sax/models/) | Optical primitives, factories, probes, RF models |
| [`parsers/`](../src/sax/parsers/) | External schematic/netlist and S-parameter adapters |
| [`interpolation.py`](../src/sax/interpolation.py), [`fit.py`](../src/sax/fit.py), [`loss.py`](../src/sax/loss.py) | Tabulated models, learned models, numerical losses |
| [`utils.py`](../src/sax/utils.py), [`constants.py`](../src/sax/constants.py) | Settings, I/O, naming, normalization, wavelength helpers |
| [`__init__.py`](../src/sax/__init__.py) | Public re-exports; module APIs also exist |

## Execution lifecycle

1. Supply component functions and a flat or recursive netlist.
2. `circuit` normalizes topology, handles internal ports/probes and arrays, prunes
   unused instances, builds a component dependency graph, and resolves models.
3. Build subcircuits from leaves upward. Each backend discovers component topology
   by evaluating models with defaults, then analyzes connections and external ports.
4. Calling the returned model merges settings, evaluates instances, and invokes
   the backend evaluator. Requested representation conversion wraps the result.
5. Reuse that callable for parameter sweeps or differentiation without rebuilding
   static topology on every call.

The dependency DAG describes **component definitions**, not the absence of optical
feedback. A valid hierarchical definition can contain feedback in its port wiring.
KLU and Filipsson-Gunnar solve such wiring; forward-only does not generally do so.

## Compatibility boundary

- [`pyproject.toml`](../pyproject.toml) declares Python >=3.11 and JAX
  >=0.6.0,<0.10.0 at the baseline revision. The lockfile is a separate reproducibility
  surface and currently needs reconciliation; see [verification](verification.md).
- JIT/autodiff claims apply to particular numerical paths, not every public API.
  File I/O, pandas/xarray preparation, NetworkX analysis, and symbolic code generation
  are host-side operations. Prefer building topology outside JAX transformations.
- Keep port sets, sparse coordinate order, representation structure, and hierarchy
  stable during numerical evaluation. Rebuild when topology changes.
- Dtype/device behavior depends on JAX configuration and dependencies. Do not assume
  importing SAX leaves global precision configuration unchanged; the measured
  environment is recorded separately from the intended numerical contracts.

Evidence: `circuit`, `_flat_circuit`, `circuit_backends`, and backend analysis/evaluation
functions. Tests: [`test_imports.py`](../src/tests/test_imports.py),
[`test_circuit.py`](../src/tests/test_circuit.py), and
[`03_backends.ipynb`](../src/tests/nbs/03_backends.ipynb).
