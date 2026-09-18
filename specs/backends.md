# Backend and numerical execution contracts

## Selection and interface

Backend names are case-insensitive. `fg` aliases `filipsson_gunnar`; `default`
resolves to the registered default, normally `klu`. Unknown names raise `ValueError`.
The registry maps each backend to three functions:

1. `analyze_instances(instances, models)` discovers default component S-matrices.
2. `analyze_circuit(analyzed_instances, nets, ports)` prepares static wiring.
3. `evaluate_circuit(analyzed, instances)` combines evaluated component values.

These are also available as lower-level APIs. Circuit construction wraps their
output into the requested representation. `klujax` is mandatory, consistent with
package metadata; a missing installation fails import rather than silently selecting
FG. The old unreachable fallback was removed. `test_backend_dependency.py` verifies
normal default selection and an isolated missing-dependency import.

Evidence: [`backends/__init__.py`](../src/sax/backends/__init__.py),
[`saxtypes/anymode.py`](../src/sax/saxtypes/anymode.py) (`val_backend`).

## Capabilities

| Backend | Computation | Native output | Important restrictions |
| --- | --- | --- | --- |
| `klu` | Sparse linear solve including feedback/reflections | `SDense` | Static sparse topology; KLU dependency; supports multiple links per endpoint |
| `filipsson_gunnar` | Repeated pairwise multiport elimination | `SDict` | Each internal endpoint occurs only once |
| `additive` | Enumerate simple paths and add edge values | Dictionary of path-value lists | Not an ordinary complex-amplitude scattering solver |

### KLU

Let `S` be the block-diagonal component matrix, `C` the internal connection matrix,
and `E` the external excitation map. Evaluation implements

```text
S_external = E.T @ S @ solve(I - C @ S, E)
```

Connections add both directions. Repeated identical endpoint pairs are deduplicated;
multiple distinct neighbors remain separate links. Symbolic sparse analysis is
reused; numerical values are assembled and solved at evaluation time. Topology
indices are made concrete NumPy arrays during discovery where possible, while
S-values remain JAX arrays.

Batch values broadcast to the joint NumPy/JAX-compatible batch shape, are flattened
for solving, then reshaped back. Complementary `(N, 1)` and `(1, M)` sweeps yield
`(N, M)`; incompatible shapes raise `ValueError`. This does not extend the same
shape contract to the path-based additive backend.
Regression: `src/tests/test_backend_broadcasting.py` covers scalar/array mixtures,
complementary/multiaxis shapes, KLU/FG agreement, JIT, and gradients.

Evidence: [`klu.py`](../src/sax/backends/klu.py), especially
`_scoo_with_numpy_indices`, `analyze_circuit_klu`, `evaluate_circuit_klu`.

### Filipsson-Gunnar

The evaluator namespaces component dictionary keys, sorts connections, applies the
interconnection formula, removes consumed internal terms, and selects external
ports. Omitted terms are treated as zero. Pairwise elimination includes reflections
and repeated propagation through feedback; it is not forward path enumeration.
Zero denominators/singular networks have no general regularization guarantee.

Evidence: [`filipsson_gunnar.py`](../src/sax/backends/filipsson_gunnar.py),
`_calculate_interconnected_value`. `_nets_to_connections_strict` rejects repeated
endpoints for FG and additive.

### Removed forward backend

`forward` and its lower-level exports have been removed. Its reliance on endpoint
ordering is incompatible with undirected native netlists. Use KLU or FG for
scattering simulation; probe forward/backward wave measurements are unchanged.
`test_backend_selection.py` verifies rejection and retains unequal-depth KLU/FG
reconvergence, JIT, and gradient coverage.

### Additive

Edges carry quantities such as path length, not necessarily scattering amplitudes.
Connection edges contribute zero. Allowed simple paths exclude consecutive component
edges. `_path_lengths` adds values along each path, forms combinations of array
values, and returns a list per port pair. Paths are **not** coherently summed into a
single S-parameter. Do not compare this output directly with a physical solver.

Evidence: [`additive.py`](../src/sax/backends/additive.py), `_path_lengths`;
example [`06_additive_backend.md`](../docs/nbs/examples/06_additive_backend.md).

## Numerical verification boundary

Use asymmetric fixtures to verify direction. Compare KLU/FG only where both support
the topology. Test scalar and
batch shapes, JIT, and a real-valued differentiated objective when changing numerical
paths. Specify tolerances and dtype; do not require bitwise equality across solvers.
Passivity, reciprocity, and power conservation are model-dependent, not universal
assertions. Solver failure on a singular network is not evidence of a valid result.

`test_backend_restrictions.py` verifies repeated-endpoint rejection for FG/additive, KLU multi-link coefficients, and additive length sums rather than products.

Existing evidence: [`03_backends.ipynb`](../src/tests/nbs/03_backends.ipynb) compares
KLU and FG within `1e-5`; [`test_probes.py`](../src/tests/test_probes.py) exercises
circuit/probe behavior. Baseline smoke checks cover a simple KLU/FG circuit's JIT and
gradient, not all representations, devices, nested transformations, or backends.
