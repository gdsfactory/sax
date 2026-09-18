# Backend and numerical execution contracts

## Selection and interface

Backend names are case-insensitive. `fg` aliases `filipsson_gunnar`; `default`
resolves to the registered default, normally `klu`. Unknown names raise `ValueError`.
The registry maps each backend to three functions:

1. `analyze_instances(instances, models)` discovers default component S-matrices.
2. `analyze_circuit(analyzed_instances, nets, ports)` prepares static wiring.
3. `evaluate_circuit(analyzed, instances)` combines evaluated component values.

These are also available as lower-level APIs. Circuit construction wraps their
output into the requested representation. The apparent FG fallback when `klujax`
is missing is not reliable: the KLU module imports `klujax` before the fallback
handler. It is a declared dependency, not a verified optional extra.

Evidence: [`backends/__init__.py`](../src/sax/backends/__init__.py),
[`saxtypes/anymode.py`](../src/sax/saxtypes/anymode.py) (`val_backend`).

## Capabilities

| Backend | Computation | Native output | Important restrictions |
| --- | --- | --- | --- |
| `klu` | Sparse linear solve including feedback/reflections | `SDense` | Static sparse topology; KLU dependency; supports multiple links per endpoint |
| `filipsson_gunnar` | Repeated pairwise multiport elimination | `SDict` | Each internal endpoint occurs only once |
| `forward` | Directed, name-based graph propagation | `SDict` | `in*` to `out*` names, oriented connections, no general reflection/feedback solution |
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

Batch values are broadcast to a shape chosen from the instance values, flattened
for solving, then reshaped back. The implementation chooses the highest-rank
instance batch shape, not a general joint broadcast shape. Equal-rank complementary
shapes such as `(N, 1)` and `(1, M)` are not a guaranteed supported circuit sweep.
Do not extrapolate S-dictionary broadcasting into universal backend broadcasting.

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
endpoints for FG, forward, and additive.

### Forward-only

Only component terms with input names starting `in` and output names starting
`out` become directed edges. Only external `in*` ports are excited and external
`out*` ports collected. Values are flattened. Connections have directional meaning
here, unlike KLU/FG's bidirectional interconnects.

Propagation uses BFS layers, not a general topological accumulation algorithm.
Treat feed-forward topology as a prerequisite, not proof of correctness for every
DAG: unequal-depth reconvergent paths need explicit checks against KLU or FG.
Optical `o*` naming does not automatically work with this backend.

Evidence: [`forward_only.py`](../src/sax/backends/forward_only.py),
`_graph_edges_directed`, `evaluate_circuit_forward`.

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
the topology; compare forward only within its restricted semantics. Test scalar and
batch shapes, JIT, and a real-valued differentiated objective when changing numerical
paths. Specify tolerances and dtype; do not require bitwise equality across solvers.
Passivity, reciprocity, and power conservation are model-dependent, not universal
assertions. Solver failure on a singular network is not evidence of a valid result.

Existing evidence: [`03_backends.ipynb`](../src/tests/nbs/03_backends.ipynb) compares
KLU and FG within `1e-5`; [`test_probes.py`](../src/tests/test_probes.py) exercises
circuit/probe behavior. Baseline smoke checks cover a simple KLU/FG circuit's JIT and
gradient, not all representations, devices, nested transformations, or backends.
