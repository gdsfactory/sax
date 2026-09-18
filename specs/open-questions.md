# Open questions and observed discrepancies

These are **not approved behavior changes**. Confirm intended behavior and add a
focused regression test before fixing one. Source inspection alone is labeled
separately from runtime reproduction. Do not infer an exhaustive bug audit.

## Representation and model behavior

| Observation | Evidence | Next check |
| --- | --- | --- |
| **Resolved:** custom modes now apply to dense as well as dictionary/COO values and wrappers | `test_custom_modes.py` | All formats tested for custom-mode replication and extraction |
| **Resolved:** `get_modes` returns unique naturally sorted modes | `test_api_contracts.py` | Multiple ports and all representations tested |
| **Resolved:** duplicate COO coordinates sum in both dictionary and dense conversion | `test_sparse_duplicates.py` | Batch, cancellation, round-trip, JIT, and gradient checks |
| **Resolved:** `phase_shifter.loss` documented as dB/µm, retaining its historical formula | `test_phase_shifter.py` | Length scaling, zero-length limit, voltage phase, and gradient checked |
| **Resolved:** constant feature/target columns use unit normalization scale | `test_fit_degenerate.py` | Finite fits, prediction index preservation, and symbolic export equivalence tested |

Reproduction seeds for the first two observations:

```python
s = {("in0", "out0"): 1.0}
sax.get_ports(sax.multimode(sax.sdense(s), modes=("X",)))  # now X (previously TE/TM)
sax.get_modes(sax.multimode(s))  # now ("TE", "TM")
```

## Construction and topology

| Observation | Evidence | Next check |
| --- | --- | --- |
| **Resolved:** KLU remains mandatory and the unreachable fallback was removed | `test_backend_dependency.py` | Isolated missing-dependency import and normal defaults tested |
| **Resolved:** explicit DAG acyclicity validation produces a dependency-cycle diagnostic | `test_hierarchy_validation.py` | Self/multi-component cycles rejected; optical feedback still works |
| **Resolved:** flattening and renaming now rewrite `nets` and preserve net metadata | `test_netlist_transforms.py`: KLU/FG equivalence, metadata, repeated endpoints, input isolation | Legacy `~` flattened names still require a valid identifier separator such as `__` for circuit construction |
| **Resolved:** probe documentation now describes both taps at unconnected/boundary targets | Existing `test_probes.py` regressions rerun | Runtime behavior preserved |
| **Resolved:** recursive YAML loading uses a suffix wildcard and deterministic ordering, preserving the root first | `test_recursive_yaml.py` | Default/custom suffixes, nested files, ignored nonmatches, and duplicate-name rejection tested |

KLU now computes a joint broadcast shape, verified by `test_backend_broadcasting.py`;
see [backends](backends.md).
Forward BFS was replaced with topological accumulation; unequal-depth reconvergence,
JIT/gradients, and cycle rejection are covered by `test_forward_backend.py`.

## Touchstone interoperability — resolved

Resolved by the Touchstone remediation commit: correct matrix direction, default
labels, frequency coordinates, and in-memory raw v1/v2 parsing. Regression evidence:
`src/tests/test_touchstone.py` (independent asymmetric writer/reader checks).
The observations below describe the original baseline, not current behavior.

In [`parsers/touchstone.py`](../src/sax/parsers/touchstone.py):

- **Reproduced:** omitted `ports` raises `ValueError` for a normal two-port file;
  no default `o1/o2` labels are generated.
- **Reproduced:** nonreciprocal direction is reversed relative to SAX's dictionary
  convention. A two-port fixture with `S21=0.5`, `S12=0.25` emits `(a,b)=0.25`,
  `(b,a)=0.5`. scikit-rf rows are outputs, but reader xarray axes are labeled
  `port_in`, then `port_out`. Writer axis handling also needs an asymmetric check;
  a mutually transposed writer/reader round trip alone would hide the error.
- **Reproduced:** raw Touchstone v1 text is put in a `.dat` temporary file, which
  the installed scikit-rf rejects without a port-count-bearing extension.
- **Inspected:** `convert_to_wavelength=False` does not add an `f` coordinate to
  the 3D data. Verify this path before promising frequency-preserving reads.

Use a real `.s2p` fixture to isolate direction/default-port tests from raw-string
handling:

```text
# Hz S RI R 50
1000000000 0 0 0.5 0 0.25 0 0 0
```

## Tooling

`uv run --locked --python 3.12 --no-default-groups --group test ...` refused the
baseline because `uv.lock` needs updating. Verification used `--frozen` instead.
**Do not silently regenerate the lockfile during a documentation/spec update.**
Reconcile it as an explicit dependency change and rerun the relevant checks.
The README's Python/dev-extra installation guidance also differs from current
`pyproject.toml` (Python >=3.11, development dependency groups).
