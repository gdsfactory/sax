# Open questions and observed discrepancies

These are **not approved behavior changes**. Confirm intended behavior and add a
focused regression test before fixing one. Source inspection alone is labeled
separately from runtime reproduction. Do not infer an exhaustive bug audit.

## Representation and model behavior

| Observation | Evidence | Next check |
| --- | --- | --- |
| Dense `multimode(..., modes=("X",))` produces TE/TM while dictionary input produces X | **Reproduced**; [`multimode.py`](../src/sax/multimode.py), dense dispatch omits `modes` | Agree that custom modes apply uniformly; test all three formats |
| `get_modes` repeats mode names per port despite documenting uniqueness | **Reproduced**; [`s.py`](../src/sax/s.py), `get_modes` | Decide unique ordering and test multiple ports |
| Duplicate COO coordinates sum in dense conversion but overwrite in dictionary conversion | **Reproduced**; `s.py`, `_scoo_to_sdense` / `_scoo_to_sdict` | Decide whether duplicates are invalid or require consistent reduction |
| `phase_shifter.loss` is multiplied by length, unlike its lumped-loss description | **Inspected**; [`models/straight.py`](../src/sax/models/straight.py) | Establish units and backward-compatibility policy before changing formula |
| Constant feature/target columns divide by zero during neural fitting normalization | **Inspected**; [`fit.py`](../src/sax/fit.py), `neural_fit` | Decide rejection vs constant-column handling; test nonfinite outcomes |

Reproduction seeds for the first two observations:

```python
s = {("in0", "out0"): 1.0}
sax.get_ports(sax.multimode(sax.sdense(s), modes=("X",)))  # currently TE/TM
sax.get_modes(sax.multimode(s))  # currently ("TE", "TM", "TE", "TM")
```

## Construction and topology

| Observation | Evidence | Next check |
| --- | --- | --- |
| Advertised no-klujax fallback is preceded by an unconditional KLU module import | **Inspected**; [`backends/__init__.py`](../src/sax/backends/__init__.py), [`klu.py`](../src/sax/backends/klu.py) | Decide whether KLU is genuinely optional; test isolated imports if so |
| DAG validation uses `is_directed()`, not an acyclicity check | **Inspected**; [`circuits.py`](../src/sax/circuits.py), `_validate_dag` | Cycles may instead fail at root validation/topological sort; define stable diagnostics |
| Flattening handles `connections` but not equivalent recursive `nets`; instance renaming also leaves `nets` references untouched | **Inspected**; [`netlists.py`](../src/sax/netlists.py), `_flatten_netlist_into`, `rename_instances` | Test semantic equivalence for supported fields before expanding guarantees |
| Unconnected probes actually insert the four-port model and expose both taps, contrary to the alias-only docstring | **Inspected and covered by passing tests**; `expand_probes`, `test_probe_on_truly_unconnected_port` in [`test_probes.py`](../src/tests/test_probes.py) | Likely documentation correction; preserve tested behavior unless deliberately changed |
| Recursive YAML discovery uses `folder_path.rglob(ext)`, with default `.pic.yml`, rather than a suffix wildcard | **Inspected**; [`utils.py`](../src/sax/utils.py), `load_recursive_netlist` | Add a multi-file fixture and decide supported naming/discovery rules |

KLU batch-shape selection and forward BFS limitations are documented in
[backends](backends.md); neither has comprehensive regression coverage here.

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
