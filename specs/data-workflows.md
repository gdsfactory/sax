# External data, interpolation, and fitting

## Netlist adapters

| API | Baseline behavior |
| --- | --- |
| `load_netlist` | Read text/path/file-like content and `yaml.safe_load` it; loading is not full circuit validation |
| `load_recursive_netlist` | Load the top file first, then suffix-matching files recursively in sorted order; reject duplicate normalized component names |
| `parse_kfnetlist` | Accept dictionary, JSON string, or object with `to_dict`; return a one-entry recursive netlist (legacy dictionary output; retained for compatibility) |
| `parse_kfnetlist_recursive` | Convert a named mapping of such netlists; preserve mapping order |
| `native.load_pic_yaml` | Load flat or `modules`/`toplevel` PIC YAML into native kfnetlist objects |
| `native.load_native_netlist` / `load_native_recursive_netlist` | Native-object PIC loaders |
| `native.from_legacy_flat` / `from_legacy_recursive` | Adapt legacy SAX dictionaries into native objects |
| `parse_mosaic` | Read dictionary/YAML/path; map components and properties into instances, nets, external ports |

`sax.circuit` builds every circuit through the native kfnetlist path:
`native.to_hierarchy` adapts legacy dictionaries, `.pic.yml`/native JSON, and
native objects into `{cell: Netlist}` plus an explicit root,
and `_circuit_native` lowers them to backend tables. Native input is never turned
into SAX's legacy dictionary schema for construction. The forward backend and direction hints have been removed. Factory
`component` models are resolved before descending into a distinct
`PlacedInstance.cell`.

kfnetlist array references translate 1-based `ia/ib` to zero-based
`instance<column.row>,port`. Collapsed references to array instances target `<0.0>`.
External members attach to the first instance member; multiple instance members
are connected in a chain, not expanded to all pairwise links. Unused layout fields
are not retained. The adapter itself does not import kfnetlist.
In particular it drops `kcl` and placed-instance `cell` references, so it does not
support general hierarchy fallback when factory names differ from cell-map keys.
The [canonical kfnetlist investigation](changes/kfnetlist-canonical.md) records a
live reproduction and a proposed migration retaining `.pic.yml` compatibility.

Mosaic resolves models through the supplied mapping, optionally an active
gdsfactory PDK, or a final-name fallback. One-member nets become external ports;
multi-member nets become all endpoint pairs. This differs from kfnetlist's chain
representation and can require KLU for multiply connected endpoints.

Evidence: [`parsers/kfnetlist.py`](../src/sax/parsers/kfnetlist.py),
[`parsers/mosaic.py`](../src/sax/parsers/mosaic.py),
[`utils.py`](../src/sax/utils.py). Test surface:
[`test_kfnetlist_parser.py`](../src/tests/test_kfnetlist_parser.py), skipped in the
original baseline environment; all 33 tests ran successfully during final remediation
with kfnetlist 0.3.0 installed.

## Tabular S-parameter interchange

The common tidy representation includes `port_in`, `port_out`, `mode_in`,
`mode_out`, amplitude `amp`, phase `phi` in radians, and frequency `f` (Hz) or
wavelength `wl` (micrometers). Writers also accept real/imaginary columns `re`/`im`.
Frequency/wavelength conversion uses `wl = C_UM_S / f`. Readers have different
conversion defaults; do not infer a single global policy.

- `parse_lumerical_dat` parses text, paths, or file-like objects using Lark, emits
  `ExperimentalWarning`, and retains frequency by default (`convert_f_to_wl=False`).
- `parse_touchstone` delegates to scikit-rf and defaults to wavelength conversion.
  Omitted labels become `o1`, `o2`, etc.; explicit labels must be unique and match
  matrix size. Matrix output/input axes are mapped to the correctly labeled table
  directions. Frequency-preserving reads provide `f`. Raw text is parsed in memory:
  v1 full-matrix records supply the inferred port count, and v2 declares its count.
- `write_touchstone(df, path=None)` returns text or writes a file and returns its
  resolved path. Missing extensions become `.sNp`; mismatched extensions warn.
- `write_lumerical_dat` returns text in memory or overwrites a path. Repeated writes
  do not append duplicate blocks, and string output creates no temporary file.
  `test_lumerical_writer.py` checks repeatability, input isolation, and parsed values.
- Both writers copy their input DataFrame. Neither is specified as a lossless preservation
  mechanism for all metadata, labels, reference impedances, or mode information.

Evidence: [`parsers/lumerical.py`](../src/sax/parsers/lumerical.py),
[`parsers/touchstone.py`](../src/sax/parsers/touchstone.py) (`_validate_columns`).
[`test_imports.py`](../src/tests/test_imports.py) checks imports, not parser numerical
round trips. `src/tests/test_touchstone.py` verifies asymmetric import/export
independently, raw v1/v2 text, default labels, validation, and frequency reads.

## Grid conversion and interpolation

- `to_df(SType, **coords)` requires one 1D coordinate array per numerical batch
  dimension, in axis order, with matching lengths. It broadcasts values and emits
  the full port/mode Cartesian product, filling missing terms with zero; absent
  mode suffixes become mode `"1"`.
- `to_df(DataArray)` stacks coordinates into rows with a target value column.
- `to_xarray(df, target_names=("amp", "phi"))` sorts non-target columns, reshapes
  into their Cartesian grid, and adds a final `targets` axis. The data must form
  a complete, unique rectangular grid; missing or duplicated combinations are not
  a supported scattered-data input.
- `interpolate_xarray` returns a dictionary keyed by final-axis target labels.
  Numeric coordinates are linearly interpolated; object/string axes are categorical
  selections. Target-axis interpolation is rejected. Unspecified numerical inputs
  default to coordinate means; unspecified categorical axes retain all categories.
- Numeric queries broadcast together. Coordinate mapping uses `jnp.interp` and
  first-order `map_coordinates(..., mode="nearest")`, so out-of-range queries
  clamp to grid boundaries rather than extrapolate.

DataFrame/xarray preparation is host-side. The interpolator treats its grid as
static data and uses JAX for numerical queries; this is not a promise to trace
arbitrary xarray operations or dynamic categorical strings.

Evidence: [`interpolation.py`](../src/sax/interpolation.py).
`test_interpolation_contract.py` verifies linear interpolation, boundary clamping,
JIT, and an interior gradient. Examples: [`12_data_parsers.md`](../docs/nbs/examples/12_data_parsers.md),
[`13_surface_models.md`](../docs/nbs/examples/13_surface_models.md). These examples
are not part of the four notebook tests run for the baseline.

## Neural fitting and losses

`sax.fit.neural_fit` takes a DataFrame, explicit target columns, and optional feature
columns (otherwise all numeric nontarget columns). It converts training arrays to
float32, standardizes each feature/target, trains a dense network with Optax Adam,
and returns parameters, normalization, prediction callable, metadata, and final loss.
Defaults include one hidden layer of width 10, tanh, seed 42, and 1000 epochs.
This is regression, not a constraint enforcing passivity or causality.

Array prediction returns target arrays; DataFrame prediction appends `<target>_pred`
columns while preserving the input index. Symbolic export requires an activation name supported by SymPy and can
render Python/JAX functions. `eval_neural_fit` executes generated source; do not
feed untrusted identifiers/fit artifacts into that path. Constant feature/target
columns use unit standardization scale for training, prediction, and export.
`test_fit_degenerate.py` verifies finite fits, nondefault DataFrame indices, and
export/prediction equivalence for all constant/nonconstant feature/target combinations.

`mse` is mean squared absolute error. `huber_loss` implements the smooth
pseudo-Huber expression `mean(delta**2*(sqrt(1+(abs(x-y)/delta)**2)-1))`, not the
piecewise Huber formula. `l2_reg` averages squared magnitudes of dictionary values
whose keys start with `w` or `b`; it assumes at least one contributing element.
Return annotations saying `float` do not force JAX scalar results into Python floats.

Evidence: [`fit.py`](../src/sax/fit.py), [`loss.py`](../src/sax/loss.py).
Import tests alone do not establish training convergence; targeted degenerate-data
and export checks supplement them. Arbitrary malformed/nonfinite datasets are not
covered by the constant-column policy.

## Supporting utilities

[`utils.py`](../src/sax/utils.py) supplies nested settings merge/update, dict
flatten/unflatten, model parameter/port renaming, grouped interpolation,
normalization/denormalization, and tabular reshaping. `merge_dicts` recursively gives
later dictionaries precedence; `update_settings` updates existing leaf keys, not
arbitrary new global parameters. Preserve wrapped model signatures where supported.
Tests: [`01_utils.ipynb`](../src/tests/nbs/01_utils.ipynb). Do not treat utility
renaming or flattening as a guaranteed lossless transformation of every netlist field.

## Native copying and serialization

Placed dictionaries/JSON are deserialized as `PlacedNetlist`; ordinary native data
uses `Netlist`. `native.copy_netlist` reconstructs using the source's concrete type
and preserves nested settings/info, arrays, topology, cell references, and geometry.
Circuit preparation stores Python-only legacy settings outside the native object;
pure native adapters/loaders retain native JSON-compatible storage constraints.
Evidence: `src/tests/test_native_settings.py`.
