# Remove SAX's netlist layer

**Status: implemented in the development branch; legacy test migration remains.**

## Intent

SAX accepts `kfnetlist.Netlist` or `kfnetlist.HierarchicalNetlist` directly for
circuit construction. kfnetlist owns netlist construction, serialization,
validation, and generic transforms. SAX retains only model resolution, probe
planning, and translation into its numerical backends' tables.

## Scope

Remove `sax.native` and `sax.netlists`, including legacy SAX/PIC input adapters,
generic transform wrappers, and redundant kfnetlist parsers/loaders. Move the
remaining simulation compilation helpers beside the circuit builder and backend
connection helpers beside their backends. Keep the Mosaic parser as an external
schematic importer, emitting kfnetlist objects directly.

## Contract

`sax.circuit` and `sax.get_required_circuit_models` accept a plain Netlist or a
HierarchicalNetlist. A plain Netlist is wrapped under `top_level_name` (default
`top_level`); a hierarchy selects an explicitly requested root, otherwise
`top_level` when present, otherwise its first entry. Mapping/JSON/legacy/PIC
inputs and the former SAX netlist convenience APIs are unsupported in this
breaking development pass. A model for `netlist_id` or factory may still replace
a referenced child. Numerical backend topology and probe semantics remain the
same for valid kfnetlist inputs.

## Verification

- Explicit reference hierarchy, model substitution, pruning, probes, and
  numerical backend acceptance checks pass on kfnetlist objects.
- Direct dictionary, JSON, and legacy calls fail clearly.
- No source import or public export refers to `sax.native` or `sax.netlists`.
- Broader suite failures are reported as migration evidence, without restoring
  compatibility during this pass.

On 2026-09-25, `test_smoke.py` and `test_explicit_netlist_hierarchy.py`
passed **18 tests**. Scoped Pyright passed with zero errors and Ruff formatting
passed. Scoped Ruff lint passed with existing `CPY001` and `PLR0917` diagnostics
excluded. Four unrelated numerical/data test modules passed **34 tests**.
The broad non-notebook run stopped during collection on six old test modules
that import the removed `sax.native` or `sax.netlists`. Excluding those modules,
the audit completed with **293 passed, 99 failed, 1 skipped**; failures largely
exercise removed dictionary/JSON/PIC and convenience APIs. This is migration
evidence, not a passing full-suite claim or a reason to restore adapters.
