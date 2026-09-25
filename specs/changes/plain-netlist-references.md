# Plain netlist references in SAX

**Status: in development.** This is a breaking migration pass. Compatibility will
be considered after the new path and its failing callers are visible.

The subsequent decision for this development branch is no legacy adapters;
the follow-up implementation is tracked in
[remove-sax-netlist-layer](remove-sax-netlist-layer.md). Earlier audit counts below
are historical evidence, not current test results.

## Intent and contract

SAX consumes plain `kfnetlist.Netlist` objects in a
`kfnetlist.HierarchicalNetlist` document.
A `RefNetlistInstance.netlist_id` selects a child definition in that document.
`component` remains the factory/model identity; a model for a reference ID may
replace that specific child, followed by qualified and unqualified factory models.
A leaf instance has no child to traverse. SAX neither guesses a child from the
factory name nor uses placement as a hierarchy identifier. kfnetlist validates
missing references and cycles when SAX ingests the document. Ordinary leaf models
and the numerical backends keep their existing behavior.

SAX uses kfnetlist's `sax` Git branch as a development dependency, locked to the
resolved commit in `uv.lock`. This is deliberately not a released-package contract.

## Scope and verification

Replace `PlacedNetlist` construction, deserialization, resolution, copying, and
flattening in the native SAX path. Make public `sax.Netlist` the kfnetlist class;
replace the old `netlists.py` dictionary algorithms, kfnetlist parser conversion,
and public YAML loader returns. Retain only backend-specific flat tables. Add
tests for plain object/dict/JSON documents,
model substitution, reference traversal, flattening, and dangling references.
Run the existing suite to identify behavior needing deliberate compatibility work.
Do not silently restore old cell-name inference or placed-instance handling during
this pass. Record failures and decide what to restore at the end of the PR.

## Open compatibility questions

- How should callers of `kfnetlist.extract` obtain plain referenced documents?
  That extractor's defaults were not changed by the upstream PR.
- Should legacy SAX dictionaries with `cell` or `placements` be adapted into
  explicit `netlist_id` references and a separate placement settings table?
- Which public transform/parser return types must retain older placed behavior?

## Public API replacement audit (2026-09-23)

`sax.Netlist` and `sax.saxtypes.Netlist` now name `kfnetlist.Netlist`.
`sax.netlist`, flattening, renaming, pruning, kfnetlist parsers, and the public
YAML loaders use plain netlist objects or `HierarchicalNetlist`. The old SAX
dictionary implementation in `netlists.py` was removed; its two remaining
connection helpers operate only on flat backend tables. The focused
explicit-reference suite passed 11 tests.

The subsequent non-notebook run passed 393 tests, failed 85, and skipped one.
This includes the earlier migration failures plus tests for the deliberately
removed SAX dictionary constructors, parser returns, public YAML loader returns,
and old dictionary transforms. It is an audit, not a passing release gate.

## Upstream hierarchy object (2026-09-25)

kfnetlist `sax` commits `9c5623d` through `d341152` add a Rust-backed
`HierarchicalNetlist` with
ordered entries, live mutable children, validation at construction and before
serialization/flattening, and methods for one-root or all-entry flattening. Its
JSON remains the existing `{id: netlist}` map. SAX now exports this as
`RecursiveNetlist`, returns it from public hierarchy constructors/loaders, and
accepts it as circuit input. Focused SAX tests passed 12/12. The subsequent
non-notebook SAX run passed 394, failed 85, and skipped one: the failure count
is unchanged from the public-API replacement audit, with one new passing test.
kfnetlist passed 310 Python tests (4 skipped, with Pydantic installed) and all
Rust workspace tests.
The later removal of legacy input adapters and SAX netlist modules is tracked
in [remove-sax-netlist-layer](remove-sax-netlist-layer.md).

## First compatibility audit (2026-09-23)

The first non-notebook `src/tests` run with the explicit-reference code passed
421 tests, failed 52, and skipped one. Those failures cluster in old placed
netlist inputs and serialization, PIC/legacy hierarchy that inferred child keys
from `component`, hierarchical probes over those implicit hierarchies, and
placement-derived model settings. This is expected migration evidence, not a
passing release gate. The focused plain-reference suite passed nine tests after
adding the explicit probe and cycle cases. Other platforms and notebooks were
not run in this audit. The old tests remain in place to guide the later
compatibility decision.

The final focused run of explicit-reference and two existing leaf/settings test
surfaces passed **16 tests**. Ruff check/format, `uv lock --check`, and
`git diff --check` passed; the runnable documentation example produced its
expected gain of 3. `ty check` on the changed source/test paths reported eight
diagnostics, mostly existing broad Mapping inference and one lowered settings
table inference; it is **not** a passing type gate. Notebook tests and other
platforms were not run.
