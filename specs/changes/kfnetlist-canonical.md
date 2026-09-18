# Canonical kfnetlist topology and unambiguous model identity

**Status: proposed, not implemented.** This investigation adds a regression and
plan, not a netlist migration. SAX inspected at `e23d710`; local kfnetlist at
`7379b68cbed3a7fe01bc11505d789fb53bae6cb9`. Implementation checklist:
[`todo.md`](../../todo.md). Existing contracts remain in
[circuits](../circuits.md) and [data workflows](../data-workflows.md).

## Required architecture

**kfnetlist is SAX's canonical public and internal netlist format.** Native
`Netlist` / `PlacedNetlist` objects are consumed directly, not adapted into SAX's
existing dictionary schema. Use `PlacedNetlist.cell` for hierarchical references
that must be distinct from factory `component`. `.pic.yml` and legacy SAX
input formats get adapters **into kfnetlist**, never the reverse.

```text
native kfnetlist objects ───────────────────┐
.pic.yml / legacy dicts → kfnetlist adapter ├→ native hierarchy / transforms
kfnetlist dict/JSON → native deserialization┘  → model resolution → backend arrays
```

Replace the old internal netlist schema rather than wrapping it with a new
importer. All hierarchy processing and topology transforms use native objects;
only numerical backend lowering produces solver-specific structures. Preserve
supported legacy inputs at the boundary, not their dict representation as the
simulation engine. Public netlist types and primary loaders should reflect the
canonical format; document intentional return-type changes.

This requires changes to model resolution and transforms, not just a type alias.
The current `TopLevelModule.to_netlists()` also needs compatibility handling for
legacy PIC features described below. Keep solver mathematics unchanged.

kfnetlist solves the factory-name half of
[SAX #120](https://github.com/gdsfactory/sax/issues/120), not the complete SAX
hierarchy problem. Its placement-aware flavor has the necessary separate `cell`
field; its default connectivity flavor and current PIC document schema do not.
No suffix-stripping heuristic can recover identities that an exporter discarded.

## Evidence and actual limitations

### gdsfactory and SAX

In gdsfactory 9.51.0, `get_netlist.py::_insert_netlist` first writes
`component_namer(inst_cell)`, then replaces it with `netlist_namer(inst_cell)` for
hierarchical instances. The default `CountedNetlistNamer` assigns different names
to parameterized cells: e.g. `identity_wrapper` and `identity_wrapper2`.
The original factory identity is not retained by this path.

SAX uses `instance['component']` for both model lookup and recursive-netlist
lookup (`circuits.py::_create_dag`, `_flat_circuit`; `netlists.py` transforms).
An explicit exact model shadows that subtree. A numbered name without an exact
model instead descends into the layout subnet. If its leaves have models, this
can succeed with the wrong physics, not merely raise a missing-model error.
[PR #121](https://github.com/gdsfactory/sax/pull/121) proposed conservative suffix
stripping but was closed unmerged. It is not an existing fix.

Two genuinely different factories may be named `coupler` and `coupler2`; model
presence and settings similarity do not prove shared origin. Do not reinterpret
either by spelling alone. An explicit `models['coupler2'] = models['coupler']`
is a valid current workaround **only when the caller knows their provenance**.

### What the local kfnetlist implementation provides

Source locations below are in `~/Projects/kfnetlist` at the revision above:

| Surface | Factory/model identity | Instantiated cell reference | Consequence |
| --- | --- | --- | --- |
| `NetlistInstance` / `Netlist` | `kcl`, `component`, `settings` | None | Good analytical leaves; insufficient for general recursive fallback |
| `PlacedInstance` / `PlacedNetlist` | Same fields | Separate `cell` | Can distinguish parameterized hierarchy variants |
| `extract(..., include_placement=False)` (default) | Factory name when available; otherwise cell name | Internal map built but not returned | Returned mapping is keyed by layout cell names, not factory names |
| `extract(..., include_placement=True)` | Same | `instance.cell = inst.cell.name` | Preserves both identities, although identity should not require geometry |
| `flatten_netlists` | Preserved for surviving leaves | Explicit `instance_cell_maps` or placed `cell` | Confirms factory name alone is insufficient; unresolved instances can be skipped |
| `kfnetlist_schema.Instance` | One `component` plus settings/info/array | No independent `cell` or `kcl` | Document schema is not yet the complete canonical identity contract |

Evidence: `src/kfnetlist/extract/_algo.py` (`_create_inst_entry`, `extract`),
`src/kfnetlist/_flatten.py`, `crates/kfnetlist-core/src/{instance,placement,flatten}.rs`,
`crates/kfnetlist-schema/src/{document,convert}.rs`,
`contributing/circuit-schema.md`. Protobuf's `Instance.module_name` does not add
an independent factory-versus-cell distinction either.

The schema's `TopLevelModule` supports `modules`/`toplevel` and bare documents,
YAML and protobuf round trips. However:

- `Module.to_netlist()` preserves instance settings/info and connectivity, not
  module settings, placements, routes, or root selection in the returned mapping.
- `${settings.x}` remains a string; this is not parameterized module elaboration.
- Route data survives in the document but is not converted to connectivity by
  `Module.to_netlist()`. SAX currently turns route `links` into logical nets.
- Schema instances require dictionaries and reject unknown fields. Legacy SAX
  permits shorthand strings, callables/partials, and filters unknown fields.
- Array dimensions are `na`/`nb`, with one-based `<ia.ib>` references (and zero-based
  `[n]`). Legacy SAX uses `columns`/`rows` and zero-based `<column.row>`.
- kfnetlist settings are JSON-compatible data. JAX arrays and Python callables
  supported in SAX cannot all be round-tripped through native JSON storage.

### SAX's existing kfnetlist adapter is a lossy compatibility adapter

`src/sax/parsers/kfnetlist.py::_convert_instance` copies only `component`, settings,
and arrays: it drops `kcl`, `cell`, `info`, and placement. `_convert_flat` ignores
the declared port list, attaches external ports to the first instance member,
and represents an n-terminal net as a chain of connections. External-only nets
and unconnected declared ports are not preserved. The 33 existing adapter tests
include flat numerical circuits and recursive dictionary conversion, but not
actual fallback into differently named parameterized child cells.

Merely routing canonical input through this adapter would recreate data loss.
Merely changing the adapter to prefer `cell` would break shared factory-model
lookup. Both identities must survive through resolution, not just import.

### Runtime observations

Installed environment: Python 3.12.14, gdsfactory 9.51.0, kfactory 3.0.4,
kfnetlist 0.3.0. A real `@gf.cell` wrapper around a straight, instantiated at
lengths 10 and 20 with four exposed ports, produced:

| Export/evaluation | Observed result |
| --- | --- |
| `top.get_netlist(recursive=True)` | Instance components `identity_wrapper`, `identity_wrapper2` |
| Plain kfnetlist extraction | Both components `identity_wrapper`; hierarchy keys `identity_wrapper_L10`, `identity_wrapper_L20` |
| Placed kfnetlist extraction | Same components, plus the correct distinct `cell` values |
| Either extraction through current SAX parser, analytical factory model `transmission = length` | Transmissions 10 and 20, as expected |
| Either extraction through current SAX parser, only straight leaf model | `ValueError: Missing models ... identity_wrapper`, despite child cells being present |

The library straight's factory name was qualified
(`straight_gdsfactorypcomponentspwaveguidespstraight`), unlike legacy short
`straight`. A migration must not assume gdsfactory function names and kfactory
factory names are interchangeable. Any short-name alias policy must be explicit
and collision-checked. Installed kfactory's `cell.netlist()` wrapper does not
expose `include_placement`; direct `extract` with `wrap_kdb_instance` was used.

The local checkout includes `kfnetlist.kfnetlist_schema`; the installed 0.3.0
package does not. Local source and installed package share a version label but
not these capabilities. Schema findings here are **source inspection**, not a
claim that the local Rust/schema tests ran. The external checkout was not edited.

## Proposed contract

### Representation boundary

Use an explicit root ID and a mapping of cell IDs to native kfnetlist objects.
SAX reads factory `component`, `kcl`, settings, and placed-instance `cell` directly
from those objects. Keep factory and cell IDs distinct, with library qualification
when needed. Preserve display names separately from generated Python-safe
identifiers; detect normalization collisions.

Adopt `PlacedNetlist` now for hierarchy that requires separate cell references;
its placement coordinates need not participate in simulation. Accept ordinary
`Netlist` for topology resolvable without those references, and diagnose missing
identity if recursive fallback cannot be resolved. Do not convert native placed
instances into SAX dictionaries or duplicate their identity in a SAX-owned schema.
Native dict/JSON serialization must preserve both identities.

Moving an optional cell reference into ordinary `NetlistInstance` would be useful
upstream cleanup, but is not a prerequisite for this migration. The PIC adapter
can construct placed instances for known subcircuit references without waiting
for a new document schema. Missing factory provenance in old exports remains
unrecoverable: never manufacture it from settings hashes or suffixes. Explicit
caller metadata must be validated and incorporated into native instances at the
legacy boundary; reject conflicts with existing references.

Keep root/module metadata and provenance that native connectivity cannot hold
in a retained document or small context alongside the native objects, not in
model kwargs. Keep native placement data on placed instances; do not duplicate
topology or instance identity in that context. Keep Python-only callable bindings and
non-JSON numerical settings in SAX-owned tables keyed by normalized instance
path; preserve existing defaults/overrides and JAX behavior without serialization.
This is a static-topology contract, not a claim that a Rust object is a JAX pytree.
Do not create a second general-purpose competing netlist schema in SAX.

### Resolution policy

Use one resolver for dependency discovery, required-model diagnostics, circuit
construction, model-aware flattening, and hierarchical probe traversal:

1. Explicit instance model binding, if supplied by the legacy callable adapter.
2. Explicit cell-specific model override, preserving intentional exact legacy
   numbered-model overrides.
3. Exact qualified factory model; an explicitly allowed unqualified factory
   binding only when unambiguous.
4. Recurse via the explicit cell reference if no analytical override exists.
5. Raise an error containing parent/instance path, factory, cell reference, and
   attempted bindings if neither is resolvable.

Factory models apply to **every** variant, with that instance's own settings.
Select the model before descending into the layout subtree: an analytical
coupler must not require dummy models for its constituent shapes. Preserve
specialized cell overrides ahead of shared factory defaults. Do not key compiled
child circuits solely by factory name; parameterized cells remain distinct.

Legacy dictionaries without separate metadata retain their exact-name behavior.
An explicit alias/provenance map can upgrade them; otherwise missing provenance
is unrecoverable. Do not promise that the raw issue fixture can be magically
repaired. Consider an opt-in diagnostic for ambiguous legacy recursive input,
not a blanket warning on every legitimate subcircuit. When implementing, replace
the reproduction's xfail with canonical/explicit-metadata conformance and a
legacy ambiguity test; do not make it pass by guessing suffixes.

Root selection must be explicit in normalized input. Preserve old first-entry /
`top_level_name` behavior at the legacy boundary, and document argument versus
PIC `toplevel` precedence. Validate hierarchy references and cycles with useful
paths; distinguish the effective model dependency DAG from permitted optical
feedback. Probe requests inside a model-replaced opaque subtree need an explicit
error or opt-in decomposition policy, never accidental loss of analytical physics.

### Compatibility and lowering

- Make native kfnetlist objects/hierarchies the direct input to `sax.circuit`.
  Current dictionaries/callables remain supported through legacy-to-kfnetlist
  conversion at the boundary; there is only one native circuit-construction path.
  Deserialize kfnetlist dict/JSON into native types, not the legacy SAX schema.
- Make primary `load_netlist` / `load_recursive_netlist` PIC loading produce native
  objects/hierarchies. Document public type/return changes and migration examples.
  If temporary dict-returning compatibility helpers are retained, label and
  isolate them; preserving their return shapes must not dictate internal design.
- Keep `.pic.yml` supported, including multi-file directory discovery, top-first
  order, custom suffix, name cleaning, and duplicate rejection. A suffix does not
  uniquely identify a YAML dialect. Add `modules`/`toplevel` support through an
  explicit schema adapter, with ambiguity errors where necessary.
- Translate legacy `{p1,p2}` nets, `connections`, route `links`, arrays, settings,
  and metadata before native validation. Preserve legacy `info -> settings`
  precedence only in the legacy adapter; canonical `info` stays metadata.
  Encode known legacy child references as native `cell` values and retain factory
  metadata where available; accepting legacy input does not recover lost provenance.
- Do not introduce expression evaluation or change root-setting semantics as a
  side effect. Retain expressions in stored documents and reject unsupported
  numerical elaboration explicitly; current SAX does not evaluate them either.
- Preserve n-terminal net membership and port declarations until backend lowering.
  Characterize KLU repeated-endpoint semantics before choosing that lowering;
  canonical net merging must not silently change existing chain/edge physics.
  Unsupported external-only nets, port aliases, or junctions require specific
  diagnostics, not silent dropping or an invented splitter. Other backends retain
  their restrictions. This is a migration gate, not permission to simplify nets.
- Transform native topology on copies. Migrate array expansion, pruning,
  rename/flatten, and probes to native objects, using native operations where
  semantics match; do not convert back to legacy dictionaries to reuse the old
  implementation. Adapt generated signatures so identities and settings survive.
  Leave scattering conventions and solver mathematics unchanged. Solver-specific
  arrays/indices are compiled output, not an alternate canonical netlist schema.

## Rollout and decisions required

The checklist in `todo.md` sequences characterization, native hierarchy adoption,
legacy-to-native input conversion, resolution, transforms, and completing the
canonical build path. It is not a plan to improve the existing kfnetlist-to-SAX
adapter first. Retire the old internal schema and conversion path; any retained
legacy export/helper APIs must be isolated and documented. No blanket flattening:
it would remove the analytical model boundaries this work is intended to protect.

Before making kfnetlist mandatory, resolve **Python support**: SAX advertises
3.11+, local kfnetlist requires 3.12+. Preferred options are upstream 3.11 support
or an explicitly approved SAX minimum-version change, not an invisible drop.
Verify actual published wheel capabilities on supported OSes and choose a real
minimum release for schema/identity APIs. Do not rely on a local editable install
or assume the 0.3.0 version label suffices.

Other decisions to settle in the first implementation slice: exact public
spelling for cell overrides and qualified model keys; PIC root-override policy;
supported canonical multi-terminal/alias cases. The architecture is already
settled: native kfnetlist throughout, `PlacedNetlist.cell` for hierarchy, and
PIC/legacy adapters into it. These API details do not reopen that decision. If identity preservation,
Python support, or lowering parity cannot be established, stop before changing
the canonical path and report the fixture/blocker and upstream change needed.

## Verification completed for this investigation

- Final `just smoke`: **5 passed, 1 xfailed**, 3.37s pytest / **4.33s wall-clock**,
  existing environment, no dependency synchronization. The new parameterized
  `test_counted_hierarchy_model_identity` has a passing explicit-alias control and
  a strict `xfail(raises=AssertionError)` numerical reproduction. No layout-tool
  import is added to smoke.
- `pytest src/tests/test_smoke.py --runxfail -k counted-name-loses-factory -q`:
  **one intended failure**, at the final transmission assertion: zero instead of
  `0.3 * exp(1.55j)`. Construction succeeds and the first variant is correct.
- Smoke + kfnetlist parser + recursive YAML + hierarchy validation suites:
  **45 passed, 1 xfailed** in 4.51s. The xfail is not a conformance pass.
- Real extraction comparison above executed in the existing environment. Temporary
  script/log: `/tmp/sax-kfnetlist-investigation.{py,log}`; the fixture recipe and
  results above remain the durable record.
- Full SAX suite, local kfnetlist build/schema suite, cross-platform wheels, and
  lint/type checks: **not run** for this test-and-plan change.

Architectural acceptance must prove that native circuit construction succeeds
with the old `parse_kfnetlist` conversion disabled; primary PIC loaders and topology
transforms produce native kfnetlist types; and no legacy netlist-schema coercion
occurs between native input and backend lowering. Numerical equivalence alone
is not sufficient if SAX still uses the old schema internally.

Migration acceptance must additionally prove numerical parity across legacy YAML,
legacy dictionaries, native objects, JSON, and supported document inputs; distinct
parameterized hierarchy fallback; overrides and missing-leaf pruning; shared
subcircuits; namespace collisions; arrays; cycles; probes; multiply connected
backend restrictions; settings precedence, JIT, and gradients. Keep smoke below
10 seconds locally, put heavy layout integration outside it, and run the full SAX
suite including notebooks before declaring the migration implemented.
