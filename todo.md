# Canonical kfnetlist migration

**Planning only; migration not yet implemented.** kfnetlist is to become SAX's
canonical public and internal netlist format—not an additional import format.
Use native `Netlist` / `PlacedNetlist` objects directly, with `PlacedNetlist.cell`
for hierarchical references. `.pic.yml` and legacy SAX dictionaries get adapters
**into kfnetlist**, never the reverse. Preserve their supported input behavior,
not SAX's old dictionary schema as the internal representation.

Required data flow:

```text
native kfnetlist objects ───────────────────┐
.pic.yml / legacy dicts → kfnetlist adapter ├→ native hierarchy / transforms
kfnetlist dict/JSON → native deserialization┘  → model resolution → backend arrays
```

Detailed findings, proposed contracts, evidence, and unresolved decisions:
[`specs/changes/kfnetlist-canonical.md`](specs/changes/kfnetlist-canonical.md).

The previous 24-item remediation is complete; its history remains in
[`specs/changes/todo-remediation.md`](specs/changes/todo-remediation.md) and Git.
This replaces that completed checklist, not its completion evidence.

## Investigation complete

- [x] Trace [#120](https://github.com/gdsfactory/sax/issues/120) through actual
  gdsfactory export and SAX model lookup. Recursive export replaces factory names
  with counted cell names; SAX can silently simulate the wrong layout subnet.
- [x] Inspect `~/Projects/kfnetlist` at `7379b68` and compare live extraction.
  Factory `component` is stable; only placed instances retain separate `cell`.
  Default plain instances cannot fully identify child netlists. SAX's adapter
  discards `cell` even when present. Analytical substitution works with native
  extraction, but hierarchical fallback still fails.
- [x] Assess PIC schema and packaging. Local document schema still has only one
  `component`; its connectivity conversion omits module settings/routes/placement.
  Local schema APIs are absent from installed kfnetlist 0.3.0. SAX supports Python
  3.11+, whereas local kfnetlist requires 3.12+.
- [x] Add a cheap strict xfail and passing explicit-alias control to
  `src/tests/test_smoke.py`. Numerical failure: second variant transmits zero
  instead of analytical coupling. `just smoke`: **5 passed, 1 xfailed, 4.33s wall**.
  `--runxfail` confirms the intended assertion fails, not import/construction.
  Relevant regressions: **45 passed, 1 xfailed**. No runtime/dependency change.

## Implementation sequence

Each numbered slice should have focused tests and its own reviewable commit
(or closely related commits). No suffix stripping and no automatic full flattening.
Run smoke after each slice; update baseline specs only as behavior is implemented.

**Progress:** slices 1–3 partially implemented (native module, factory/cell
resolution, PIC loader, native probes). Slice 7 cannot complete until the
directed-connection blocker is resolved (see change doc). Legacy input still uses
the legacy circuit path; native input uses the native builder.

### 1. Agree the identity and compatibility boundary

- [ ] Define independent factory/model identity and instantiated-cell identity,
  explicit root selection, qualified library keys, and metadata ownership.
  Use native placed-instance `cell` references; decide the public spelling of
  cell-specific overrides and root selection, not a competing SAX identity schema.
  Confirm current exact legacy overrides stay ahead of shared factory defaults.
- [ ] Resolve Python 3.11 versus 3.12 before adding a mandatory dependency. Verify
  supported wheels and released APIs; choose a real minimum kfnetlist version.
  Do not change the lockfile or Python minimum incidentally.
- [x] Add characterization fixtures: two variants sharing a factory, two distinct
  child topologies without a factory model, qualified-name collisions, legitimate
  numeric factory names, and a model-replaced subtree with missing leaf models.
  Preserve a real gdsfactory extraction fixture/integration test outside smoke.
  Implemented in `src/tests/test_native_kfnetlist.py` and smoke; native only.
  **Exit:** the chosen resolution policy is testable without naming heuristics.

### 2. Adopt native kfnetlist hierarchy, using PlacedNetlist

- [ ] Use a cell-ID mapping of native kfnetlist objects plus an explicit root.
  Read `component`, `kcl`, settings, and `cell` directly from native instances.
  Use `PlacedNetlist` for hierarchy requiring separate cell references; accept
  plain `Netlist` for circuits resolvable without those references. Diagnose
  unresolved plain hierarchy rather than guessing cell names.
- [x] Preserve both identities in native copying, dict/JSON serialization, and
  transforms. Keep geometry optional for simulation; retain placement data when
  supplied. Root/module metadata and Python-only bindings may live alongside the
  native hierarchy, but must not duplicate its topology or instance identities.
  Implemented for native input (adapted from legacy/placed/PIC); legacy transforms
  still operate on dictionaries.
- [x] Treat moving `cell` into ordinary kfnetlist instances as optional upstream
  cleanup, not a prerequisite or reason to build a SAX-owned substitute schema.
  **Exit:** two same-factory/different-cell native instances survive serialization
  and remain directly usable by SAX without conversion to legacy dictionaries.
- [x] Add native flat/top-level/hierarchical probes and internal-port policy.
  `expand_probes_tables`, `handle_internal_ports`, `plan_hierarchical_probes`.
- [ ] Extend native input to all remaining transform surfaces (rename/flatten,
  pruning) once the canonical circuit path is unblocked.

### 3. Make native objects the circuit and netlist API foundation

- [ ] Make `sax.circuit` consume native Netlist/PlacedNetlist and native hierarchy
  mappings directly. Native dict/JSON inputs deserialize into those same types;
  this is not an adapter into SAX's old netlist schema. Ensure no input mutation.
- [ ] Replace internal dependence on SAX's old TypedDict/Pydantic netlist schema.
  Define public netlist types/helpers around kfnetlist; document intentional
  return-type changes. Any temporary legacy-returning wrappers must be explicit,
  isolated compatibility APIs, not the format consumed by circuit construction.
- [ ] Move legacy dictionary and callable/partial conversion to the input edge,
  producing native topology plus Python-only bindings/settings where necessary.
  Never JSON-coerce callables or JAX settings. Do not invoke `parse_kfnetlist`'s
  old kfnetlist-to-SAX-dictionary conversion anywhere in the canonical path.
  **Exit:** native objects are the single topology representation throughout
  hierarchy processing, with no conversion back into the legacy netlist schema.

### 4. Keep `.pic.yml` an input format, not a second simulation engine

- [x] Make the primary `.pic.yml` loading path an adapter that returns native
  kfnetlist objects/hierarchies. `native.load_pic_yaml` handles flat and
  `modules`/`toplevel` documents. Public `load_netlist`/`load_recursive_netlist`
  still return dictionaries (transitional); wiring them to native is pending.
  Add numerical end-to-end multi-file `.pic.yml` tests.
- [x] Translate legacy shorthand instances, `{p1,p2}` nets, connections, route
  links, `columns`/`rows`, and zero-based references into native kfnetlist objects.
  `from_legacy_flat`/`_scan_legacy_arrays`. Legitimate provenance is preserved
  where present; lost factories are never inferred from suffixes.
  Preserve legacy `info -> settings` precedence in this adapter only.
- [ ] Add explicit `modules`/`toplevel` document support without confusing it with
  a recursive dictionary. Define root-argument precedence, one-based native
  versus zero-based legacy arrays, and retained layout/module metadata.
  Preserve or explicitly reject unsupported expressions; do not invent YAML
  expression evaluation or silently drop route links.
  **Exit:** equivalent legacy YAML, dict, and native fixtures simulate identically;
  unsupported document features produce precise errors rather than altered physics.

### 5. Resolve models before traversing implementation subcircuits

- [ ] Centralize precedence: direct instance binding → explicit cell override →
  qualified/unambiguous factory binding → referenced child cell → diagnostic.
  Use it for DAG construction, required-model reporting, and circuit building.
  Models replacing a subtree must not require that subtree's leaf models.
- [ ] Keep compiled child circuits keyed by cell identity, not factory identity.
  Verify distinct variant settings, nested/global overrides, shared subcircuits,
  exact numbered overrides, namespace collisions, cycles, and optical feedback.
- [ ] Replace the issue xfail with passing canonical/explicit-metadata acceptance
  and a legacy ambiguity test once supported. Preserve the explicit-alias control;
  raw legacy data with lost provenance must not be "fixed" by suffix guessing.
  **Exit:** both analytical variants and no-model hierarchical fallback are
  numerically correct, and required-model diagnostics use the same resolver.

### 6. Migrate transforms and make net lowering explicit

- [ ] Make array expansion, pruning, rename/flatten, and probe traversal operate
  on and return native topology, preserving both identities and instance settings.
  Use native operations where their semantics match SAX's requirements; do not
  convert through legacy dictionaries to reuse old transforms. Adapt model
  signature generation and test input isolation. Define behavior for probes
  inside an analytically replaced opaque subtree.
- [ ] Retain native net membership and declared ports until lowering. Characterize
  n-terminal nets, repeated endpoints, aliases, singleton/unconnected ports, and
  external-only nets. Prove legacy KLU parity and keep other backend restrictions;
  unsupported cases must error, not silently become arbitrary chains/splitters.
  **Exit:** transform equivalence and backend-lowering fixtures pass, including
  arrays/probes and multiply connected cases; no solver mathematics is changed.

### 7. Switch the internal default and verify compatibility

**Blocked** on directed-connection parity: kfnetlist `Net` is undirected, so the
forward backend cannot recover declared signal direction from native nets. Per the
stop condition, the canonical switch is not made. Legacy input keeps the legacy
path; native input uses the native builder. See the change doc and
`test_native_forward_backend_direction_blocker`.

- [ ] Complete the native circuit path and retire the old internal netlist schema.
  Keep supported legacy input adapters pointing into kfnetlist, not a parallel
  legacy simulation path. Document public loader/parser/type changes and isolate
  any explicitly retained compatibility wrappers.
- [ ] Add an architectural regression: native circuit construction must succeed
  with the old kfnetlist-to-SAX adapter disabled. Assert that `.pic.yml` loading
  and topology transforms return native types and that no legacy schema coercion
  is used between native input and numerical backend lowering.
- [ ] Add the approved dependency/version policy, deliberately update the lockfile,
  and run `uv lock --check` plus isolated installation/import tests on supported
  Python/OS combinations. Keep layout extraction dependencies out of core imports.
- [ ] Run focused identity/YAML/parser/transform/backend suites, smoke (<10s locally),
  and full `src/tests` including notebooks. Compare KLU/FG results, broadcasting,
  JIT, and real-objective gradients across equivalent input formats.
- [ ] Update baseline specs, user docs, migration examples, and deprecation notes
  only for implemented behavior. Explain exact model aliases for old ambiguous
  exports and preferred identity-preserving extraction for new designs.
  **Exit:** all acceptance cases pass; report skips/unsupported features honestly.

## Stop conditions / non-goals

Do not implement this entire plan merely to complete the investigation. Before
implementation, settle the material API/dependency decisions in slice 1. If an
upstream identity API, release/wheel support, or net-lowering parity blocks a
slice, stop before switching the canonical path; record the minimal failing
fixture, attempted approach, blocker, and decision/upstream release needed.

This is not a solver rewrite, layout-routing engine, protobuf mandate, arbitrary
parameter-expression interpreter, or permission to discard `.pic.yml` support.
Do not modify the sibling kfnetlist repository without separate authorization.
