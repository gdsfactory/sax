# SAX native-netlist remediation

**Status: active goal.** The user authorized all stages and delegated unresolved
decisions to agent judgment. Implementation started from clean commit `6e5d767`.

## Goal and completion criteria

Make the native kfnetlist circuit path preserve supported SAX behavior, resolve
all 18 branch-review findings, and establish the remaining migration contracts
and verification evidence. Use `PlacedNetlist` for concrete hierarchy references
and remove the `forward` backend. Keep the implementation within SAX while the
upstream schema discussion continues.

The goal is complete when all five stages meet their verification criteria,
remaining API decisions are documented, and the final evidence distinguishes
passed, failed, and unrun checks. External blockers must be recorded rather than
marked complete. Goal execution is now authorized.

## Agreed decisions and boundaries

- Use `component` for analytical-model lookup and `PlacedInstance.cell` for child
  netlists. Preserve both identities in serialization and transforms. Do not add
  a competing hierarchy schema or instance enum in SAX, guess identities from
  suffixes, or automatically flatten every subcircuit.
- Implement independent native copying and selective flattening in SAX now,
  behind small helpers with behavioral tests. Replace the mechanisms with
  upstream APIs once available in a supported dependency; model-selection policy
  remains SAX's responsibility. Use existing cell-level exclusions when sufficient.
- Remove the `forward` backend, its exports, direction hints, and native-input
  rejection guard. Physical forward/backward probe measurements remain supported.
- Preserve legacy dictionaries and `.pic.yml` as input formats, numerical indexing
  conventions, and static topology. Do not change solver mathematics or add YAML
  expression evaluation. Python >=3.12 and mandatory kfnetlist are already adopted;
  do not change dependencies or the lockfile incidentally.
- Preserve unrelated work and do not modify the sibling kfnetlist repository.
  The old completed 24-item remediation remains historical evidence in
  `specs/changes/todo-remediation.md`, not a second active checklist.

## Execution and stage commits

Once the user starts this goal, read the relevant specs and implement one stage
at a time. Update task checkboxes and the stage's evidence as work progresses.
For every stage, record actual commands, outcomes, remaining limits, and any
changed decisions. Use existing regression coverage where sufficient; add tests
for uncovered behavior rather than mirroring implementation details.

After a stage's outcome and verification criteria are met, mark it complete and
commit its related changes together with the updated `work.md`, as required by
`AGENTS.md`. Include the completed-stage status and evidence in that commit.
Exclude unrelated changes. Drafting this plan completes no execution stage and
requires no stage commit. Do not mark tests or tasks complete merely because
similar historical checks passed.

| Stage | Outcome | Status | Verification |
| --- | --- | --- | --- |
| 1 | Backend scope and identity/API boundaries settled | Complete | 80 focused tests passed |
| 2 | Input settings, placements, and native identities preserved | Complete | 55 acceptance + 51 regression tests passed |
| 3 | Hierarchy, transforms, probes, and topology corrected | Not started | Not run |
| 4 | PIC workflows and real extraction validated | Not started | Not run |
| 5 | Compatibility evidence, documentation, and cleanup complete | Not started | Not run |

## Stage 1 — Retire forward and settle contract boundaries

**Outcome:** native connectivity needs no direction metadata, and the identity
and public API rules needed by later stages are explicit.

- [x] Remove `forward` from backend registration, accepted backend types/names,
  lower-level APIs, and public exports. Remove native direction hints and the
  special rejection of undirected inputs. Resolves review findings **6, 7, 13**.
- [x] Update affected tests, specs, user documentation, and example navigation.
  Retain useful KLU/FG reconvergence coverage and all physical probe semantics.
- [x] Settle the model-key/namespace contract before introducing new behavior:
  library-qualified identities (`kcl` plus `component`), cell-specific overrides,
  and explicit callable bindings. Preserve intentional exact legacy overrides.
  Document precedence shared by construction, required-model discovery, probes,
  and flattening; do not add implicit suffix-based aliases.
- [x] Specify unresolved-identity diagnostics with parent/instance path, factory,
  cell reference, and attempted bindings. Cover same-name factories in different
  libraries, legitimate numeric factory names, and missing child references.
- [x] Decide and record public loader/type contracts and the permitted backend
  coercion boundary. Do not change public return types solely to match stale
  prose. This establishes the decision needed for finding **18** and stage 4.

**Decisions:** use `library::component` keys; exact cell overrides win, and bare
factory bindings are accepted only when the factory has a single library in the
supplied hierarchy. Keep legacy public loader dictionary returns for compatibility
and explicit `native.load_*` native returns. Backend discovery may validate its
lowered tables using local identifier aliases; native topology and public identities
must remain intact. Missing identities report paths and all attempted lookups.
Callable instance bindings will become instance-local in stage 2. These decisions
use the user's authorization to exercise judgment rather than pause for approval.

**Verification:** rejected `forward` selection has a useful error; remaining
backend selection/imports work; focused KLU/FG/additive and probe checks pass.
New identity-policy tests establish only the agreed namespace behavior.
**Evidence:** existing `.venv`, focused pytest across backend selection/dependency/
restrictions, native identity/input, probes, and hierarchy validation: **80 passed**
in 21.68s. Formatting and `git diff --check` passed. Existing branch-wide lint/type
debt remains assigned to stage 5; this stage commit bypasses hooks rather than
claiming they pass. **Stage commit:** this stage's commit includes this record.

## Stage 2 — Preserve settings, placements, and native types

**Outcome:** adaptation and serialization preserve supported simulation inputs
without forcing Python-only values into native JSON storage.

- [x] Preserve keyword partial arguments as per-instance settings and reject
  positional partials (**1**). Cover distinct partials of the same function and
  explicit evaluation overrides in `_replace_callable_instances`.
- [x] Restore the legacy `info`-to-settings merge and its conflict precedence
  (**2**); retain native `info` as metadata.
- [x] Keep JAX/NumPy arrays, complex values, and other supported non-JSON model
  settings in SAX-owned tables outside native topology serialization (**4**).
  Preserve defaults, overrides, broadcasting, JIT, and real-objective gradients.
- [x] Carry legacy placement data into the native path with existing normalization
  semantics (**3**). A model given `x=7` must receive `x=7`, not the default zero.
  Preserve supplied placed geometry without requiring it to affect simulation.
- [x] Deserialize placed dictionaries/JSON as the correct native type (**10**)
  and provide independent copying that preserves type and data (**11**).
  Exercise ordinary and placed netlists, nested settings, arrays, metadata,
  cell references, geometry, and input isolation. Keep the helper replaceable by
  the upstream copy API when available.

**Verification:** focused settings/adaptation tests reproduce the old failures
and pass after correction. Native objects and their supported serialized forms
produce equivalent asymmetric circuit results. Copies can be transformed without
changing the originals; JIT/gradient checks use real objectives where relevant.
**Evidence:** existing `.venv`: native settings/input/identity + smoke **55 passed**
(19.34s); broadcasting, legacy netlist/circuit, and probes **51 passed** (29.81s).
New tests include trace-time construction gradients, independent same-named callable
bindings, asymmetric placed JSON fallback, and ordinary/placed copy isolation.
Formatting and whitespace checks passed. Hooks remain deferred to stage 5's
quality cleanup. **Stage commit:** this stage's commit includes this record.

## Stage 3 — Correct hierarchy, transforms, probes, and lowering

**Outcome:** topology preparation preserves model boundaries and connectivity,
rejects unsupported cases explicitly, and does not mutate caller input.

- [ ] Restore pruning before dependency/model validation (**5**), including
  disconnected layout-only subcircuits while retaining probe-required instances.
  Fix hierarchy dispatch (**12**) and synthetic graph-node collisions (**15**).
  Preserve root/order, native types, and input isolation. Cover required-model
  discovery and a connected instance named `__port_0`.
- [ ] Make selective flattening preserve analytically replaced instances (**9**),
  even when their child definitions are present. Use existing cell exclusions
  where sufficient and isolate any additional SAX mechanism for replacement by
  upstream support. Check rename/flatten/prune equivalence, settings, identities,
  array handling, and caller-input isolation.
- [ ] Reject hierarchical probe-port collisions at every parent level (**8**).
  Preserve explicit rejection of probing inside an analytically replaced opaque
  subtree; never silently substitute the layout's physics to satisfy a probe.
- [ ] Validate effective root ports after internal-port handling and probe
  expansion (**16**). Restore the useful at-least-one-port diagnostic while
  preserving valid probe-only circuits.
- [ ] Decide and implement supported native lowering for declared unconnected
  ports and external-only nets (**14**); extend acceptance cases to aliases,
  singleton ports, n-terminal nets, and repeated endpoints. Establish KLU parity
  and restrictions of the remaining backends. Unsupported cases must error rather
  than disappear or acquire arbitrary junction/splitter behavior.

**Open decision:** numerical semantics for unsupported ports, aliases, and
junctions remain unresolved. Characterize current behavior and choose supported
semantics or explicit rejection before implementing that task. Existing legacy
multi-link tests do not establish every native-net case.

**Verification:** focused hierarchy/transform/probe tests pass, including missing
unused models, collisions, shared child definitions, arrays, and input isolation.
Use asymmetric numerical fixtures for lowering and model-boundary equivalence.
**Evidence:** not run. **Stage commit:** pending execution and verification.

## Stage 4 — Validate PIC workflows and real extraction

**Outcome:** documented input paths agree where they carry equivalent identities,
and the migration is tested against a real extracted design.

- [ ] Apply the public loader/type decision from stage 1 and reconcile the
  migration audit and tests (**18**). Test the actual public APIs; the existing
  `test_public_pic_loaders_produce_native` only exercises `native.*` helpers.
- [ ] Define and test root precedence among explicit arguments, PIC `toplevel`,
  and legacy first-entry roots. Define module metadata/settings and native versus
  legacy array indexing. Preserve or explicitly reject unsupported expressions
  without implementing expression evaluation or changing root-setting semantics
  incidentally; retain supported route links.
- [ ] Verify multi-file discovery, custom suffixes, root/order preservation, name
  normalization and duplicate-name rejection. Native file keys currently use raw
  stems while the legacy loader uses `clean_string`; resolve that compatibility
  difference deliberately. Add numerical `modules`/`toplevel` coverage.
- [ ] Check parity across legacy dictionaries, PIC input, native objects, and
  placed dict/JSON when equivalent identities are present. Cover shared/deep
  subcircuits, distinct variant settings, nested/global overrides, arrays,
  dependency cycles versus optical feedback, and caller-input isolation. Retain
  an explicit legacy-alias control and the no-suffix-guessing case.
- [ ] Add a durable gdsfactory/kfactory extraction integration fixture using
  `PlacedNetlist`, outside smoke. Exercise two variants sharing a factory,
  distinct-child fallback without an analytical model, and analytical replacement
  whose layout leaves have no models. Do not rely solely on hand-built objects.

**Verification:** public loader and numerical PIC tests pass; actual extraction
works with the agreed identity/override policy. Reuse existing tests where they
already establish an acceptance case. The earlier extraction investigation alone
is not an automated regression fixture.
**Evidence:** not run. **Stage commit:** pending execution and verification.

## Stage 5 — Complete verification, documentation, and cleanup

**Outcome:** all review findings have evidence of resolution, the supported public
contract is documented, and remaining verification limits are explicit.

- [ ] Fix introduced lint/type/formatting problems (**17**) without auto-fixing
  unrelated baseline issues. Run scoped checks and `git diff --check`; investigate
  remaining failures rather than relying on the historical error counts.
- [ ] Run the full `src/tests` suite including notebooks after focused checks pass.
  Run smoke with its local under-10-second target, and verify the lockfile without
  regenerating it. Use isolated notebook kernel setup where needed; record the
  environment and exact passed/failed/not-run results.
- [ ] Verify required APIs against a published supported kfnetlist release and
  supported installation platforms. Distinguish released-package evidence from a
  local checkout and record unavailable platform checks as gaps, not passes.
- [ ] Update user docs, examples, specs, migration guidance, and any agreed
  deprecation notes. Explain identity-preserving PlacedNetlist extraction, factory
  and cell overrides, explicit aliases for ambiguous legacy exports, loader/root
  contracts, and forward-backend removal. Run affected doc/example checks.
- [ ] Audit all 18 findings and retained acceptance items against implementation
  and evidence. Update this document's stage status, verification record, and
  remaining work before the final stage commit. Do not declare the goal complete
  while required decisions, fixes, or checks remain unresolved.

**Verification:** the final audit accounts for every finding and all five stages;
checks meet the agreed completion criteria. External verification blockers remain
visible until resolved or the user explicitly revises the required scope.
**Evidence:** not run. **Stage commit:** pending execution and verification.

## Upstream follow-up

These issues are feature requests, not evidence that upstream changes are needed
before SAX can proceed. Upstream replacement is future follow-up, not a blocker
for completing the five stages.

| Issue | Interim SAX approach | Future replacement |
| --- | --- | --- |
| https://github.com/gdsfactory/kfnetlist/issues/22 | Use `PlacedNetlist.cell`; no new SAX instance enum/schema | Reassess hierarchy identity when an agreed upstream representation is available |
| https://github.com/gdsfactory/kfnetlist/issues/23 | Independent copying in an isolated helper | Replace with a native copy API preserving type and data |
| https://github.com/gdsfactory/kfnetlist/issues/24 | Use existing cell exclusions and isolated additional selection logic only where needed | Adopt per-instance/path exclusions when supported |

No upstream direction-metadata request is planned. Connectivity remains undirected.

## Review-to-stage map

| Review findings | Resolution stage |
| --- | --- |
| 6, 7, 13: forward direction and rejection behavior | 1: remove backend and direction machinery |
| 1, 2, 3, 4: partials, info, placements, numerical settings | 2: input adaptation |
| 10, 11: placed serialization and copying | 2: preserve native types |
| 5, 8, 9, 12, 14, 15, 16: pruning, probes, flattening, lowering | 3: topology and transforms |
| 18: public loader/audit mismatch | 1: contract decision; 4: implementation and evidence |
| 17: introduced quality-check failures | Each affected stage; final audit in 5 |

## Original reproduction details

The table preserves the 12 SAX findings from the review against `origin/main`.
The six other findings are tracked explicitly in the stages and map above.
These are historical observations, not current execution results.

| Review # | Problem and reproduction | Intended fix and verification |
| --- | --- | --- |
| 1 | Partial-function arguments are discarded: `partial(leaf, v=3)` produces the default `1`. Positional partials are silently accepted. | Preserve keyword arguments as instance settings and reject positional partials. Test distinct partials of the same function and explicit evaluation overrides. Location: `src/sax/circuits.py::_replace_callable_instances`. |
| 2 | Legacy instance `info` no longer contributes model settings: `info={"v": 3}` produces `1`. | Restore the legacy `info`-to-settings merge and precedence; keep native `info` as metadata. Test conflicting legacy `settings`/`info` values. Location: `src/sax/native.py::_add_legacy_instance`. |
| 3 | Legacy placements are discarded: a placement-dependent model given `x=7` receives `x=0`. | Preserve supported legacy placements through adaptation and model evaluation, including existing normalization semantics. Location: `src/sax/native.py::from_legacy_flat`. |
| 4 | JAX arrays, NumPy arrays, and complex model settings fail construction because they are passed through native JSON-compatible storage. | Retain non-JSON settings in SAX-owned tables outside native topology serialization. Verify defaults, overrides, broadcasting, JIT, and real-objective gradients where applicable. Location: `src/sax/native.py::_add_legacy_instance`. |
| 5 | Disconnected instances are not pruned before model validation. An unused instance with an unknown model blocks a valid circuit; disconnected layout-only subcircuits demand unnecessary models. | Restore pruning before dependency/model validation while retaining instances needed by probes. Cover circuit construction and required-model discovery. Location: `src/sax/circuits.py::_circuit_native`, `_native_dag`. |
| 8 | Hierarchical probes silently overwrite parent ports: a probe named `tap` replaces an existing `tap_fwd`. | Reject generated-port collisions at every level through which a probe is exposed. Location: `src/sax/circuits.py::_circuit_native` (`extra_ports`). |
| 12 | Native hierarchy pruning passes `{cell: Netlist}` to a function expecting a single netlist, raising `AttributeError`. | Dispatch pruning over hierarchy entries, preserving types, root/order, and input isolation. Location: `src/sax/netlists.py::remove_unused_instances`. |
| 14 | Declared unconnected native ports disappear from the output; external-only nets follow the same dropping path. | Define supported numerical behavior or reject unsupported topology explicitly, rather than silently dropping ports/nets. Exact numerical semantics remain to be decided before implementation. Location: `src/sax/native.py::lower`. |
| 15 | Synthetic pruning nodes collide with valid instance names: a connected instance named `__port_0` is deleted. | Use graph node identities that cannot collide with user instance names. Add a regression verifying that connected instances survive and input is unchanged. Location: `src/sax/native.py::remove_unused_instances`. |
| 16 | Port validation precedes internal-port removal. When every external port is dropped, the backend raises `Need at least one array to stack`. | Validate effective top-level ports after internal-port handling and probe expansion, preserving the useful at-least-one-port diagnostic and valid probe-only circuits. Location: `src/sax/circuits.py::_circuit_native`. |
| 17 | Review checks found 46 introduced Ruff violations, 11 Pyright errors, four files needing formatting, and an EOF whitespace error. | Correct introduced lint/type/formatting failures without auto-fixing unrelated baseline issues. Rerun scoped checks and `git diff --check`; counts are historical review observations, not current verification. |
| 18 | The migration audit claims primary PIC loaders return native objects, but public loaders still return dictionaries. Its cited test exercises only `native.*` helpers. | Reconcile the audit, actual public loader contract, and tests. Do not silently change public return types solely to satisfy the prose. Locations: `specs/changes/kfnetlist-canonical.md`, `src/tests/test_native_kfnetlist.py::test_public_pic_loaders_produce_native`, `src/sax/utils.py`. |

## Historical verification baseline

At review time, the existing-environment non-notebook suite passed **412 tests
with 1 expected failure**, and `uv lock --check` passed. Additional reproductions
exposed the behavioral findings despite that suite passing. Lint, type,
formatting, and whitespace checks failed. Notebook and cross-platform checks were
not run for the review. No implementation fixes have been started.

The retired `todo.md` mixed proposals, completed work, and unsupported completion
claims. Its relevant identity, PIC, extraction, topology, and release-verification
items are now assigned to the stages above. Its old unchecked boxes do not imply
that already-implemented features must be rebuilt.
