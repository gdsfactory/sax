# Netlists and circuit composition

## Input contract

A flat netlist has `instances`, with optional `connections`, `nets`, `ports`,
`placements`, and `settings`. A recursive netlist maps component names to flat
netlists. `sax.netlist` wraps flat input under `top_level_name` (default `top_level`)
or moves that named entry first. Otherwise the first recursive entry is the root.
This helper alone is not full validation.

```python
net = {
    "instances": {"a": "straight", "b": {"component": "straight",
                                           "settings": {"length": 20.0}}},
    "connections": {"a,out0": "b,in0"},
    "ports": {"in": "a,in0", "out": "b,out0"},
}
```

- An instance normalizes from a component string, callable, keyword-only partial,
  or dictionary with `component` and optional `settings`/`array`. Partial positional
  arguments are rejected. Instance `info` entries are merged into settings.
- Instance-port references use `instance,port`. `connections` is a mapping of
  endpoint pairs; `nets` is a list of `{p1, p2}` records. Circuit construction
  combines both. Net metadata is not a modeled transmission element.
- Route bundles with `links` normalize into nets. Unrecognized netlist fields are
  filtered by the schema; this is not lossless round-trip storage for layout data.
- Placements normalize x/dx, y/dy, rotation rounded modulo 360, and mirror. They
  reach models only through supported `placement` settings.
- A constructed top-level circuit must expose at least one port. One-port circuits
  are supported. Invalid recursive entries can be warned about and skipped during
  coercion; do not assume all malformed input fails immediately.

Evidence: [`saxtypes/netlist.py`](../src/sax/saxtypes/netlist.py) (`val_instance`,
`val_netlist`, `val_recnet`), [`netlists.py`](../src/sax/netlists.py) (`netlist`).
Tests: [`test_netlist.py`](../src/tests/test_netlist.py),
[`test_circuit.py`](../src/tests/test_circuit.py).

## Construction and hierarchy

`circuit(netlist, models=None, *, backend="default", return_type="SDict", ...)`
returns `(model, CircuitInfo)`. `CircuitInfo` contains the component dependency
DAG, constructed/resolved component models, and canonical backend name.

Construction adapts callable instances and settings, normalizes native topology,
validates/plans probe paths, and prunes instances disconnected from ports or probe
roots before validating model dependencies. Lowering expands arrays, handles
internal ports, and inserts probes; root-port validation uses the resulting ports.
Dependencies are constructed leaf-first; an explicitly supplied model can stand
in for a subcircuit. Missing leaf models raise `ValueError` with model diagnostics.
Hierarchy must be acyclic even when optical wiring has feedback. Explicit
acyclicity validation raises `ValueError` for recursive component definitions;
`test_hierarchy_validation.py` distinguishes these from valid optical feedback.

Known identity limitation ([#120](https://github.com/gdsfactory/sax/issues/120)):
`component` serves as both model key and recursive cell key. Counted gdsfactory
hierarchy variants can silently fall back to layout subcircuits instead of their
shared analytical model. Native kfnetlist input now avoids this: `sax.circuit`
accepts `Netlist`/`PlacedNetlist` hierarchies and resolves factory `component`
before descending into a distinct `PlacedInstance.cell`. Legacy dictionaries and
`.pic.yml` still use the legacy path and retain the limitation; see
[the canonical change proposal](changes/kfnetlist-canonical.md) for the
directed-connection blocker. `test_native_kfnetlist.py` and the smoke suite cover
factory substitution, distinct-cell fallback, overrides, missing-model
diagnostics, arrays, probes, settings, JIT, and gradients.

Array instances expand to `name<column.row>`. References can infer/patch array
extents. Evaluation uses the base name's settings for every element; this is not
independent per-element parameter addressing. Array extent inference and expansion do not mutate caller input.

`flatten_netlist` separately inlines children using `~` by default. Both
`connections` and `nets` endpoints are rewritten, including links within one child,
and net names/settings are retained. Use `sep="__"` when constructing a circuit
from the result: the legacy `~` separator is not a valid model-signature identifier.
Instance renaming updates both wiring formats without mutating input. Native
flattening preserves analytical model boundaries using explicit instance-cell maps
and existing cell exclusions. It can retain one modeled instance while expanding
another instance of the same cell. Hierarchy pruning dispatches per cell and uses
instance connectivity directly, without synthetic names that could collide.
Flattening is not a general hierarchical placement/settings composition API.
Regression tests: `src/tests/test_netlist_transforms.py`.

## Native model keys and API boundary

Cell-specific `models[cell]` overrides precede qualified factory bindings
`models["library::component"]`, followed by an exact bare factory binding when
that factory appears in only one library in the supplied hierarchy. Conflicting
bare bindings raise a diagnostic instead of silently selecting one library.
No suffix stripping is performed; `coupler2` is an independent factory name.
Explicit callable instances are instance-local bindings. Their adapter uses
collision-avoiding internal model keys, so same-named Python functions do not
overwrite one another or a separately supplied factory model. Keyword partial
arguments become per-instance settings; positional partials are rejected. Resolution then follows a concrete child `cell`, with
exact component-name fallback for legacy/plain hierarchy input. Missing-model
errors identify the instance path, library, factory, cell, and attempted keys.

Native keys remain unchanged in dependency information. Backend discovery uses
local identifier aliases so the existing lowered-table validators do not restrict
qualified identities. This is backend-table validation, not canonical topology.
Public `load_netlist`/`load_recursive_netlist` keep their dictionary returns;
`native.load_*` are explicit native-object loaders. Root/document compatibility
and user examples are covered by remediation stage 4.

## Evaluation and settings precedence

Backend discovery evaluates component models with defaults, so defaults must be
callable and describe the topology used later. Runtime changes should change
values, not port sets or sparse coordinate ordering.

For ordinary instances, effective parameter precedence is:

1. Model signature defaults.
2. Instance netlist settings whose keys occur in the model's defaults.
3. Global call-time settings propagated to matching keys throughout nested settings.
4. Explicit instance/nested call-time settings, which override globals.

Thus `model(wl=wl, a={"length": 30})` distributes `wl` while overriding `a`'s length.
Legacy settings (including `info` merged over explicit instance settings) are
retained in per-cell/per-instance Python tables during circuit preparation, outside
native JSON serialization. Arrays, complex values, and traced numerical settings
therefore remain usable. Native `info` stays metadata. Placement is passed to
models exposing a `placement` parameter when placed data is present; legacy
coordinates use x+dx, y+dy, rounded rotation modulo 360, and boolean mirror.

Unknown instance netlist keys are filtered, but unknown explicit call-time instance
keys may reach the model and raise an error. Do not assume root netlist `settings`
are automatically applied: `_flat_circuit` builds defaults from instances/models.

Return formats are `SDict` (default), `SCoo`, and `SDense`. The implementation
recognizes additional aliases/types, but unsupported return types raise `ValueError`.
`test_api_contracts.py` verifies rejection and all three documented formats.

Evidence: [`circuits.py`](../src/sax/circuits.py) (`circuit`, `_flat_circuit`,
`_forward_global_settings`, `resolve_array_instances`),
[`utils.py`](../src/sax/utils.py) (`get_settings`, `merge_dicts`, `update_settings`).
Baseline smoke checks exercised globals, explicit overrides, JIT, and gradients
for a two-instance circuit using KLU and FG; see [verification](verification.md).

## Port wiring and modes

Missing model ports raise `KeyError` during wiring expansion unless
`ignore_impossible_connections=True`, which skips unavailable port connections or
exposures. This is not a blanket suppression of invalid netlists.
Multimode connections join matching mode names; only the common modes are wired.
Connecting a single-mode port to a multimode port raises `ValueError`. External
multimode ports are exposed with `@mode` suffixes.

Multiply connected ports are supported by KLU only. Other backends reject any
repeated endpoint. An explicit splitter/junction model is different from simply
joining multiple endpoints; choose the intended physics rather than assuming
multi-link wiring inserts a power-conserving junction.

## Probes and internal ports

Top-level ports exposing already connected nodes are dropped with a warning by
default. `on_internal_port="ignore"` drops them silently; `"as_probes"` converts
them to measurement probes with a warning.

Explicit `probes={"mid": "a,out0"}` inserts `_ideal_probe` and exposes `mid_fwd`
and `mid_bwd`. **Forward means traveling into the targeted instance port**, not
left-to-right netlist order. Both tap ports are currently created for connected,
boundary, and truly unconnected targets. Boundary probes preserve the original
external port through the probe. The docstring now explicitly describes both taps
for all target types.

Dot-separated paths such as `sub.a,out0` target hierarchical instances and expose
taps through parents. Invalid hierarchy paths and generated name collisions raise
`ValueError`. Expansion operates on component definitions: shared components may
acquire additional internal probe structure. Probes are nonphysical copying
instruments, not energy-conserving splitters.

Evidence: `expand_probes`, `_expand_probes_recursive`, `extract_port_probes` in
[`netlists.py`](../src/sax/netlists.py),
[`models/probes.py`](../src/sax/models/probes.py).
Tests: [`test_probes.py`](../src/tests/test_probes.py), notably forward direction,
non-perturbation, boundary/unconnected cases, hierarchy, conflicts, and internal
port policies. These do not establish arbitrary multiply connected probe behavior.

## Native topology validation

Native nets with more than two instance endpoints are rejected: use an explicit
junction model or specify pairwise edges with the intended KLU semantics. Explicit
pairwise multi-links retain KLU's existing behavior and FG/additive restrictions.
External-only nets, unattached declared external ports, external aliases targeting
one instance port, and contradictory external attachments are rejected explicitly.
A singleton internal endpoint is an unconnected model port and adds no connection.
Unknown instance references and out-of-range array references are rejected.

Probe names are checked against caller ports/instances before pruning, and against
ports at every ancestor when exposed through hierarchy. Probes into model-replaced
subtrees are rejected. Portless root/child circuits may be retained by explicit
probe paths; a root with no effective ports after transformations is rejected with
the at-least-one-port diagnostic. Array index dots are not hierarchy separators,
and runtime array settings remain keyed by the base instance name.

Evidence: `src/tests/test_native_topology.py`, `test_probes.py`,
`test_backend_restrictions.py`, and `test_hierarchy_validation.py`.
