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

Construction extracts direct callable instances, normalizes the recursive netlist,
patches arrays, handles internal ports, expands probes, resolves arrays, filters
portless sub-netlists, and prunes instances disconnected from external ports.
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
independent per-element parameter addressing. Array patching can mutate the input
before subsequent copying: do not assume the whole constructor is side-effect-free.

`flatten_netlist` separately inlines children using `~` by default. Both
`connections` and `nets` endpoints are rewritten, including links within one child,
and net names/settings are retained. Use `sep="__"` when constructing a circuit
from the result: the legacy `~` separator is not a valid model-signature identifier.
Instance renaming updates both wiring formats without mutating input. Flattening
is not a general hierarchical placement/settings composition API.
Regression tests: `src/tests/test_netlist_transforms.py`.

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
