# Netlists and circuit composition

## Input contract

A netlist is a plain `kfnetlist.Netlist`. A hierarchy is a
`kfnetlist.HierarchicalNetlist` mapping document-local IDs to netlists; a reference
instance uses `netlist_id` to select a child. `sax.circuit` accepts either object.
A plain netlist is simulated under `top_level_name` (default `top_level`); a
hierarchy uses an explicit root, then `top_level` if present, then its first entry.
Children remain mutable; kfnetlist validates the document before circuit use.
SAX accepts no legacy dictionary or JSON netlist inputs in this breaking pass.

```python
import sax
from kfnetlist import Netlist, NetlistPort, PortRef

net = Netlist()
net.create_inst("a", "pdk", "straight", {"length": 20.0})
net.create_port("in")
net.create_port("out")
net.create_net(NetlistPort(name="in"), PortRef("a", "in"))
net.create_net(PortRef("a", "out"), NetlistPort(name="out"))
model, info = sax.circuit(net, {"straight": straight_model})
```

An instance records a library, component, settings, and optionally an explicit
child `netlist_id`. SAX does not infer a child from the component name. kfnetlist
owns netlist validation, serialization, and generic transforms. SAX keeps
instances, nets, and ports as kfnetlist objects through compilation and mode
expansion. Backends derive numerical endpoint indices from the compiled netlist.
The constructed root must expose at least one port.

Evidence: [`saxtypes/__init__.py`](../src/sax/saxtypes/__init__.py),
[`circuits.py`](../src/sax/circuits.py), and
[`_circuit_compiler.py`](../src/sax/_circuit_compiler.py).
Tests: [`test_explicit_netlist_hierarchy.py`](../src/tests/test_explicit_netlist_hierarchy.py).

## Construction and hierarchy

`circuit(netlist, models=None, *, backend="default", return_type="SDict", ...)`
returns `(model, CircuitInfo)`. `CircuitInfo` contains the component dependency
DAG, constructed/resolved component models, and canonical backend name.

Construction validates the hierarchy, plans probes, and prunes instances
disconnected from ports or probe roots before validating model dependencies.
Compilation asks kfnetlist to expand arrays, then handles internal ports and
inserts probes.
Dependencies are constructed leaf-first; an explicitly supplied model can stand
in for a referenced child. Missing leaf models raise `ValueError` with model
diagnostics. Hierarchy must be acyclic even when optical wiring has feedback.

SAX uses `component` as the factory/model key and `netlist_id` as the explicit
child definition key. It does not infer hierarchy from factory names or accept
placed instances. Array instances expand to `name<column.row>`; evaluation uses
the base instance's settings for every element.

Generic flattening is `HierarchicalNetlist.flatten(root, separator=...)` in
kfnetlist. `Netlist.prune_unconnected(keep_instances=...)` retains components
reachable from declared ports or probe roots. `Netlist.expand_arrays()` creates
scalar instances and rewrites array references. SAX supplies probe roots and
applies its solver-specific pairwise wiring rules after those transformations.

## Native model keys and API boundary

Reference-specific `models[netlist_id]` overrides precede qualified factory bindings
`models["library::component"]`, followed by an exact bare factory binding when
that factory appears in only one library in the supplied hierarchy. Conflicting
bare bindings raise a diagnostic instead of silently selecting one library.
No suffix stripping is performed; `coupler2` is an independent factory name.
SAX no longer binds direct Python callables inside netlist instances: model
functions are supplied through the `models` mapping. Resolution then follows an
explicit child `netlist_id`. Leaf instances have no child fallback.
Missing-model errors identify the instance path, library, factory, reference,
and attempted keys.

Native keys remain unchanged in dependency information. Model bindings are held
separately from kfnetlist instances during backend discovery, so qualified model
names do not alter the circuit topology.
SAX no longer exposes kfnetlist parser, YAML loader, or legacy PIC adapter
functions. Construct or deserialize netlist objects with kfnetlist, then pass
them to `sax.circuit`. Root selection is described above and in
`docs/native-netlists.md`.

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
Instance settings must be JSON-compatible kfnetlist settings. Native `info` is
metadata, not a model parameter source. Plain netlists carry no placement.

Unknown instance netlist keys are filtered, but unknown explicit call-time instance
keys may reach the model and raise an error. Do not assume root netlist `settings`
are automatically applied: `_flat_circuit` builds defaults from instances/models.

Return formats are `SDict` (default), `SCoo`, and `SDense`. The implementation
recognizes additional aliases/types, but unsupported return types raise `ValueError`.
`test_api_contracts.py` verifies rejection and all three documented formats.

Evidence: [`circuits.py`](../src/sax/circuits.py) (`circuit`, `_flat_circuit`,
`_forward_global_settings`) and
[`_circuit_compiler.py`](../src/sax/_circuit_compiler.py) (`lower_bindings`),
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

Evidence: `expand_probes` in [`_circuit_compiler.py`](../src/sax/_circuit_compiler.py),
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

Current object-path evidence: `test_explicit_netlist_hierarchy.py`,
`test_kfnetlist_compilation.py`, `test_native_topology.py`, and `test_probes.py`.

## Root selection and removed adapters

`top_level_name` selects the root of a `HierarchicalNetlist`. Without it SAX
uses `top_level` if present, then the first entry. An unknown root is an error.
For a plain `Netlist`, the selected name defaults to `top_level`. Legacy SAX
netlist dictionaries, PIC documents, JSON text, and placed netlists are rejected.
Use kfnetlist constructors or deserializers before calling SAX.
