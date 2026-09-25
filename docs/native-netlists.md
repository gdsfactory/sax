# Native netlists and migration

SAX's development path uses plain `kfnetlist.Netlist` objects. A document is a
`kfnetlist.HierarchicalNetlist`, an ordered mapping from IDs to mutable netlists.
An instance's `component` names its factory/model;
`netlist_id` names a child definition when one is included. The two IDs can differ.
SAX requires Python 3.12+ and currently resolves kfnetlist from its Git `sax`
branch through `uv.lock`. This contract is under development and is not tied to
a published kfnetlist version.

## Build and simulate a referenced hierarchy

```python
import kfnetlist as kfn
import sax

child = kfn.Netlist()
child.create_inst("wg", "pdk", "waveguide", {"gain": 3.0})
child.create_port("in")
child.create_port("out")
child.create_net(kfn.NetlistPort(name="in"), kfn.PortRef("wg", "in"))
child.create_net(kfn.PortRef("wg", "out"), kfn.NetlistPort(name="out"))

top = kfn.Netlist()
top.create_inst("arm", "pdk", "make_arm", netlist_id="arm_3")
top.create_port("in")
top.create_port("out")
top.create_net(kfn.NetlistPort(name="in"), kfn.PortRef("arm", "in"))
top.create_net(kfn.PortRef("arm", "out"), kfn.NetlistPort(name="out"))

document = kfn.HierarchicalNetlist({"top_level": top, "arm_3": child})
model, info = sax.circuit(
    document,
    {"waveguide": lambda gain=1.0: {("in", "out"): gain}},
)
result = model()
```

A model for `make_arm` replaces the whole referenced instance; otherwise SAX
traverses `arm_3`. `models["arm_3"]` can override that particular child. The
reference must point to a netlist in the document. Construction validates
references and rejects cycles. Child netlists remain mutable; call
`document.validate()` after edits, or rely on hierarchy operations to validate
before use. Plain `Netlist` objects carry no placement. On the kfnetlist `sax`
branch, `extract(..., include_placement=False)` emits `netlist_id` references
to the child netlists in its returned document. Wrap that mapping in
`HierarchicalNetlist` before passing it to SAX. Bind models using the
extracted `component` factory IDs; gdsfactory may qualify them beyond a short
function name such as `straight`.

## Model lookup

Lookup order is an exact `models[instance.netlist_id]` override when present,
then `models["library::component"]`, then bare `models[component]` when
unambiguous. If no model matches, SAX follows `netlist_id`. Leaf instances require
a model. SAX does not infer children from factory names.

## Roots and serialization

Construct or deserialize netlists with kfnetlist and pass `Netlist` or
`HierarchicalNetlist` objects directly to `sax.circuit`. A plain netlist uses
`top_level_name` or `top_level`. A hierarchy uses an explicitly requested root,
then `top_level` if present, then its first entry. A missing requested root is
an error. SAX no longer parses legacy dictionaries, PIC YAML, or JSON strings
as circuit inputs; its former `native`, `netlists`, parser, and YAML loader
convenience APIs have been removed in this development pass.

## Backend and topology changes

The `forward` backend and its public lower-level functions have been removed.
Use KLU (default), FG, or additive where applicable. Physical forward/backward
probe measurements remain available. Connectivity is undirected; component
S-matrices may still be asymmetric.

Native external ports must connect to instances. Unattached ports, external-only
nets, aliases for the same instance endpoint, and nets joining more than two
instance ports raise explicit errors. Represent junction physics with a model.
Explicit pairwise multilinks retain KLU's existing behavior; FG and additive
retain their restrictions on multiply connected ports.

Generic netlist operations, including document flattening, belong to
kfnetlist. SAX calls `Netlist.prune_unconnected()` and
`Netlist.expand_arrays()` before applying model resolution and probe planning.
Numerical backends derive their solver indices from the resulting kfnetlist
topology.

## Development dependency

`pyproject.toml` resolves kfnetlist from its `sax` branch. `uv.lock` records the
specific commit used for a reproducible development environment. Refresh the
lockfile when intentionally testing a newer `sax` revision. The earlier wheel
platform checks apply to the published 0.3.0 package, not to this Git build.
