# Native netlists and migration

SAX's development path uses plain `kfnetlist.Netlist` objects. A document is a
mapping from IDs to netlists. An instance's `component` names its factory/model;
`netlist_id` names a child definition when one is included. The two IDs can differ.
SAX requires Python 3.12+ and currently resolves kfnetlist from its Git `main`
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

model, info = sax.circuit(
    {"top_level": top, "arm_3": child},
    {"waveguide": lambda gain=1.0: {("in", "out"): gain}},
)
result = model()
```

A model for `make_arm` replaces the whole referenced instance; otherwise SAX
traverses `arm_3`. `models["arm_3"]` can override that particular child. The
reference must point to a netlist in the document; kfnetlist validates this and
rejects cycles. Plain `Netlist` objects carry no placement. The upstream
extraction producer must emit explicit references for SAX to traverse an
extracted hierarchy in this development pass.

## Model lookup

Lookup order is an exact `models[instance.netlist_id]` override when present,
then `models["library::component"]`, then bare `models[component]` when
unambiguous. If no model matches, SAX follows `netlist_id`. Leaf instances require
a model. SAX does not infer children from factory names.

## PIC loaders and roots

Public `sax.load_netlist` and `sax.load_recursive_netlist` keep their dictionary
returns. Explicit native helpers are available through `from sax import native`:

| Helper | Result |
| --- | --- |
| `native.load_pic_yaml(text_or_mapping, top_level_name=None)` | `(cells, root)`, including all modules |
| `native.load_native_netlist(text_path_or_mapping)` | One native netlist; rejects multi-cell documents |
| `native.load_native_recursive_netlist(path, ext=".pic.yml")` | `(cells, root)`, discovering matching files recursively |

An explicit `top_level_name` wins over a PIC document's `toplevel`. Otherwise SAX
uses `top_level` if present, then the first hierarchy entry. An explicitly missing
root is an error. Recursive file loaders normalize standalone filename stems,
preserve the root first, and reject duplicate names. Module documents retain
all child definitions and their document root.

Instance settings, connections, `{p1,p2}` nets, route links, and arrays are supported.
Legacy array indices are zero-based (`a<0.0>`); native `ia`/`ib` are one-based.
Native loaders reject `${...}` expressions and unsupported module-level
settings/info/metadata explicitly. Keep the original document with the public
loader if you need that metadata. Legacy flat root settings keep their previous
ignored behavior; put model defaults on instances and supply evaluation overrides
to the returned circuit function.

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

Native copy/rename/prune/flatten operations preserve caller input. Use
`native.flatten_netlist(cells, root, models=models)` when flattening must retain
analytical model boundaries. SAX currently implements copying and per-instance
flatten selection through small helpers pending equivalent upstream support.

## Development dependency

`pyproject.toml` resolves kfnetlist from its `main` branch. `uv.lock` records the
specific commit used for a reproducible development environment. Refresh the
lockfile when intentionally testing a newer `main` revision. The earlier wheel
platform checks apply to the published 0.3.0 package, not to this Git build.
