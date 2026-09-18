# Native netlists and migration

SAX accepts `kfnetlist.Netlist`, `PlacedNetlist`, native dictionaries/JSON, and
legacy SAX dictionaries. Circuit construction uses native topology for all inputs.
Python 3.12 or newer and kfnetlist 0.3.x are required.

## Extract a layout without losing hierarchy identity

Use `include_placement=True` when extracting a hierarchical layout. A placed
instance preserves two separate identities: `component` identifies its factory,
and `cell` references its concrete child netlist. Placement coordinates need not
influence your analytical model.

This runnable example extracts two parameterizations of one factory. Both receive
the same analytical model with their own lengths:

```python
import gdsfactory as gf
import jax.numpy as jnp
import kfactory as kf
import numpy as np
from kfnetlist.extract import extract

import sax

gf.gpdk.PDK.activate()
top = gf.Component()
for name, length, y in (("a", 10.0, 0.0), ("b", 20.0, 30.0)):
    inst = top << gf.components.straight(length=length)
    inst.name = name
    inst.dmovey(y)
    top.add_port(name + "_in", port=inst.ports["o1"])
    top.add_port(name + "_out", port=inst.ports["o2"])

cells = extract(
    top,
    wrap_kdb_instance=lambda inst: kf.Instance(kcl=top.kcl, instance=inst),
    include_placement=True,
)


def straight(length=10.0, wl=1.55):
    phase = 2 * jnp.pi * 2.4 * length / wl
    return sax.reciprocal({("o1", "o2"): jnp.exp(1j * phase)})


factory = cells[top.name].instances["a"].component
model, info = sax.circuit(cells, {factory: straight}, top_level_name=top.name)
result = model(wl=1.55)
np.testing.assert_allclose(result["a_in", "a_out"], jnp.exp(2j * jnp.pi * 2.4 * 10 / 1.55))
np.testing.assert_allclose(result["b_in", "b_out"], jnp.exp(2j * jnp.pi * 2.4 * 20 / 1.55))
```

A model replaces the whole instance; models for its layout children are unnecessary.
Without that model, SAX follows `instance.cell` to the child definition. Distinct
parameterized children therefore remain distinct. Probing inside an analytically
replaced subtree raises an error; supply its child models to simulate that subtree.

## Model lookup and overrides

Lookup order is an exact `models[instance.cell]` override, a qualified
`models["library::component"]` binding, then a bare `models[instance.component]`
binding when the factory occurs in only one library in the supplied hierarchy.
If no model matches, SAX follows the explicit cell reference, then an exact
hierarchy key matching `component`. Errors list the instance path and attempted
identities. Names are never inferred by stripping numeric suffixes.

For old exports that have lost factory provenance, use an explicit alias only
when you know that two names represent the same factory:

```python
models = {"straight": straight, "straight2": straight}
```

Use `models[cell_name]` when one concrete child needs a specialized model. Direct
callable instances and keyword partials remain supported in legacy dictionaries;
separate partials retain independent defaults. Positional partial arguments are
rejected. Non-JSON NumPy/JAX/complex settings stay in SAX's numerical settings
tables; native serialization itself still requires JSON-compatible values.
Legacy instance `info` overrides `settings`; native `info` remains metadata.

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

## Installation verification

On 2026-09-18, dependency resolution passed for Python 3.12, 3.13, and 3.14 on
macOS ARM64, Linux x86-64/ARM64, and Windows x86-64. These are package-resolution
checks; runtime tests were run on macOS ARM64 with Python 3.12.14.

Intel macOS could not resolve SAX's required klujax dependency. Native Windows
ARM64 could not resolve kfnetlist. The kfnetlist 0.3.0 release provides wheels for
macOS ARM64/x86-64, Linux ARM64/x86-64, and Windows x86-64:
https://pypi.org/project/kfnetlist/0.3.0/ . A kfnetlist wheel alone does not establish
that every SAX dependency supports a platform.
