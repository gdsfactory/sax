"""Native kfnetlist integration.

kfnetlist is SAX's canonical netlist format. This module adapts supported
legacy inputs *into* native :class:`kfnetlist.Netlist` objects and lowers native
topology to the flat tables consumed by the numerical backends. Native input is
never converted into SAX's legacy dictionary netlist schema.

Two hierarchy identities are kept distinct:

* ``component`` is the factory / analytical-model name shared by every
  parameterization of a cell.
* ``PlacedInstance.cell`` is the concrete instantiated cell and is used to look
  up child netlists when no analytical model replaces the instance.

Model resolution precedence (see ``specs/changes/kfnetlist-canonical.md``):

1. explicit cell-specific model (``models[cell]``),
2. exact factory model (``models[component]``),
3. recurse via the instantiated cell reference (``cells[cell]``),
4. recurse via a cell whose name equals the factory name,
5. otherwise the instance is missing a model.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from kfnetlist import (
    Netlist,
    NetlistPort,
    PlacedNetlist,
    PortArrayRef,
    PortRef,
)

if TYPE_CHECKING:
    pass

__all__ = [
    "NativeHierarchy",
    "copy_netlist",
    "from_legacy_flat",
    "from_legacy_recursive",
    "is_native",
    "is_native_hierarchy",
    "lower",
    "lower_bindings",
    "placements",
    "resolve",
    "to_hierarchy",
]

NativeHierarchy = dict[str, Netlist]
"""Mapping of cell name to native kfnetlist netlist (SAX's canonical hierarchy)."""


def is_native(obj: object) -> bool:
    """Return whether *obj* is a native kfnetlist netlist object."""
    return isinstance(obj, Netlist)


def is_native_hierarchy(obj: object) -> bool:
    """Return whether *obj* is a native ``{cell name: Netlist}`` mapping."""
    if not isinstance(obj, Mapping) or not obj:
        return False
    return all(isinstance(v, Netlist) for v in obj.values())


# ---------------------------------------------------------------------------
# Adapters into native topology
# ---------------------------------------------------------------------------


def _split_legacy_endpoint(endpoint: str) -> tuple[str, str, int | None, int | None]:
    """Split ``instance,port`` into ``(base instance, port, col, row)``."""
    instance, _, port = endpoint.partition(",")
    if not port:
        return endpoint, "", None, None
    if "<" in instance:
        base, _, rest = instance.partition("<")
        indices, _, _ = rest.partition(">")
        first, _, second = indices.partition(".")
        col = int(first)
        row = int(second) if second else 0
        return base, port, col, row
    return instance, port, None, None


def _scan_legacy_arrays(flat: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    """Infer native array sizes from legacy endpoints that index instances."""
    sizes: dict[str, tuple[int, int]] = {}
    declared = flat.get("instances") or {}
    for name, inst in declared.items():
        array = inst.get("array") if isinstance(inst, Mapping) else None
        if array:
            sizes[str(name)] = (
                max(int(array.get("columns", 1)), 1),
                max(int(array.get("rows", 1)), 1),
            )

    endpoints: list[str] = []
    endpoints.extend(str(k) for k in (flat.get("connections") or {}))
    endpoints.extend(str(v) for v in (flat.get("connections") or {}).values())
    for net in flat.get("nets") or []:
        endpoints.append(str(net["p1"]))
        endpoints.append(str(net["p2"]))
    endpoints.extend(str(v) for v in (flat.get("ports") or {}).values())

    for endpoint in endpoints:
        base, _, col, row = _split_legacy_endpoint(endpoint)
        if col is None:
            continue
        na, nb = sizes.get(base, (1, 1))
        sizes[base] = (max(na, col + 1), max(nb, (row or 0) + 1))
    return sizes


def _add_legacy_instance(nl: Netlist, name: str, inst: Any, sizes: Mapping[str, tuple[int, int]]) -> None:
    if isinstance(inst, str):
        component, settings, info = inst, {}, {}
    elif isinstance(inst, Mapping):
        if "component" not in inst:
            msg = f"Instance {name!r} is missing a 'component' key: {inst!r}."
            raise ValueError(msg)
        component = str(inst["component"])
        settings = dict(inst.get("settings") or {})
        info = dict(inst.get("info") or {})
    else:
        msg = f"Unsupported legacy instance {name!r}: {inst!r}."
        raise TypeError(msg)
    na, nb = sizes.get(name, (1, 1))
    nl.create_inst(
        name,
        kcl="",
        component=component,
        settings=settings,
        na=na,
        nb=nb,
        info=info,
    )


def _native_member(endpoint: str) -> NetlistPort | PortRef | PortArrayRef:
    instance, port, col, row = _split_legacy_endpoint(endpoint)
    if not port:
        return NetlistPort(name=endpoint)
    if col is not None:
        return PortArrayRef(instance=instance, port=port, ia=col + 1, ib=(row or 0) + 1)
    return PortRef(instance=instance, port=port)


def from_legacy_flat(flat: Mapping[str, Any]) -> Netlist:
    """Adapt one legacy SAX flat netlist into a native kfnetlist netlist."""
    nl = Netlist()
    for name in (flat.get("ports") or {}):
        nl.create_port(str(name))
    sizes = _scan_legacy_arrays(flat)
    for name, inst in (flat.get("instances") or {}).items():
        _add_legacy_instance(nl, str(name), inst, sizes)
    for src, tgt in (flat.get("connections") or {}).items():
        nl.create_net(_native_member(str(src)), _native_member(str(tgt)))
    for net in flat.get("nets") or []:
        nl.create_net(_native_member(str(net["p1"])), _native_member(str(net["p2"])))
    for bundle in (flat.get("routes") or {}).values():
        for src, tgt in (bundle.get("links") or {}).items():
            nl.create_net(_native_member(str(src)), _native_member(str(tgt)))
    for name, endpoint in (flat.get("ports") or {}).items():
        nl.create_net(NetlistPort(name=str(name)), _native_member(str(endpoint)))
    return nl


def from_legacy_recursive(recnet: Mapping[str, Any]) -> NativeHierarchy:
    """Adapt a legacy recursive netlist into a native hierarchy."""
    return {str(name): from_legacy_flat(flat) for name, flat in recnet.items()}


def _looks_native(data: Mapping[str, Any]) -> bool:
    instances = data.get("instances")
    if not isinstance(instances, Mapping) or not instances:
        return False
    return any(
        isinstance(inst, Mapping) and "kcl" in inst and "component" in inst
        for inst in instances.values()
    )


def legacy_orientation(flat: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Directed endpoint pairs declared by a legacy flat netlist.

    kfnetlist nets are undirected; this preserves the signal direction that the
    legacy ``connections``/``nets`` encoding carried so directed backends (the
    ``forward`` backend) remain correct.
    """
    pairs: list[tuple[str, str]] = []
    for src, tgt in (flat.get("connections") or {}).items():
        pairs.append((str(src), str(tgt)))
    for net in flat.get("nets") or []:
        pairs.append((str(net["p1"]), str(net["p2"])))
    return pairs


def to_hierarchy(
    netlist: object,
    *,
    top_level_name: str = "top_level",
) -> tuple[NativeHierarchy, str, dict[str, list[tuple[str, str]]]]:
    """Normalize supported input into ``(cells, root, orientations)``.

    ``orientations`` records directed endpoint pairs that native unordered nets
    cannot represent; it is populated only for legacy/dict input.
    """
    if is_native(netlist):
        return {top_level_name: netlist}, top_level_name, {}

    if is_native_hierarchy(netlist):
        cells = dict(netlist)
        root = top_level_name if top_level_name in cells else next(iter(cells))
        return cells, root, {}

    if isinstance(netlist, str):
        try:
            decoded = json.loads(netlist)
        except json.JSONDecodeError:
            msg = "Native string input must be JSON-encoded kfnetlist data."
            raise TypeError(msg) from None
        return to_hierarchy(decoded, top_level_name=top_level_name)

    if isinstance(netlist, Mapping):
        data = dict(netlist)
        if "instances" in data:
            if _looks_native(data):
                return {top_level_name: Netlist.from_dict(data)}, top_level_name, {}
            return (
                {top_level_name: from_legacy_flat(data)},
                top_level_name,
                {top_level_name: legacy_orientation(data)},
            )

        cells: NativeHierarchy = {}
        orientations: dict[str, list[tuple[str, str]]] = {}
        for name, flat in data.items():
            key = str(name)
            if isinstance(flat, Netlist):
                cells[key] = flat
            elif isinstance(flat, Mapping) and _looks_native(flat):
                cells[key] = Netlist.from_dict(dict(flat))
            elif isinstance(flat, Mapping):
                cells[key] = from_legacy_flat(flat)
                orientations[key] = legacy_orientation(flat)
            else:
                msg = f"Unsupported netlist entry {name!r}: {type(flat)}."
                raise TypeError(msg)
        root = top_level_name if top_level_name in cells else next(iter(cells))
        return cells, root, orientations

    msg = f"Cannot interpret {type(netlist)} as a netlist."
    raise TypeError(msg)


# ---------------------------------------------------------------------------
# Lowering to backend topology tables
# ---------------------------------------------------------------------------


def _expanded_name(name: str, i: int, j: int, na: int, nb: int) -> str:
    if na <= 1 and nb <= 1:
        return name
    return f"{name}<{i}.{j}>"


def lower(
    nl: Netlist,
    orientation: list[tuple[str, str]] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]], dict[str, str]]:
    """Lower a native netlist to ``(instances, nets, ports)`` tables.

    These flat tables are the compiled input to the numerical backends, not a
    canonical netlist representation. Array instances are expanded to
    ``name<column.row>`` entries (zero-based), matching SAX instance naming.

    ``orientation`` supplies directed endpoint pairs for nets whose direction
    native unordered nets cannot carry (used by the ``forward`` backend).
    """
    directed: dict[frozenset[str], tuple[str, str]] = {
        frozenset((a, b)): (a, b) for a, b in (orientation or [])
    }
    instances: dict[str, dict[str, Any]] = {}
    array_sizes: dict[str, tuple[int, int]] = {}
    for name, inst in nl.instances.items():
        arr = inst.array
        na = max(int(arr.na), 1) if arr is not None else 1
        nb = max(int(arr.nb), 1) if arr is not None else 1
        array_sizes[name] = (na, nb)
        for i in range(na):
            for j in range(nb):
                entry: dict[str, Any] = {"component": inst.component}
                if inst.settings:
                    entry["settings"] = dict(inst.settings)
                instances[_expanded_name(name, i, j, na, nb)] = entry

    def member_endpoint(member: Any) -> str | None:
        if isinstance(member, NetlistPort):
            return None
        if isinstance(member, PortArrayRef):
            key = _expanded_name(
                member.instance,
                member.ia - 1,
                member.ib - 1,
                *array_sizes.get(member.instance, (1, 1)),
            )
            return f"{key},{member.port}"
        assert isinstance(member, PortRef)
        key = _expanded_name(
            member.instance, 0, 0, *array_sizes.get(member.instance, (1, 1))
        )
        return f"{key},{member.port}"

    declared = {p.name for p in nl.ports}
    ports: dict[str, str] = {}
    nets: list[dict[str, str]] = []
    for net in nl.nets:
        members = list(net)
        externals = [m for m in members if isinstance(m, NetlistPort)]
        internal = [m for m in members if not isinstance(m, NetlistPort)]
        endpoints = [member_endpoint(m) for m in internal]
        endpoints = [e for e in endpoints if e is not None]
        for external in externals:
            if external.name in declared and endpoints:
                ports[external.name] = endpoints[0]
        for a, b in zip(endpoints, endpoints[1:]):
            hint = directed.get(frozenset((a, b)))
            if hint is not None:
                nets.append({"p1": hint[0], "p2": hint[1]})
            else:
                nets.append({"p1": a, "p2": b})
    return instances, nets, ports


def resolve(
    inst: Any,
    models: Mapping[str, Any] | None,
    cells: Mapping[str, Any] | None,
) -> str | None:
    """Resolve one native instance to a model/cell key.

    Returns ``None`` when the instance cannot be resolved.
    """
    models = models or {}
    cells = cells or {}
    component = inst.component
    cell = getattr(inst, "cell", None)
    if cell is not None and cell in models:
        return cell
    if component in models:
        return component
    if cell is not None and cell in cells:
        return cell
    if component in cells:
        return component
    return None


def copy_netlist(nl: Netlist) -> Netlist:
    """Return an independent native copy of *nl*."""
    return Netlist.from_dict(nl.to_dict())


def lower_bindings(
    nl: Netlist,
    models: Mapping[str, Any],
    cells: Mapping[str, Any],
    orientation: list[tuple[str, str]] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]], dict[str, str], dict[str, str | None]]:
    """Lower *nl* and resolve every expanded instance to a model/cell key."""
    instances, nets, ports = lower(nl, orientation)
    bindings: dict[str, str | None] = {}
    for name, inst in nl.instances.items():
        arr = inst.array
        na = max(int(arr.na), 1) if arr is not None else 1
        nb = max(int(arr.nb), 1) if arr is not None else 1
        key = resolve(inst, models, cells)
        for i in range(na):
            for j in range(nb):
                bindings[_expanded_name(name, i, j, na, nb)] = key
    return instances, nets, ports, bindings


def placements(nl: Netlist) -> dict[str, dict[str, Any]]:
    """Extract legacy-shaped placement settings from a placed netlist."""
    result: dict[str, dict[str, Any]] = {}
    for name, inst in nl.instances.items():
        placement = getattr(inst, "placement", None)
        if placement is None:
            continue
        result[name] = {
            "x": float(placement.x),
            "y": float(placement.y),
            "rotation": float(placement.orientation),
            "mirror": bool(placement.mirror),
        }
    return result


# ---------------------------------------------------------------------------
# Probe and internal-port handling on lowered topology tables
# ---------------------------------------------------------------------------


def _endpoint_instance(endpoint: str) -> str:
    return endpoint.split(",")[0]


def handle_internal_ports(
    instances: Mapping[str, Any],
    nets: list[dict[str, str]],
    ports: dict[str, str],
    on_internal_port: str,
) -> tuple[dict[str, str], dict[str, str]]:
    """Drop or convert external ports that target internal connection nodes."""
    import warnings

    internal: set[str] = set()
    for net in nets:
        internal.add(net["p1"])
        internal.add(net["p2"])
    probes: dict[str, str] = {}
    kept: dict[str, str] = {}
    for name, endpoint in ports.items():
        if endpoint not in internal:
            kept[name] = endpoint
            continue
        if on_internal_port == "as_probes":
            warnings.warn(
                f"Port '{name}' maps to internal node '{endpoint}' which is "
                "already part of a connection. It will be interpreted as a probe "
                f"(creating '{name}_fwd' and '{name}_bwd' ports).",
                stacklevel=2,
            )
            probes[name] = endpoint
        elif on_internal_port == "warn":
            warnings.warn(
                f"Port '{name}' maps to internal node '{endpoint}' which is "
                "already part of a connection. It will be dropped. Use the "
                "probes= argument of circuit() to explicitly create measurement "
                "probes.",
                stacklevel=2,
            )
    return kept, probes


def expand_probes_tables(
    instances: dict[str, dict[str, Any]],
    nets: list[dict[str, str]],
    ports: dict[str, str],
    probes: Mapping[str, str],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]], dict[str, str]]:
    """Insert ideal probes into lowered topology tables.

    ``_fwd`` measures the wave travelling into the targeted instance port.
    """
    if not probes:
        return instances, nets, ports
    instances = dict(instances)
    ports = dict(ports)
    nets = list(nets)
    for probe_name, target in probes.items():
        fwd_port = f"{probe_name}_fwd"
        bwd_port = f"{probe_name}_bwd"
        if fwd_port in ports or bwd_port in ports:
            msg = (
                f"Probe '{probe_name}' would create ports '{fwd_port}'/"
                f"'{bwd_port}' which conflict with existing ports."
            )
            raise ValueError(msg)
        probe_instance = f"_probe_{probe_name}"
        if probe_instance in instances:
            msg = (
                f"Probe instance name '{probe_instance}' conflicts with an "
                "existing instance."
            )
            raise ValueError(msg)

        in_side: str | None = None
        for i, net in enumerate(nets):
            if net["p1"] == target:
                in_side = net["p2"]
                nets.pop(i)
                break
            if net["p2"] == target:
                in_side = net["p1"]
                nets.pop(i)
                break
        if in_side is None:
            # Boundary or unconnected: route the existing external port through.
            for pname, endpoint in ports.items():
                if endpoint == target:
                    in_side = f"{probe_instance},in"
                    ports[pname] = in_side
                    break

        instances[probe_instance] = {"component": "_ideal_probe"}
        if in_side is not None and in_side != f"{probe_instance},in":
            nets.append({"p1": in_side, "p2": f"{probe_instance},in"})
        nets.append({"p1": f"{probe_instance},out", "p2": target})
        ports[fwd_port] = f"{probe_instance},tap_fwd"
        ports[bwd_port] = f"{probe_instance},tap_bwd"
    return instances, nets, ports


def plan_hierarchical_probes(
    cells: Mapping[str, Netlist],
    root: str,
    models: Mapping[str, Any],
    probes: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, dict[str, str]], dict[str, list[tuple[str, str]]]]:
    """Split probes into root probes, per-cell probes, and bubble-up paths."""
    top: dict[str, str] = {}
    per_cell: dict[str, dict[str, str]] = {}
    paths: dict[str, list[tuple[str, str]]] = {}
    for probe_name, target in probes.items():
        parts = target.split(".")
        if len(parts) == 1:
            top[probe_name] = target
            continue
        current = root
        path: list[tuple[str, str]] = []
        for instance_name in parts[:-1]:
            inst = cells[current].instances.get(instance_name)
            if inst is None:
                msg = (
                    f"Hierarchical probe '{probe_name}': instance "
                    f"'{instance_name}' not found in component '{current}'."
                )
                raise ValueError(msg)
            key = resolve(inst, models, cells)
            if key is None or key not in cells or key in models:
                msg = (
                    f"Hierarchical probe '{probe_name}': component "
                    f"'{inst.component}' (used by instance '{instance_name}' "
                    f"in '{current}') is not defined in the recursive netlist. "
                    "Only sub-circuits (not primitives) can be probed."
                )
                raise ValueError(msg)
            path.append((instance_name, current))
            current = key
        per_cell.setdefault(current, {})[probe_name] = parts[-1]
        paths[probe_name] = path
    return top, per_cell, paths


def load_pic_yaml(
    content: str | Mapping[str, Any],
) -> tuple[NativeHierarchy, str]:
    """Load a PIC document (``.pic.yml`` mapping) into a native hierarchy.

    Accepts the legacy flat/recursive SAX mapping and the ``modules``/``toplevel``
    document shape. Returns ``(cells, root)``.
    """
    import yaml

    if isinstance(content, str):
        data = yaml.safe_load(content)
    else:
        data = dict(content)
    if not isinstance(data, Mapping):
        msg = f"PIC document must be a mapping, got {type(data)}."
        raise TypeError(msg)

    if "modules" in data:
        modules = data["modules"]
        toplevel = data.get("toplevel")
        cells = {str(name): from_legacy_flat(module) for name, module in modules.items()}
        if toplevel is None:
            toplevel = next(iter(cells))
        if toplevel not in cells:
            msg = f"Unknown toplevel module {toplevel!r}."
            raise ValueError(msg)
        return cells, toplevel

    if "instances" in data:
        return {"top_level": from_legacy_flat(data)}, "top_level"

    cells = {str(name): from_legacy_flat(flat) for name, flat in data.items()}
    return cells, next(iter(cells))


def load_native_netlist(content_or_path: object) -> Netlist:
    """Load a single native netlist from YAML content, a path, or a mapping."""
    from pathlib import Path

    if isinstance(content_or_path, Netlist):
        return content_or_path
    if isinstance(content_or_path, Mapping):
        cells, root = load_pic_yaml(content_or_path)
        return cells[root]
    if hasattr(content_or_path, "read"):
        content: object = content_or_path.read()  # type: ignore[union-attr]
    elif isinstance(content_or_path, (str, Path)) and "\n" not in str(content_or_path):
        path = Path(str(content_or_path))
        content = path.read_text() if path.exists() else str(content_or_path)
    else:
        content = content_or_path
    cells, root = load_pic_yaml(content)  # type: ignore[arg-type]
    return cells[root]


def load_native_recursive_netlist(
    top_level_path: object,
    ext: str = ".pic.yml",
) -> tuple[NativeHierarchy, str]:
    """Load a directory of PIC YAML files into a native hierarchy.

    Mirrors ``sax.load_recursive_netlist`` discovery (suffix match, sorted,
    duplicate rejection) but returns ``(cells, root)`` with native objects.
    """
    from pathlib import Path

    top_level_path = Path(str(top_level_path)).resolve()
    folder_path = top_level_path.parent

    def _net_name(path: Path) -> str:
        return path.name.removesuffix(ext)

    root = _net_name(top_level_path)
    cells: NativeHierarchy = {root: load_native_netlist(top_level_path)}
    for path in sorted(folder_path.rglob(f"*{ext}")):
        if not path.is_file() or path.resolve() == top_level_path:
            continue
        name = _net_name(path)
        if name in cells:
            msg = f"Duplicate recursive netlist component name {name!r}: {path}."
            raise ValueError(msg)
        cells[name] = load_native_netlist(path)
    return cells, root


def hierarchy_cell_maps(
    cells: Mapping[str, Netlist],
    models: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, str]]:
    """Build kfnetlist ``instance -> cell`` maps from a native hierarchy."""
    models = models or {}
    maps: dict[str, dict[str, str]] = {}
    for cell_name, nl in cells.items():
        entry: dict[str, str] = {}
        for name, inst in nl.instances.items():
            key = resolve(inst, models, cells)
            if key is not None and key in cells:
                entry[name] = key
        maps[cell_name] = entry
    return maps


def flatten_netlist(
    cells: Mapping[str, Netlist],
    root: str,
    *,
    models: Mapping[str, Any] | None = None,
    separator: str = "__",
    recursive: bool = True,
) -> Netlist:
    """Flatten a native hierarchy's root netlist using kfnetlist's flattening.

    Cells referenced by a factory that is replaced by an analytical model are not
    inlined (they have no child netlist). ``separator`` joins hierarchical
    instance names; ``__`` is Python-identifier safe.
    """
    from kfnetlist import flatten_netlists

    maps = hierarchy_cell_maps(cells, models)
    flattened = flatten_netlists(
        dict(cells),
        None,
        instance_cell_maps=maps,
        recursive=recursive,
        separator=separator,
    )
    return flattened[root]


def flatten_recursive_netlist(
    cells: Mapping[str, Netlist],
    *,
    models: Mapping[str, Any] | None = None,
    separator: str = "__",
) -> NativeHierarchy:
    """Flatten every cell of a native hierarchy in place-independent fashion."""
    from kfnetlist import flatten_netlists

    maps = hierarchy_cell_maps(cells, models)
    return flatten_netlists(
        dict(cells),
        None,
        instance_cell_maps=maps,
        recursive=True,
        separator=separator,
    )


def remove_unused_instances(nl: Netlist) -> Netlist:
    """Return a copy of *nl* with instances unreachable from its ports removed.

    Uses native ``remove_instances``; connectivity is read from the lowered
    topology tables so no legacy dictionary schema is involved.
    """
    import networkx as nx

    instances, nets, ports = lower(nl)
    graph = nx.Graph()
    for name in instances:
        graph.add_node(name)
    for net in nets:
        graph.add_edge(net["p1"].split(",")[0], net["p2"].split(",")[0])
    roots = {f"__port_{i}": ep.split(",")[0] for i, ep in enumerate(ports.values())}
    for node, target in roots.items():
        graph.add_node(node)
        graph.add_edge(node, target)
    keep: set[str] = set()
    for node in roots:
        keep |= nx.descendants(graph, node)
    base_keep = {name.split("<")[0] for name in keep}
    remove = [
        base
        for base in _base_instance_names(nl)
        if base not in base_keep
    ]
    result = copy_netlist(nl)
    if remove:
        result.remove_instances(remove)
    return result


def _base_instance_names(nl: Netlist) -> list[str]:
    return list(nl.instances)


def rename_instances(
    nl: Netlist, mapping: Mapping[str, str]
) -> Netlist:
    """Return a native copy with instances renamed by *mapping*."""
    d = nl.to_dict()
    instances = {}
    for name, inst in d.get("instances", {}).items():
        instances[mapping.get(name, name)] = inst
    d["instances"] = instances
    for net in d.get("nets", []):
        for member in net:
            if isinstance(member, dict) and "instance" in member:
                member["instance"] = mapping.get(member["instance"], member["instance"])
    placements = d.get("placements")
    if isinstance(placements, dict):
        d["placements"] = {
            mapping.get(name, name): value for name, value in placements.items()
        }
    factory = PlacedNetlist if isinstance(nl, PlacedNetlist) else Netlist
    return factory.from_dict(d)


def rename_models(nl: Netlist, mapping: Mapping[str, str]) -> Netlist:
    """Return a native copy with instance factory ``component`` names remapped."""
    d = nl.to_dict()
    for inst in d.get("instances", {}).values():
        if isinstance(inst, dict) and "component" in inst:
            inst["component"] = mapping.get(inst["component"], inst["component"])
    factory = PlacedNetlist if isinstance(nl, PlacedNetlist) else Netlist
    return factory.from_dict(d)
