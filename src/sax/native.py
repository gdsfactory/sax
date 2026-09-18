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
2. library-qualified factory model (``models["library::component"]``),
3. exact bare factory model (``models[component]``), if unambiguous,
4. recurse via the instantiated cell reference (``cells[cell]``),
5. recurse via a cell whose name equals the factory name,
6. otherwise the instance is missing a model.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from itertools import pairwise
from typing import Any, TypeGuard, TypeVar

from kfnetlist import (
    Netlist,
    NetlistInstance,
    NetlistPort,
    PlacedNetlist,
    Placement,
    PortArrayRef,
    PortRef,
)

from .saxtypes.netlist import Instance, Instances, Nets, val_placement

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

NetlistT = TypeVar("NetlistT", bound=Netlist)

NativeHierarchy = dict[str, Netlist]
"""Mapping of cell name to native kfnetlist netlist (SAX's canonical hierarchy)."""

InstanceSettings = dict[str, dict[str, Any]]
HierarchySettings = dict[str, InstanceSettings]


def is_native(obj: object) -> TypeGuard[Netlist]:
    """Return whether *obj* is a native kfnetlist netlist object."""
    return isinstance(obj, Netlist)


def is_native_hierarchy(obj: object) -> TypeGuard[Mapping[str, Netlist]]:
    """Return whether *obj* is a native ``{cell name: Netlist}`` mapping."""
    if not isinstance(obj, Mapping) or not obj:
        return False
    return all(isinstance(k, str) and isinstance(v, Netlist) for k, v in obj.items())


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
                max(int(array.get("columns", array.get("num_a", 1))), 1),
                max(int(array.get("rows", array.get("num_b", 1))), 1),
            )

    endpoints: list[str] = []
    endpoints.extend(str(k) for k in (flat.get("connections") or {}))
    endpoints.extend(str(v) for v in (flat.get("connections") or {}).values())
    for net in flat.get("nets") or []:
        endpoints.append(str(net["p1"]))
        endpoints.append(str(net["p2"]))
    endpoints.extend(str(v) for v in (flat.get("ports") or {}).values())
    for bundle in (flat.get("routes") or {}).values():
        for source, target in (bundle.get("links") or {}).items():
            endpoints.extend((str(source), str(target)))

    for endpoint in endpoints:
        base, _, col, row = _split_legacy_endpoint(endpoint)
        if col is None:
            continue
        na, nb = sizes.get(base, (1, 1))
        sizes[base] = (max(na, col + 1), max(nb, (row or 0) + 1))
    return sizes


def _add_legacy_instance(
    nl: Netlist,
    name: str,
    inst: object,
    sizes: Mapping[str, tuple[int, int]],
    placement: Mapping[str, Any],
    settings_table: InstanceSettings | None,
) -> None:
    if isinstance(inst, str):
        component, settings, info = inst, {}, {}
    elif isinstance(inst, Mapping):
        if "component" not in inst:
            msg = f"Instance {name!r} is missing a 'component' key: {inst!r}."
            raise ValueError(msg)
        component = str(inst["component"])
        settings = dict(inst.get("settings") or {})
        info = dict(inst.get("info") or {})
        settings.update(info)
    else:
        msg = f"Unsupported legacy instance {name!r}: {inst!r}."
        raise TypeError(msg)
    na, nb = sizes.get(name, (1, 1))
    if settings_table is not None:
        settings_table[name] = settings
        settings, info = {}, {}
    if isinstance(nl, PlacedNetlist):
        normalized = val_placement(placement)
        cell = (
            str(inst.get("cell", component)) if isinstance(inst, Mapping) else component
        )
        nl.create_inst(
            name,
            kcl="",
            component=component,
            settings=settings,
            na=na,
            nb=nb,
            info=info,
            cell=cell,
            placement=Placement(
                x=normalized["x"],
                y=normalized["y"],
                orientation=float(normalized["rotation"]),
                mirror=normalized["mirror"],
                bbox={"left": 0.0, "bottom": 0.0, "right": 0.0, "top": 0.0},
            ),
        )
    else:
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


def from_legacy_flat(
    flat: Mapping[str, Any],
    *,
    settings_table: InstanceSettings | None = None,
) -> Netlist:
    """Adapt one legacy SAX flat netlist into a native kfnetlist netlist."""
    placed = bool(flat.get("placements")) or any(
        isinstance(inst, Mapping) and "cell" in inst
        for inst in flat.get("instances", {}).values()
    )
    nl = PlacedNetlist() if placed else Netlist()
    for name in flat.get("ports") or {}:
        nl.create_port(str(name))
    sizes = _scan_legacy_arrays(flat)
    for name, inst in (flat.get("instances") or {}).items():
        _add_legacy_instance(
            nl,
            str(name),
            inst,
            sizes,
            (flat.get("placements") or {}).get(name, {}),
            settings_table,
        )
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
    if isinstance(data.get("ports"), list):
        return True
    instances = data.get("instances")
    if not isinstance(instances, Mapping) or not instances:
        return False
    return any(
        isinstance(inst, Mapping) and "kcl" in inst and "component" in inst
        for inst in instances.values()
    )


def deserialize_netlist(data: Mapping[str, Any]) -> Netlist:
    """Decode native data without dropping placed-instance identity/geometry."""
    placed = any(
        "cell" in inst or "placement" in inst
        for inst in data.get("instances", {}).values()
    )
    factory = PlacedNetlist if placed else Netlist
    return factory.from_dict(dict(data))


def to_hierarchy(
    netlist: object,
    *,
    top_level_name: str | None = None,
    settings_table: HierarchySettings | None = None,
) -> tuple[NativeHierarchy, str]:
    """Normalize supported input into ``(cells, root)``."""
    flat_name = top_level_name if top_level_name is not None else "top_level"
    if is_native(netlist):
        return {flat_name: netlist}, flat_name

    if is_native_hierarchy(netlist):
        cells = dict(netlist)
        root = _select_root(cells, top_level_name)
        return cells, root

    if isinstance(netlist, str):
        try:
            decoded = json.loads(netlist)
        except json.JSONDecodeError:
            msg = "Native string input must be JSON-encoded kfnetlist data."
            raise TypeError(msg) from None
        return to_hierarchy(
            decoded,
            top_level_name=top_level_name,
            settings_table=settings_table,
        )

    if isinstance(netlist, Mapping):
        return _mapping_hierarchy(netlist, top_level_name, settings_table)
    msg = f"Cannot interpret {type(netlist)} as a netlist."
    raise TypeError(msg)


def _adapt_cell(
    data: Mapping[str, Any],
    key: str,
    settings_table: HierarchySettings | None,
) -> Netlist:
    if _looks_native(data):
        return deserialize_netlist(data)
    saved: InstanceSettings | None = None
    if settings_table is not None:
        saved = {}
        settings_table[key] = saved
    return from_legacy_flat(data, settings_table=saved)


def _is_pic_document(data: Mapping[str, Any]) -> bool:
    if "modules" not in data:
        return False
    modules = data["modules"]
    if "toplevel" in data or not isinstance(modules, Mapping):
        return True
    # A hierarchy may itself contain a concrete cell named "modules".
    if _looks_native(modules):
        return False
    instances = modules.get("instances")
    is_instance_table = isinstance(instances, Mapping) and all(
        isinstance(inst, str)
        or callable(inst)
        or (isinstance(inst, Mapping) and "component" in inst)
        for inst in instances.values()
    )
    return not is_instance_table


def _mapping_hierarchy(
    data: Mapping[str, Any],
    top_level_name: str | None,
    settings_table: HierarchySettings | None,
) -> tuple[NativeHierarchy, str]:
    if data and all(
        isinstance(flat, Mapping) and _looks_native(flat) for flat in data.values()
    ):
        decoded = {str(name): deserialize_netlist(flat) for name, flat in data.items()}
        return decoded, _select_root(decoded, top_level_name)
    if _is_pic_document(data):
        modules = data["modules"]
        if not isinstance(modules, Mapping) or not modules:
            msg = "PIC modules must be a nonempty mapping of cell definitions."
            raise ValueError(msg)
        for name, module in modules.items():
            _validate_pic_module(module, str(name))
        root = _select_root(modules, top_level_name, preferred=data.get("toplevel"))
        cells = {
            str(name): _adapt_cell(module, str(name), settings_table)
            for name, module in modules.items()
        }
        return cells, root
    if "instances" in data:
        root = top_level_name if top_level_name is not None else "top_level"
        return {root: _adapt_cell(data, root, settings_table)}, root
    cells: NativeHierarchy = {}
    for name, flat in data.items():
        key = str(name)
        if isinstance(flat, Netlist):
            cells[key] = flat
        elif isinstance(flat, Mapping):
            cells[key] = _adapt_cell(flat, key, settings_table)
        else:
            msg = f"Unsupported netlist entry {name!r}: {type(flat)}."
            raise TypeError(msg)
    return cells, _select_root(cells, top_level_name)


# ---------------------------------------------------------------------------
# Lowering to backend topology tables
# ---------------------------------------------------------------------------


def _expanded_name(name: str, i: int, j: int, na: int, nb: int) -> str:
    if na <= 1 and nb <= 1:
        return name
    return f"{name}<{i}.{j}>"


def _member_endpoint(
    member: NetlistPort | PortRef | PortArrayRef,
    array_sizes: Mapping[str, tuple[int, int]],
) -> str | None:
    if isinstance(member, NetlistPort):
        return None
    if member.instance not in array_sizes:
        msg = f"Net references unknown instance {member.instance!r}."
        raise ValueError(msg)
    if isinstance(member, PortArrayRef):
        na, nb = array_sizes[member.instance]
        if not (1 <= member.ia <= na and 1 <= member.ib <= nb):
            msg = f"Array reference {member!r} is outside its declared dimensions."
            raise ValueError(msg)
        key = _expanded_name(
            member.instance,
            member.ia - 1,
            member.ib - 1,
            *array_sizes.get(member.instance, (1, 1)),
        )
        return f"{key},{member.port}"
    key = _expanded_name(
        member.instance, 0, 0, *array_sizes.get(member.instance, (1, 1))
    )
    return f"{key},{member.port}"


def _record_external_ports(
    externals: list[NetlistPort],
    endpoints: list[str],
    declared: set[str],
    ports: dict[str, str],
) -> None:
    for external in externals:
        if external.name not in declared:
            msg = f"Net references undeclared external port {external.name!r}."
            raise ValueError(msg)
        endpoint = endpoints[0]
        if external.name in ports and ports[external.name] != endpoint:
            msg = f"External port {external.name!r} targets multiple instance ports."
            raise ValueError(msg)
        ports[external.name] = endpoint


def _validate_external_ports(declared: set[str], ports: dict[str, str]) -> None:
    missing = declared - ports.keys()
    if missing:
        msg = (
            f"Declared external ports have no instance connection: {sorted(missing)!r}."
        )
        raise ValueError(msg)
    if len(set(ports.values())) != len(ports):
        msg = "External port aliases targeting the same instance port are unsupported."
        raise ValueError(msg)


def lower(
    nl: Netlist,
) -> tuple[Instances, Nets, dict[str, str]]:
    """Lower a native netlist to ``(instances, nets, ports)`` tables.

    These flat tables are the compiled input to the numerical backends, not a
    canonical netlist representation. Array instances are expanded to
    ``name<column.row>`` entries (zero-based), matching SAX instance naming.
    """
    instances: Instances = {}
    array_sizes: dict[str, tuple[int, int]] = {}
    for name, inst in nl.instances.items():
        arr = inst.array
        na = max(int(arr.na), 1) if arr is not None else 1
        nb = max(int(arr.nb), 1) if arr is not None else 1
        array_sizes[name] = (na, nb)
        for i in range(na):
            for j in range(nb):
                entry: Instance = {"component": inst.component}
                if inst.settings:
                    entry["settings"] = dict(inst.settings)
                instances[_expanded_name(name, i, j, na, nb)] = entry

    declared = {p.name for p in nl.ports}
    ports: dict[str, str] = {}
    nets: Nets = []
    for net in nl.nets:
        members = list(net)
        externals = [m for m in members if isinstance(m, NetlistPort)]
        internal = [m for m in members if not isinstance(m, NetlistPort)]
        endpoints = [_member_endpoint(m, array_sizes) for m in internal]
        endpoints = [e for e in endpoints if e is not None]
        if externals and not endpoints:
            msg = (
                "External-only nets are unsupported; connect each port to an instance."
            )
            raise ValueError(msg)
        if len(endpoints) > 2:
            msg = (
                "Native nets with more than two instance ports require an explicit "
                "junction model or explicitly specified pairwise connections."
            )
            raise ValueError(msg)
        _record_external_ports(externals, endpoints, declared, ports)
        for a, b in pairwise(endpoints):
            nets.append({"p1": a, "p2": b})
    _validate_external_ports(declared, ports)
    return instances, nets, ports


def resolve(
    inst: NetlistInstance,
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
    library = inst.kcl
    qualified = f"{library}::{component}"
    if library and qualified in models:
        return qualified
    if component in models:
        libraries = {
            other.kcl
            for nl in cells.values()
            for other in nl.instances.values()
            if other.component == component
        }
        if len(libraries) > 1:
            msg = (
                f"Ambiguous factory model {component!r} across libraries "
                f"{sorted(libraries)!r}; supply library::component bindings "
                "or explicit cell overrides."
            )
            raise ValueError(msg)
        return component
    if cell is not None and cell in cells:
        return cell
    if component in cells:
        return component
    return None


def missing_model_message(inst: NetlistInstance, path: str) -> str:
    """Describe the identities and lookups of an unresolved instance."""
    cell = getattr(inst, "cell", None)
    qualified = f"{inst.kcl}::{inst.component}"
    attempts = [
        f"models[{cell!r}]",
        f"models[{qualified!r}]",
        f"models[{inst.component!r}]",
        f"cells[{cell!r}]",
        f"cells[{inst.component!r}]",
    ]
    return (
        f"Missing models at {path!r}: factory={inst.component!r}, "
        f"library={inst.kcl!r}, cell={cell!r}. Tried {', '.join(attempts)}."
    )


def copy_netlist(nl: NetlistT) -> NetlistT:
    """Return an independent native copy of *nl*."""
    return type(nl).from_dict(nl.to_dict())


def lower_bindings(
    nl: Netlist,
    models: Mapping[str, Any],
    cells: Mapping[str, Any],
) -> tuple[
    Instances,
    Nets,
    dict[str, str],
    dict[str, str | None],
]:
    """Lower *nl* and resolve every expanded instance to a model/cell key."""
    instances, nets, ports = lower(nl)
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
    return endpoint.split(",", 1)[0]


def handle_internal_ports(
    nets: Nets,
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


def _intercept_probe_target(
    nets: Nets,
    ports: dict[str, str],
    target: str,
    probe_instance: str,
) -> str | None:
    for i, net in enumerate(nets):
        if target in (net["p1"], net["p2"]):
            nets.pop(i)
            return net["p2"] if net["p1"] == target else net["p1"]
    for pname, endpoint in ports.items():
        if endpoint == target:
            in_side = f"{probe_instance},in"
            ports[pname] = in_side
            return in_side
    return None


def expand_probes_tables(
    instances: Instances,
    nets: Nets,
    ports: dict[str, str],
    probes: Mapping[str, str],
) -> tuple[Instances, Nets, dict[str, str]]:
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

        in_side = _intercept_probe_target(nets, ports, target, probe_instance)

        instances[probe_instance] = {"component": "_ideal_probe"}
        if in_side is not None and in_side != f"{probe_instance},in":
            nets.append({"p1": in_side, "p2": f"{probe_instance},in"})
        nets.append({"p1": f"{probe_instance},out", "p2": target})
        ports[fwd_port] = f"{probe_instance},tap_fwd"
        ports[bwd_port] = f"{probe_instance},tap_bwd"
    return instances, nets, ports


def _validate_probe_name(nl: Netlist, name: str, *, insert_instance: bool) -> None:
    ports = {port.name for port in nl.ports}
    if {f"{name}_fwd", f"{name}_bwd"} & ports:
        msg = f"Probe {name!r} would create ports that conflict with existing ports."
        raise ValueError(msg)
    if insert_instance and f"_probe_{name}" in nl.instances:
        msg = (
            f"Probe instance name '_probe_{name}' conflicts with an existing instance."
        )
        raise ValueError(msg)


def _probe_instance_name(nl: Netlist, name: str) -> str:
    base, _, col, row = _split_legacy_endpoint(f"{name},_")
    inst = nl.instances.get(base)
    if inst is None:
        return name  # Preserve the existing missing-instance diagnostic.
    na, nb = (inst.array.na, inst.array.nb) if inst.array is not None else (1, 1)
    col, row = col or 0, row or 0
    if not (0 <= col < na and 0 <= row < nb):
        msg = f"Probe instance {name!r} is outside its declared array dimensions."
        raise ValueError(msg)
    return _expanded_name(base, col, row, na, nb)


def _probe_endpoint(nl: Netlist, endpoint: str) -> str:
    instance, separator, port = endpoint.partition(",")
    return f"{_probe_instance_name(nl, instance)}{separator}{port}"


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
        parts = re.split(r"\.(?![^<]*>)", target)
        if len(parts) == 1:
            _validate_probe_name(cells[root], probe_name, insert_instance=True)
            top[probe_name] = _probe_endpoint(cells[root], target)
            continue
        current = root
        path: list[tuple[str, str]] = []
        for instance_name in parts[:-1]:
            _validate_probe_name(cells[current], probe_name, insert_instance=False)
            inst = cells[current].instances.get(instance_name.split("<", 1)[0])
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
            path.append((_probe_instance_name(cells[current], instance_name), current))
            current = key
        _validate_probe_name(cells[current], probe_name, insert_instance=True)
        per_cell.setdefault(current, {})[probe_name] = _probe_endpoint(
            cells[current], parts[-1]
        )
        paths[probe_name] = path
    return top, per_cell, paths


def _select_root(
    cells: Mapping[str, Any],
    requested: str | None,
    *,
    preferred: str | None = None,
) -> str:
    if not cells:
        msg = "Netlist hierarchy must contain at least one cell."
        raise ValueError(msg)
    selected = requested if requested is not None else preferred
    if selected is not None:
        if selected not in cells:
            msg = (
                f"Unknown top-level cell {selected!r}; "
                f"available cells: {list(cells)!r}."
            )
            raise ValueError(msg)
        return selected
    return "top_level" if "top_level" in cells else next(iter(cells))


def _reject_pic_expressions(value: object, path: str) -> None:
    if isinstance(value, str) and "${" in value:
        msg = (
            f"Unsupported PIC expression at {path}: {value!r}. "
            "Supply numerical settings."
        )
        raise ValueError(msg)
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_pic_expressions(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_pic_expressions(child, f"{path}[{index}]")


def _validate_pic_module(module: object, name: str) -> None:
    if not isinstance(module, Mapping):
        msg = f"PIC module {name!r} must be a mapping."
        raise TypeError(msg)
    _reject_pic_expressions(module, name)
    for field in ("settings", "info", "metadata"):
        if module.get(field):
            msg = (
                f"PIC module {name!r}: module-level {field!r} is unsupported by "
                "native loaders; supply instance settings or retain the original "
                "document with sax.load_netlist."
            )
            raise ValueError(msg)


def load_pic_yaml(
    content: str | Mapping[str, Any],
    *,
    top_level_name: str | None = None,
) -> tuple[NativeHierarchy, str]:
    """Load legacy/PIC document data into native cells and an explicit root.

    An explicit root wins over document ``toplevel``. Legacy root settings remain
    ignored for compatibility; module-document settings/metadata and expressions
    are rejected when they cannot be represented by native connectivity.
    """
    import yaml

    data = yaml.safe_load(content) if isinstance(content, str) else dict(content)
    if not isinstance(data, Mapping):
        msg = f"PIC document must be a mapping, got {type(data)}."
        raise TypeError(msg)
    _reject_pic_expressions(data, "document")
    return to_hierarchy(data, top_level_name=top_level_name)


def _read_pic_string(content: str) -> str:
    from pathlib import Path

    if "\n" not in content:
        try:
            path = Path(content)
            if path.is_file():
                return path.read_text()
        except OSError:
            pass  # Long single-line YAML/JSON is content, not a filename.
    return content


def load_native_netlist(content_or_path: object) -> Netlist:
    """Load a single native netlist; use ``load_pic_yaml`` for a hierarchy."""
    from pathlib import Path

    if isinstance(content_or_path, Netlist):
        return content_or_path
    if isinstance(content_or_path, Mapping):
        content = content_or_path
    elif callable(reader := getattr(content_or_path, "read", None)):
        content = reader()
    elif isinstance(content_or_path, Path):
        content = content_or_path.read_text()
    elif isinstance(content_or_path, str):
        content = _read_pic_string(content_or_path)
    else:
        msg = f"Cannot load native netlist from {type(content_or_path)}."
        raise TypeError(msg)
    if not isinstance(content, (str, Mapping)):
        msg = "Native netlist input must contain text or a mapping."
        raise TypeError(msg)
    cells, root = load_pic_yaml(content)
    if len(cells) != 1:
        msg = (
            "Document contains a hierarchy; "
            "use native.load_pic_yaml to retain all cells."
        )
        raise ValueError(msg)
    return cells[root]


def load_native_recursive_netlist(
    top_level_path: object,
    ext: str = ".pic.yml",
) -> tuple[NativeHierarchy, str]:
    """Load native PIC cells with normalized names and duplicate rejection."""
    from pathlib import Path

    from .utils import clean_string

    top_path = Path(str(top_level_path)).resolve()

    def read_cells(path: Path) -> tuple[NativeHierarchy, str]:
        import yaml

        data = yaml.safe_load(path.read_text())
        loaded, root = load_pic_yaml(data)
        if "instances" in data:
            root = clean_string(path.name.removesuffix(ext))
            loaded = {root: next(iter(loaded.values()))}
        return loaded, root

    cells, root = read_cells(top_path)
    cells = {root: cells[root], **cells}
    for path in sorted(top_path.parent.rglob(f"*{ext}")):
        if not path.is_file() or path.resolve() == top_path:
            continue
        loaded, _ = read_cells(path)
        for name, nl in loaded.items():
            if name in cells:
                msg = f"Duplicate recursive netlist component name {name!r}: {path}."
                raise ValueError(msg)
            cells[name] = nl
    return cells, root


def hierarchy_cell_maps(
    cells: Mapping[str, Netlist],
    models: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, str]]:
    """Build kfnetlist ``instance -> cell`` maps from a native hierarchy."""
    models = models or {}
    maps: dict[str, dict[str, str]] = {}
    opaque = "_sax_opaque_model"
    while opaque in cells:
        opaque += "_"
    for cell_name, nl in cells.items():
        entry: dict[str, str] = {}
        for name, inst in nl.instances.items():
            key = resolve(inst, models, cells)
            if key in models:
                # Explicit maps override PlacedInstance.cell during flattening.
                # A deliberately absent target leaves this instance untouched.
                entry[name] = opaque
            elif key is not None and key in cells:
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
        exclude=list(set(models or {}) & set(cells)),
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
        exclude=list(set(models or {}) & set(cells)),
        recursive=True,
        separator=separator,
    )


def remove_unused_instances(
    nl: Netlist,
    *,
    keep: tuple[str, ...] = (),
) -> Netlist:
    """Copy and prune instances disconnected from declared ports or probe roots."""
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(nl.instances)
    roots = {endpoint.split(",", 1)[0].split("<", 1)[0] for endpoint in keep}
    declared = {port.name for port in nl.ports}
    for net in nl.nets:
        members = list(net)
        names = [m.instance for m in members if isinstance(m, (PortRef, PortArrayRef))]
        graph.add_edges_from(pairwise(names))
        if any(isinstance(m, NetlistPort) and m.name in declared for m in members):
            roots.update(names)
    reachable: set[str] = set()
    for root in roots:
        if root in graph:
            reachable.update(nx.node_connected_component(graph, root))
    result = copy_netlist(nl)
    unused = [name for name in nl.instances if name not in reachable]
    if unused:
        result.remove_instances(unused)
    return result


def rename_instances(nl: Netlist, mapping: Mapping[str, str]) -> Netlist:
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
