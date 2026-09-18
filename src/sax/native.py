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


def to_hierarchy(
    netlist: object,
    *,
    top_level_name: str = "top_level",
) -> tuple[NativeHierarchy, str]:
    """Normalize supported input into ``(native cells, root name)``."""
    if is_native(netlist):
        return {top_level_name: netlist}, top_level_name

    if is_native_hierarchy(netlist):
        cells = dict(netlist)
        root = top_level_name if top_level_name in cells else next(iter(cells))
        return cells, root

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
                return {top_level_name: Netlist.from_dict(data)}, top_level_name
            return {top_level_name: from_legacy_flat(data)}, top_level_name

        cells: NativeHierarchy = {}
        for name, flat in data.items():
            if isinstance(flat, Netlist):
                cells[str(name)] = flat
            elif isinstance(flat, Mapping) and _looks_native(flat):
                cells[str(name)] = Netlist.from_dict(dict(flat))
            elif isinstance(flat, Mapping):
                cells[str(name)] = from_legacy_flat(flat)
            else:
                msg = f"Unsupported netlist entry {name!r}: {type(flat)}."
                raise TypeError(msg)
        root = top_level_name if top_level_name in cells else next(iter(cells))
        return cells, root

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
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]], dict[str, str]]:
    """Lower a native netlist to ``(instances, nets, ports)`` tables.

    These flat tables are the compiled input to the numerical backends, not a
    canonical netlist representation. Array instances are expanded to
    ``name<column.row>`` entries (zero-based), matching SAX instance naming.
    """
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
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]], dict[str, str], dict[str, str | None]]:
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
