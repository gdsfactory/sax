"""Simulation-specific compilation from kfnetlist topology to backend tables."""

from __future__ import annotations

import re
from collections.abc import Mapping
from itertools import pairwise
from typing import Any

from kfnetlist import (
    Netlist,
    NetlistInstance,
    NetlistPort,
    PortArrayRef,
    PortRef,
    RefNetlistInstance,
)

from .saxtypes.netlist import Instance, Instances, Nets


def _split_array_endpoint(endpoint: str) -> tuple[str, str, int | None, int | None]:
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
    """Lower a kfnetlist Netlist to ``(instances, nets, ports)`` tables.

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
    """Resolve one kfnetlist instance to a model/cell key.

    Returns ``None`` when the instance cannot be resolved.
    """
    models = models or {}
    cells = cells or {}
    component = inst.component
    cell = inst.netlist_id if isinstance(inst, RefNetlistInstance) else None
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
    return None


def missing_model_message(inst: NetlistInstance, path: str) -> str:
    """Describe the identities and lookups of an unresolved instance."""
    cell = inst.netlist_id if isinstance(inst, RefNetlistInstance) else None
    qualified = f"{inst.kcl}::{inst.component}"
    attempts = [
        *([f"models[{cell!r}]"] if cell is not None else []),
        f"models[{qualified!r}]",
        f"models[{inst.component!r}]",
        *([f"cells[{cell!r}]"] if cell is not None else []),
    ]
    return (
        f"Missing models at {path!r}: factory={inst.component!r}, "
        f"library={inst.kcl!r}, netlist_id={cell!r}. Tried {', '.join(attempts)}."
    )


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
    base, _, col, row = _split_array_endpoint(f"{name},_")
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


def prune_unconnected_instances(
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
    result = Netlist.from_dict(nl.to_dict())
    unused = [name for name in nl.instances if name not in reachable]
    if unused:
        result.remove_instances(unused)
    return result
