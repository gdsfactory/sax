"""Simulation-specific compilation of kfnetlist topology."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from itertools import pairwise
from typing import Any

from kfnetlist import (
    LeafNetlistInstance,
    Net,
    Netlist,
    NetlistInstance,
    NetlistPort,
    PortRef,
    RefNetlistInstance,
)


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


def _record_external_ports(
    externals: list[NetlistPort],
    endpoints: list[PortRef],
    declared: set[str],
    ports: dict[str, PortRef],
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


def _validate_external_ports(declared: set[str], ports: dict[str, PortRef]) -> None:
    missing = declared - ports.keys()
    if missing:
        msg = (
            f"Declared external ports have no instance connection: {sorted(missing)!r}."
        )
        raise ValueError(msg)
    if len(set(ports.values())) != len(ports):
        msg = "External port aliases targeting the same instance port are unsupported."
        raise ValueError(msg)


def _build_netlist(
    instances: Mapping[str, NetlistInstance],
    ports: Iterable[str],
    nets: Iterable[Net],
) -> Netlist:
    """Build a kfnetlist object from its typed instances, ports, and nets."""
    result = Netlist()
    for name, inst in instances.items():
        array = inst.array
        result.create_inst(
            name,
            inst.kcl,
            inst.component,
            inst.settings,
            na=array.na if array is not None else 1,
            nb=array.nb if array is not None else 1,
            info=inst.info,
            netlist_id=(
                inst.netlist_id if isinstance(inst, RefNetlistInstance) else None
            ),
        )
    for name in ports:
        result.create_port(name)
    for net in nets:
        result.add_net(net)
    return result


def lower(nl: Netlist) -> Netlist:
    """Apply SAX's pairwise solver rules to an array-expanded kfnetlist."""
    expanded = nl.expand_arrays()
    result = _build_netlist(expanded.instances, (p.name for p in expanded.ports), ())
    declared = {p.name for p in expanded.ports}
    ports: dict[str, PortRef] = {}
    for net in expanded.nets:
        members = list(net)
        externals = [m for m in members if isinstance(m, NetlistPort)]
        endpoints = [m for m in members if isinstance(m, PortRef)]
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
            result.create_net(a, b)
    _validate_external_ports(declared, ports)
    for name, ref in ports.items():
        result.create_net(NetlistPort(name), ref)
    return result


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
) -> tuple[Netlist, dict[str, str | None]]:
    """Lower *nl* and resolve every expanded instance to a model/cell key."""
    expanded = lower(nl)
    bindings: dict[str, str | None] = {}
    for name, inst in nl.instances.items():
        arr = inst.array
        na = max(int(arr.na), 1) if arr is not None else 1
        nb = max(int(arr.nb), 1) if arr is not None else 1
        key = resolve(inst, models, cells)
        for i in range(na):
            for j in range(nb):
                bindings[_expanded_name(name, i, j, na, nb)] = key
    return expanded, bindings


def port_ref(endpoint: str) -> PortRef:
    """Parse a probe endpoint into a kfnetlist port reference."""
    instance, separator, port = endpoint.partition(",")
    if not separator or not instance or not port:
        msg = f"Expected an instance port like 'instance,port'; got {endpoint!r}."
        raise ValueError(msg)
    return PortRef(instance, port)


def add_external_ports(nl: Netlist, ports: Mapping[str, str]) -> Netlist:
    """Attach generated parent-facing probe ports to a compiled netlist."""
    result = Netlist.from_dict(nl.to_dict())
    for name, endpoint in ports.items():
        if any(p.name == name for p in result.ports):
            msg = f"Hierarchical probe port {name!r} conflicts with an existing port."
            raise ValueError(msg)
        result.create_port(name)
        result.create_net(NetlistPort(name), port_ref(endpoint))
    return result


def handle_internal_ports(
    nl: Netlist,
    on_internal_port: str,
) -> tuple[Netlist, dict[str, str]]:
    """Drop or convert external ports that target internal connection nodes."""
    import warnings

    internal: set[PortRef] = set()
    attached: dict[str, PortRef] = {}
    for net in nl.nets:
        refs = [member for member in net if isinstance(member, PortRef)]
        externals = [member for member in net if isinstance(member, NetlistPort)]
        if externals:
            for external in externals:
                attached[external.name] = refs[0]
        elif len(refs) >= 2:
            internal.update(refs)
    probes: dict[str, str] = {}
    kept: list[str] = []
    for port in nl.ports:
        name = port.name
        ref = attached[name]
        endpoint = f"{ref.instance},{ref.port}"
        if ref not in internal:
            kept.append(name)
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
    surviving_nets = [
        net
        for net in nl.nets
        if not any(
            isinstance(member, NetlistPort) and member.name not in kept
            for member in net
        )
    ]
    return _build_netlist(nl.instances, kept, surviving_nets), probes


def expand_probes(
    nl: Netlist,
    probes: Mapping[str, str],
) -> Netlist:
    """Insert ideal probes into a compiled kfnetlist object.

    ``_fwd`` measures the wave travelling into the targeted instance port.
    """
    if not probes:
        return nl
    result = nl
    for probe_name, target in probes.items():
        ports = [port.name for port in result.ports]
        fwd_port = f"{probe_name}_fwd"
        bwd_port = f"{probe_name}_bwd"
        if fwd_port in ports or bwd_port in ports:
            msg = (
                f"Probe '{probe_name}' would create ports '{fwd_port}'/"
                f"'{bwd_port}' which conflict with existing ports."
            )
            raise ValueError(msg)
        probe_instance = f"_probe_{probe_name}"
        if result.has_instance(probe_instance):
            msg = (
                f"Probe instance name '{probe_instance}' conflicts with an "
                "existing instance."
            )
            raise ValueError(msg)

        target_ref = port_ref(target)
        in_ref = PortRef(probe_instance, "in")
        out_ref = PortRef(probe_instance, "out")
        nets = list(result.nets)
        intercepted: PortRef | None = None
        for index, net in enumerate(nets):
            members = list(net)
            if target_ref in members and all(isinstance(m, PortRef) for m in members):
                intercepted = next(
                    (m for m in members if isinstance(m, PortRef) and m != target_ref),
                    None,
                )
                nets.pop(index)
                break
        if intercepted is not None:
            nets.append(Net([intercepted, in_ref]))
        else:
            for index, net in enumerate(nets):
                members = list(net)
                if target_ref in members and any(
                    isinstance(m, NetlistPort) for m in members
                ):
                    external = next(m for m in members if isinstance(m, NetlistPort))
                    nets[index] = Net([external, in_ref])
                    break
        nets.append(Net([out_ref, target_ref]))
        nets.append(Net([NetlistPort(fwd_port), PortRef(probe_instance, "tap_fwd")]))
        nets.append(Net([NetlistPort(bwd_port), PortRef(probe_instance, "tap_bwd")]))
        instances = result.instances
        instances[probe_instance] = LeafNetlistInstance(
            "sax", "_ideal_probe", name=probe_instance
        )
        result = _build_netlist(instances, [*ports, fwd_port, bwd_port], nets)
    return result


def expand_modes(  # noqa: C901
    nl: Netlist,
    instance_port_modes: Mapping[str, Mapping[str, set[str]]],
    *,
    ignore_impossible_connections: bool = False,
) -> Netlist:
    """Keep mode-expanded wiring in kfnetlist until solver analysis."""
    result = _build_netlist(nl.instances, (), ())

    def modes(ref: PortRef) -> set[str] | None:
        try:
            return instance_port_modes[ref.instance][ref.port]
        except KeyError as error:
            if ignore_impossible_connections:
                return None
            available = list(instance_port_modes[ref.instance])
            msg = (
                f"Instance {ref.instance} does not contain port {ref.port}. "
                f"Available ports: {available}."
            )
            raise KeyError(msg) from error

    for net in nl.nets:
        refs = [member for member in net if isinstance(member, PortRef)]
        external = next(
            (member for member in net if isinstance(member, NetlistPort)), None
        )
        if external is not None:
            ref = refs[0]
            active = modes(ref)
            if active is None:
                continue
            if not active:
                result.create_port(external.name)
                result.create_net(external, ref)
            else:
                for mode in sorted(active):
                    name = f"{external.name}@{mode}"
                    result.create_port(name)
                    result.create_net(
                        NetlistPort(name), PortRef(ref.instance, f"{ref.port}@{mode}")
                    )
            continue
        if len(refs) < 2:
            continue
        left, right = refs
        left_modes = modes(left)
        right_modes = modes(right)
        if left_modes is None or right_modes is None:
            continue
        if not left_modes and not right_modes:
            result.add_net(net)
        elif not left_modes or not right_modes:
            msg = (
                "trying to connect a multimode model to single mode model.\n"
                "Please update your models dictionary.\n"
                f"Problematic connection: '{left.instance},{left.port}':"
                f"'{right.instance},{right.port}'"
            )
            raise ValueError(msg)
        else:
            for mode in sorted(left_modes & right_modes):
                result.create_net(
                    PortRef(left.instance, f"{left.port}@{mode}"),
                    PortRef(right.instance, f"{right.port}@{mode}"),
                )
    return result


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
