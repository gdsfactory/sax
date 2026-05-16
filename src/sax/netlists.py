"""Netlist utilities."""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Literal, cast, overload

import networkx as nx
from natsort import natsorted

import sax

__all__ = [  # noqa: RUF022
    "netlist",
    "flatten_netlist",
    "expand_probes",
    "extract_port_probes",
]


def netlist(
    netlist: sax.AnyNetlist,
    *,
    top_level_name: str = "top_level",
) -> sax.RecursiveNetlist:
    """Convert a netlist to recursive netlist format.

    Ensures that the input netlist is in recursive netlist format, which is a
    dictionary mapping component names to their netlists. If a top-level circuit
    with the specified name exists, it will be promoted to the top level.

    Args:
        netlist: Input netlist in any format (flat netlist or recursive netlist).
        top_level_name: Name to use for the top-level circuit when converting
            from flat netlist format. Defaults to "top_level".

    Returns:
        Recursive netlist dictionary with component name keys.

    Raises:
        ValueError: If the input netlist format is invalid.

    Example:
        ```python
        # Convert flat netlist to recursive format
        flat_net = {
            "instances": {"wg1": {"component": "waveguide"}},
            "ports": {"in": "wg1,in", "out": "wg1,out"},
        }
        rec_net = netlist(flat_net, top_level_name="my_circuit")
        # Result: {"my_circuit": flat_net}
        ```
    """
    if not isinstance(netlist, dict):
        msg = (
            "Invalid argument for `netlist`. "
            "Expected type: dict | Netlist | RecursiveNetlist. "
            f"Got: {type(netlist)}."
        )
        raise TypeError(msg)

    if "instances" in netlist:
        return {top_level_name: cast(sax.Netlist, netlist)}

    recnet = cast(sax.RecursiveNetlist, netlist)
    top_level = recnet.get(top_level_name, None)
    if top_level is None:
        return recnet
    return {top_level_name: top_level, **recnet}


def flatten_netlist(recnet: sax.RecursiveNetlist, sep: str = "~") -> sax.Netlist:
    """Flatten a recursive netlist into a single flat netlist.

    Converts a hierarchical (recursive) netlist into a single flat netlist by
    inlining all sub-circuits. Instance names are prefixed to avoid conflicts.

    Args:
        recnet: Recursive netlist to flatten.
        sep: Separator used for hierarchical instance naming. Defaults to "~".

    Returns:
        Single flat netlist with all hierarchies inlined.

    Example:
        ```python
        # Flatten a hierarchical netlist
        recnet = {
            "top": {
                "instances": {"sub1": {"component": "subcircuit"}},
                "ports": {"in": "sub1,in"},
            },
            "subcircuit": {
                "instances": {"wg1": {"component": "waveguide"}},
                "ports": {"in": "wg1,in"},
            },
        }
        flat = flatten_netlist(recnet)
        # Result has instances like "sub1~wg1" for the flattened hierarchy
        ```
    """
    first_name = next(iter(recnet.keys()))
    net = deepcopy(recnet[first_name])
    _flatten_netlist_into(recnet, net, sep)
    return net


@overload
def remove_unused_instances(netlist: sax.Netlist) -> sax.Netlist: ...


@overload
def remove_unused_instances(netlist: sax.RecursiveNetlist) -> sax.RecursiveNetlist: ...


def remove_unused_instances(netlist: sax.AnyNetlist) -> sax.AnyNetlist:
    if "instances" in netlist:
        net = cast(sax.Netlist, deepcopy(netlist))
        names = _get_nodes_to_remove(_get_connectivity_graph(net), net)
        _remove_instances(net, names)
        _remove_connections(net, names)
        _remove_ports(net, names)
        return net

    recnet = cast(sax.RecursiveNetlist, {**netlist})
    for name, flat_netlist in recnet.items():
        recnet[name] = remove_unused_instances(flat_netlist)

    return recnet


@overload
def rename_instances(
    netlist: sax.Netlist,
    mapping: dict[sax.InstanceName, sax.InstanceName],
) -> sax.Netlist: ...


@overload
def rename_instances(
    netlist: sax.RecursiveNetlist,
    mapping: dict[sax.InstanceName, sax.InstanceName],
) -> sax.RecursiveNetlist: ...


def rename_instances(
    netlist: sax.AnyNetlist,
    mapping: dict[sax.InstanceName, sax.InstanceName],
) -> sax.AnyNetlist:
    if (recnet := sax.try_into[sax.RecursiveNetlist](netlist)) is not None:
        return {k: rename_instances(v, mapping) for k, v in recnet.items()}

    # it's a sax.Netlist now:
    net: sax.Netlist = sax.into[sax.Netlist](netlist)
    new: sax.Netlist = {"instances": {}, "ports": {}}
    inverse_mapping = {v: k for k, v in mapping.items()}
    if len(inverse_mapping) != len(mapping):
        msg = "Duplicate names to map onto found."
        raise ValueError(msg)
    new["instances"] = {mapping.get(k, k): v for k, v in net["instances"].items()}
    new["connections"] = {}
    for ip1, ip2 in net.get("connections", {}).items():
        i1, p1 = ip1.split(",")
        i2, p2 = ip2.split(",")
        i1 = mapping.get(i1, i1)
        i2 = mapping.get(i2, i2)
        new["connections"][f"{i1},{p1}"] = f"{i2},{p2}"
    new["ports"] = {}
    for q, ip in net.get("ports", {}).items():
        i, p = ip.split(",")
        i = mapping.get(i, i)
        new["ports"][q] = f"{i},{p}"

    new["placements"] = {
        mapping.get(k, k): v for k, v in net.get("placements", {}).items()
    }
    return {**net, **new}


@overload
def rename_models(
    netlist: sax.Netlist,
    mapping: dict[sax.Name, sax.Name],
) -> sax.Netlist: ...


@overload
def rename_models(
    netlist: sax.RecursiveNetlist,
    mapping: dict[sax.Name, sax.Name],
) -> sax.RecursiveNetlist: ...


def rename_models(
    netlist: sax.AnyNetlist,
    mapping: dict[sax.Name, sax.Name],
) -> sax.AnyNetlist:
    if (recnet := sax.try_into[sax.RecursiveNetlist](netlist)) is not None:
        return {k: rename_models(v, mapping) for k, v in recnet.items()}

    # it's a sax.Netlist now:
    net: sax.Netlist = deepcopy(sax.into[sax.Netlist](netlist))
    inverse_mapping = {v: k for k, v in mapping.items()}
    if len(inverse_mapping) != len(mapping):
        msg = "Duplicate names to map onto found."
        raise ValueError(msg)

    for inst in net["instances"].values():
        inst["component"] = mapping.get(inst["component"], inst["component"])
    return net


def _remove_ports(net: sax.Netlist, names: set[sax.InstanceName]) -> None:
    for pname, conn in list(net.get("ports", {}).items()):
        name, _ = conn.split(",")
        if name in names and pname in net.get("ports", {}):
            del net.get("ports", {})[pname]


def _remove_connections(net: sax.Netlist, names: set[sax.InstanceName]) -> None:
    net["connections"] = {**net.get("connections", {})}
    for conn1, conn2 in list(net["connections"].items()):
        for conn in [conn1, conn2]:
            name, _ = conn.split(",")
            if name in names and conn1 in net["connections"]:
                del net["connections"][conn1]
    if not net["connections"]:
        del net["connections"]
    if "nets" in net:
        net["nets"] = [
            n
            for n in net.get("nets", [])
            if n["p1"].split(",")[0] not in names and n["p2"].split(",")[0] not in names
        ]


def _remove_instances(net: sax.Netlist, names: set[sax.InstanceName]) -> None:
    for name in names:
        if name in net["instances"]:
            del net["instances"][name]


def _get_nodes_to_remove(graph: nx.Graph, netlist: sax.Netlist) -> set[str]:
    nodes = set()
    for port in netlist.get("ports", {}):
        nodes |= nx.descendants(graph, port)
    return set(graph.nodes) - nodes


def _get_connectivity_netlist(netlist: sax.Netlist) -> dict:
    return {
        "instances": natsorted(netlist["instances"]),
        "connections": [
            (c1.split(",")[0], c2.split(",")[0])
            for c1, c2 in netlist.get("connections", {}).items()
        ]
        + [
            (n["p1"].split(",")[0], n["p2"].split(",")[0])
            for n in netlist.get("nets", [])
        ],
        "ports": [(p, c.split(",")[0]) for p, c in netlist.get("ports", {}).items()],
    }


def _get_connectivity_graph(netlist: sax.Netlist) -> nx.Graph:
    graph = nx.Graph()
    connectivity_netlist = _get_connectivity_netlist(netlist)
    for name in connectivity_netlist["instances"]:
        graph.add_node(name)
    for c1, c2 in connectivity_netlist["connections"]:
        graph.add_edge(c1, c2)
    for c1, c2 in connectivity_netlist["ports"]:
        graph.add_edge(c1, c2)
    return graph


def _flatten_netlist_into(  # noqa: PLR0912,C901
    recnet: sax.RecursiveNetlist, net: sax.Netlist, sep: str
) -> None:
    for name, instance in list(net["instances"].items()):
        component = instance["component"]
        if component not in recnet:
            continue
        del net["instances"][name]
        child_net = deepcopy(recnet[component])
        _flatten_netlist_into(recnet, child_net, sep)
        for iname, iinstance in child_net["instances"].items():
            net["instances"][f"{name}{sep}{iname}"] = iinstance
        ports = {k: f"{name}{sep}{v}" for k, v in child_net.get("ports", {}).items()}
        net["connections"] = net.get("connections", {})
        for ip1, ip2 in list(net["connections"].items()):
            n1, p1 = ip1.split(",")
            n2, p2 = ip2.split(",")
            if n1 == name:
                del net["connections"][ip1]
                if p1 not in ports:
                    warnings.warn(
                        f"Port {ip1} not found. Connection {ip1}<->{ip2} ignored.",
                        stacklevel=2,
                    )
                    continue
                net["connections"][ports[p1]] = ip2
            elif n2 == name:
                if p2 not in ports:
                    warnings.warn(
                        f"Port {ip2} not found. Connection {ip1}<->{ip2} ignored.",
                        stacklevel=2,
                    )
                    continue
                net["connections"][ip1] = ports[p2]
        child_net["connections"] = child_net.get("connections", {})
        for ip1, ip2 in child_net["connections"].items():
            net["connections"][f"{name}{sep}{ip1}"] = f"{name}{sep}{ip2}"
        for p, ip2 in list(net.get("ports", {}).items()):
            try:
                n2, p2 = ip2.split(",")
            except ValueError:
                warnings.warn(
                    f"Unconventional port definition ignored: {p}->{ip2}.", stacklevel=2
                )
                continue
            if n2 == name:
                if p2 in ports:
                    net.get("ports", {})[p] = ports[p2]
                else:
                    del net.get("ports", {})[p]


def _connections_to_nets(connections: sax.Connections) -> sax.Nets:
    return [{"p1": k, "p2": v} for k, v in connections.items()]


def _nets_to_connections_strict(nets: sax.Nets) -> sax.Connections:
    connections: sax.Connections = {}
    seen: set[str] = set()
    for net in nets:
        for p in (net["p1"], net["p2"]):
            if p in seen:
                msg = (
                    "Multiply connected ports are only supported with the 'klu' "
                    f"backend. Port {p!r} appears in multiple connections."
                )
                raise ValueError(msg)
            seen.add(p)
        connections[net["p1"]] = net["p2"]
    return connections


@overload
def convert_nets_to_connections(
    netlist: sax.Netlist,
) -> sax.Netlist: ...


@overload
def convert_nets_to_connections(
    netlist: sax.RecursiveNetlist,
) -> sax.RecursiveNetlist: ...


def convert_nets_to_connections(
    netlist: sax.AnyNetlist,
) -> sax.AnyNetlist:
    if (recnet := sax.try_into[sax.RecursiveNetlist](netlist)) is not None:
        return {k: convert_nets_to_connections(v) for k, v in recnet.items()}
    net: sax.Netlist = sax.into[sax.Netlist](netlist)
    nets = net.pop("nets", [])
    connections = net.get("connections", {})
    connections = _nets_to_connections(nets, connections)
    net["connections"] = connections
    return net


def _nets_to_connections(
    nets: sax.Nets, connections: sax.Connections
) -> sax.Connections:
    connections = dict(connections.items())
    inverse_connections = {v: k for k, v in connections.items()}

    def _is_connected(p: sax.InstancePort) -> bool:
        return (p in connections) or (p in inverse_connections)

    def _add_connection(p: sax.InstancePort, q: sax.InstancePort) -> None:
        connections[p] = q
        inverse_connections[q] = p

    def _get_connected_port(p: sax.InstancePort) -> sax.InstancePort:
        if p in connections:
            return connections[p]
        return inverse_connections[p]

    for net in nets:
        p = net["p1"]
        q = net["p2"]
        if _is_connected(p):
            _q = _get_connected_port(p)
            msg = (
                "SAX currently does not support multiply connected ports. "
                f"Got {p}<->{q} and {p}<->{_q}"
            )
            raise ValueError(
                msg,
            )
        if _is_connected(q):
            _p = _get_connected_port(q)
            msg = (
                "SAX currently does not support multiply connected ports. "
                f"Got {p}<->{q} and {_p}<->{q}"
            )
            raise ValueError(
                msg,
            )
        _add_connection(p, q)
    return connections


@overload
def expand_probes(
    netlist: sax.Netlist,
    probes: dict[str, str],
) -> sax.Netlist: ...


@overload
def expand_probes(
    netlist: sax.RecursiveNetlist,
    probes: dict[str, str],
) -> sax.RecursiveNetlist: ...


def expand_probes(  # noqa: PLR0912,PLR0915,C901
    netlist: sax.AnyNetlist,
    probes: dict[str, str],
) -> sax.AnyNetlist:
    """Expand probes into ideal_probe instances and update connections/ports.

    This function inserts ideal 4-port measurement probes at specified connection
    points in the netlist. Each probe intercepts a connection and exposes forward
    and backward traveling wave ports.

    Probes can target instance ports at any level of a recursive netlist using
    dot-separated paths. For example, ``"sub1.wg1,out0"`` targets port ``out0``
    of instance ``wg1`` inside the component used by instance ``sub1``.

    Args:
        netlist: The netlist to expand probes in.
        probes: A mapping from probe names to instance ports where probes should
            be inserted. Instance ports can use dot-separated paths to target
            sub-circuits (e.g. ``"sub1.wg1,out0"``). If the instance port is
            part of an existing connection, a 4-port probe is inserted. If the
            instance port is unconnected, only the "X_fwd" port is created as a
            direct alias.

    Returns:
        A new netlist with probe instances inserted and connections/ports updated.
        For probes on connected ports, two new ports are added: "X_fwd" and "X_bwd".
        For probes on unconnected ports, only "X_fwd" is added.

    Raises:
        ValueError: If probe ports would conflict with existing ports, or if a
            hierarchical path references a nonexistent instance or component.

    Example:
        ```python
        netlist = {
            "instances": {"wg1": "waveguide", "wg2": "waveguide"},
            "connections": {"wg1,out": "wg2,in"},
            "ports": {"in": "wg1,in", "out": "wg2,out"},
        }
        probes = {"mid": "wg1,out"}
        expanded = expand_probes(netlist, probes)
        # expanded now has instances: wg1, wg2, _probe_mid
        # expanded now has ports: in, out, mid_fwd, mid_bwd
        ```
    """
    if not probes:
        return netlist

    # Handle recursive netlist
    if (recnet := sax.try_into[sax.RecursiveNetlist](netlist)) is not None:
        return _expand_probes_recursive(recnet, probes)

    # It's a flat netlist
    net: sax.Netlist = deepcopy(sax.into[sax.Netlist](netlist))
    connections = dict(net.get("connections", {}).items())
    inverse_connections = {v: k for k, v in connections.items()}
    nets = list(net.get("nets", []))
    ports = dict(net.get("ports", {}).items())
    instances = dict(net["instances"].items())

    for probe_name, instance_port in probes.items():
        # Validate probe name won't conflict with existing ports
        fwd_port = f"{probe_name}_fwd"
        bwd_port = f"{probe_name}_bwd"
        if fwd_port in ports or bwd_port in ports:
            msg = (
                f"Probe '{probe_name}' would create ports '{fwd_port}'/'{bwd_port}' "
                f"which conflict with existing ports."
            )
            raise ValueError(msg)

        # Validate instance name won't conflict
        probe_instance_name = f"_probe_{probe_name}"
        if probe_instance_name in instances:
            msg = (
                f"Probe instance name '{probe_instance_name}' conflicts with "
                f"an existing instance."
            )
            raise ValueError(msg)

        # Find the connection containing this instance port.
        # Track orientation so _fwd means "signal flowing INTO instance_port"
        # Check connections dict first, then nets list.
        in_side = None
        out_side = None

        if instance_port in connections:
            # instance_port is on the left side: instance_port -> other_port
            other_port = connections[instance_port]
            del connections[instance_port]
            del inverse_connections[other_port]
            in_side = other_port
            out_side = instance_port
        elif instance_port in inverse_connections:
            # instance_port is on the right side: other_port -> instance_port
            other_port = inverse_connections[instance_port]
            del connections[other_port]
            del inverse_connections[instance_port]
            in_side = other_port
            out_side = instance_port
        else:
            # Check nets list
            for i, n in enumerate(nets):
                if n["p1"] == instance_port:
                    other_port = n["p2"]
                    nets.pop(i)
                    in_side = other_port
                    out_side = instance_port
                    break
                if n["p2"] == instance_port:
                    other_port = n["p1"]
                    nets.pop(i)
                    in_side = other_port
                    out_side = instance_port
                    break

        if in_side is None or out_side is None:
            # instance_port is not in any connection or net.
            # Check if it's exposed as a component port (boundary case).
            port_name = None
            for pn, ip in ports.items():
                if ip == instance_port:
                    port_name = pn
                    break

            if port_name is not None:
                # Rewrite the existing port to enter through the probe
                in_side = f"{probe_instance_name},in"
                ports[port_name] = in_side
            # else: truly unconnected — probe,in will be dangling

            out_side = instance_port

        # Insert probe instance
        instances[probe_instance_name] = {"component": "_ideal_probe"}

        # Create new connections: in_side -> probe,in and probe,out -> out_side
        # This orients the probe so _fwd captures signal flowing INTO instance_port
        if in_side is not None and in_side != f"{probe_instance_name},in":
            connections[in_side] = f"{probe_instance_name},in"
            inverse_connections[f"{probe_instance_name},in"] = in_side
        connections[f"{probe_instance_name},out"] = out_side
        inverse_connections[out_side] = f"{probe_instance_name},out"

        # Add probe ports
        ports[fwd_port] = f"{probe_instance_name},tap_fwd"
        ports[bwd_port] = f"{probe_instance_name},tap_bwd"

    net["instances"] = instances
    net["connections"] = connections
    net["nets"] = nets
    net["ports"] = ports
    return net


def _expand_probes_recursive(  # noqa: C901
    netlist: sax.RecursiveNetlist,
    probes: dict[str, str],
) -> sax.RecursiveNetlist:
    """Expand probes in a recursive netlist, supporting hierarchical paths."""
    recnet = deepcopy(netlist)
    top_level_name = next(iter(recnet))

    # Separate top-level probes from hierarchical probes
    top_level_probes: dict[str, str] = {}
    # Group hierarchical probes by target component
    # Maps: target_component -> {probe_name: instance_port}
    component_probes: dict[str, dict[str, str]] = {}
    # Maps: probe_name -> [(instance_name, parent_component), ...]
    probe_paths: dict[str, list[tuple[str, str]]] = {}

    for probe_name, instance_port in probes.items():
        parts = instance_port.split(".")
        if len(parts) == 1:
            top_level_probes[probe_name] = instance_port
        else:
            # Walk the hierarchy to find the target component
            current_component = top_level_name
            path: list[tuple[str, str]] = []
            for instance_name in parts[:-1]:
                current_netlist = recnet[current_component]
                instances = current_netlist.get("instances", {})
                if instance_name not in instances:
                    msg = (
                        f"Hierarchical probe '{probe_name}': instance "
                        f"'{instance_name}' not found in component "
                        f"'{current_component}'. "
                        f"Available instances: {list(instances.keys())}"
                    )
                    raise ValueError(msg)
                inst = instances[instance_name]
                child_component = inst["component"] if isinstance(inst, dict) else inst
                if child_component not in recnet:
                    msg = (
                        f"Hierarchical probe '{probe_name}': component "
                        f"'{child_component}' (used by instance "
                        f"'{instance_name}' in '{current_component}') "
                        f"is not defined in the recursive netlist. "
                        f"Only sub-circuits (not primitives) can be probed."
                    )
                    raise ValueError(msg)
                path.append((instance_name, current_component))
                current_component = child_component

            target_instance_port = parts[-1]
            if current_component not in component_probes:
                component_probes[current_component] = {}
            component_probes[current_component][probe_name] = target_instance_port
            probe_paths[probe_name] = path

    # Expand top-level probes using existing flat-netlist logic
    if top_level_probes:
        recnet[top_level_name] = expand_probes(recnet[top_level_name], top_level_probes)

    # Expand hierarchical probes at their target component level
    for component_name, comp_probes in component_probes.items():
        recnet[component_name] = expand_probes(recnet[component_name], comp_probes)

    # Bubble up: expose probe ports through each intermediate level
    for probe_name, path in probe_paths.items():
        fwd_port = f"{probe_name}_fwd"
        bwd_port = f"{probe_name}_bwd"
        for instance_name, parent_component in reversed(path):
            parent_ports = recnet[parent_component].get("ports", {})
            if fwd_port in parent_ports or bwd_port in parent_ports:
                msg = (
                    f"Hierarchical probe '{probe_name}' would create ports "
                    f"'{fwd_port}'/'{bwd_port}' which conflict with existing "
                    f"ports in component '{parent_component}'."
                )
                raise ValueError(msg)
            parent_ports[fwd_port] = f"{instance_name},{fwd_port}"
            parent_ports[bwd_port] = f"{instance_name},{bwd_port}"

    return recnet


@overload
def extract_port_probes(
    netlist: sax.Netlist,
    on_internal_port: Literal["warn", "ignore", "as_probes"] = "warn",
) -> tuple[sax.Netlist, dict[str, str]]: ...


@overload
def extract_port_probes(
    netlist: sax.RecursiveNetlist,
    on_internal_port: Literal["warn", "ignore", "as_probes"] = "warn",
) -> tuple[sax.RecursiveNetlist, dict[str, str]]: ...


def extract_port_probes(
    netlist: sax.AnyNetlist,
    on_internal_port: Literal["warn", "ignore", "as_probes"] = "warn",
) -> tuple[sax.AnyNetlist, dict[str, str]]:
    """Handle ports that map to internal connection nodes.

    When a netlist port maps to an instance port that is already part of a
    connection (in ``connections`` or ``nets``), the behaviour depends on
    *on_internal_port*:

    * ``"warn"`` (default) — drop the port and emit a warning.
    * ``"ignore"`` — drop the port silently.
    * ``"as_probes"`` — convert the port to a probe (creating ``_fwd`` /
      ``_bwd`` ports), matching the legacy behaviour.

    Args:
        netlist: The netlist to inspect (flat or recursive).
        on_internal_port: How to handle ports on internal nodes.

    Returns:
        A tuple of (modified_netlist, probes) where *probes* maps original
        port names to instance port strings.  The dict is empty unless
        *on_internal_port* is ``"as_probes"``.
    """
    # Handle recursive netlist: only inspect the top-level
    if (recnet := sax.try_into[sax.RecursiveNetlist](netlist)) is not None:
        top_level_name = next(iter(recnet))
        top_level = recnet[top_level_name]
        modified_top, probes = extract_port_probes(top_level, on_internal_port)
        result: sax.RecursiveNetlist = {
            top_level_name: modified_top,
            **{k: v for k, v in recnet.items() if k != top_level_name},
        }
        return result, probes

    # Flat netlist
    net: sax.Netlist = deepcopy(sax.into[sax.Netlist](netlist))
    connections = net.get("connections", {})
    nets = net.get("nets", [])
    ports = dict(net.get("ports", {}).items())

    # Collect all instance ports that are internal (part of a connection)
    internal_ports: set[str] = set()
    for left, right in connections.items():
        internal_ports.add(left)
        internal_ports.add(right)
    for n in nets:
        internal_ports.add(n["p1"])
        internal_ports.add(n["p2"])

    probes: dict[str, str] = {}
    for port_name, instance_port in list(ports.items()):
        if instance_port in internal_ports:
            if on_internal_port == "as_probes":
                warnings.warn(
                    f"Port '{port_name}' maps to internal node "
                    f"'{instance_port}' which is already part of a "
                    f"connection. It will be interpreted as a probe "
                    f"(creating '{port_name}_fwd' and "
                    f"'{port_name}_bwd' ports).",
                    stacklevel=2,
                )
                probes[port_name] = instance_port
            elif on_internal_port == "warn":
                warnings.warn(
                    f"Port '{port_name}' maps to internal node "
                    f"'{instance_port}' which is already part of a "
                    f"connection. It will be dropped. Use the probes= "
                    f"argument of circuit() to explicitly create "
                    f"measurement probes.",
                    stacklevel=2,
                )
            # "ignore" falls through — just delete silently
            del ports[port_name]

    net["ports"] = ports
    return net, probes
