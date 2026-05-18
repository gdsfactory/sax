"""Parse kfnetlist netlists into SAX format.

Converts kfnetlist's Netlist objects (or their dict/JSON representations) into
SAX Netlist and RecursiveNetlist dictionaries suitable for ``sax.circuit()``.
"""

from __future__ import annotations

import json
from typing import Any

import sax

__all__ = [
    "parse_kfnetlist",
    "parse_kfnetlist_recursive",
]


def parse_kfnetlist(
    netlist: Any,
    *,
    top_level_name: str = "top_level",
) -> sax.RecursiveNetlist:
    """Convert a kfnetlist Netlist (or its dict/JSON form) into a SAX RecursiveNetlist.

    Accepts a kfnetlist ``Netlist`` object, a dict from ``Netlist.to_dict()``,
    or a JSON string from ``Netlist.to_json()``.

    Args:
        netlist: A kfnetlist Netlist object, dict, or JSON string.
        top_level_name: Name for the top-level circuit entry.

    Returns:
        A SAX RecursiveNetlist (single entry keyed by *top_level_name*).

    Example:
        ```python
        from kfnetlist import Netlist
        import sax

        kf_nl = Netlist.from_json(json_string)
        recnet = sax.parse_kfnetlist(kf_nl)
        circuit_fn, info = sax.circuit(recnet, models={...})
        ```
    """
    d = _to_dict(netlist)
    sax_netlist = _convert_flat(d)
    return {top_level_name: sax_netlist}


def parse_kfnetlist_recursive(
    netlists: dict[str, Any] | str,
) -> sax.RecursiveNetlist:
    """Convert a dict of kfnetlist Netlists into a SAX RecursiveNetlist.

    This is the natural entry point for hierarchical designs produced by
    ``kfnetlist.extract.extract()``, which returns ``dict[str, Netlist]``.

    Args:
        netlists: Mapping of cell names to kfnetlist Netlist objects/dicts,
            or a JSON string encoding such a mapping.

    Returns:
        A SAX RecursiveNetlist.

    Example:
        ```python
        from kfnetlist.extract import extract
        import sax

        cell_netlists = extract(top_cell)
        recnet = sax.parse_kfnetlist_recursive(cell_netlists)
        circuit_fn, info = sax.circuit(recnet, models={...})
        ```
    """
    if isinstance(netlists, str):
        netlists = json.loads(netlists)

    result: sax.RecursiveNetlist = {}
    for name, nl in netlists.items():
        d = _to_dict(nl)
        result[name] = _convert_flat(d)
    return result


def _to_dict(netlist: Any) -> dict:
    """Normalise *netlist* into a plain dict."""
    if isinstance(netlist, str):
        return json.loads(netlist)
    if isinstance(netlist, dict):
        return netlist
    if hasattr(netlist, "to_dict"):
        return netlist.to_dict()
    msg = f"Cannot convert {type(netlist)} to a kfnetlist dict."
    raise TypeError(msg)


def _format_port_ref(member: dict, array_instances: set[str]) -> str:
    """Format a kfnetlist PortRef/PortArrayRef dict as a SAX InstancePort string."""
    instance = member["instance"]
    port = member["port"]
    if "ia" in member and "ib" in member:
        ia = int(member["ia"]) - 1
        ib = int(member["ib"]) - 1
        return f"{instance}<{ia}.{ib}>,{port}"
    if instance in array_instances:
        return f"{instance}<0.0>,{port}"
    return f"{instance},{port}"


def _convert_instance(inst: dict) -> sax.Instance:
    """Convert a kfnetlist instance dict to a SAX Instance."""
    result: sax.Instance = {"component": inst["component"]}
    settings = inst.get("settings")
    if settings:
        result["settings"] = dict(settings)
    array = inst.get("array")
    if array is not None:
        na = int(array.get("na", 1))
        nb = int(array.get("nb", 1))
        if na > 1 or nb > 1:
            result["array"] = {"columns": na, "rows": nb}
    return result


def _is_port(member: dict) -> bool:
    return "name" in member and "instance" not in member


def _convert_flat(d: dict) -> sax.Netlist:
    """Convert a single kfnetlist dict into a SAX flat Netlist."""
    instances: sax.Instances = {}
    array_instances: set[str] = set()
    for name, inst in d.get("instances", {}).items():
        instances[name] = _convert_instance(inst)
        array = inst.get("array")
        if array is not None:
            na = int(array.get("na", 1))
            nb = int(array.get("nb", 1))
            if na > 1 or nb > 1:
                array_instances.add(name)

    connections: sax.Connections = {}
    ports: sax.Ports = {}

    def _fmt(member: dict) -> str:
        return _format_port_ref(member, array_instances)

    for net_members in d.get("nets", []):
        ext_ports = [m for m in net_members if _is_port(m)]
        inst_ports = [m for m in net_members if not _is_port(m)]

        if len(ext_ports) == 1 and len(inst_ports) == 1:
            ports[ext_ports[0]["name"]] = _fmt(inst_ports[0])

        elif len(ext_ports) == 0 and len(inst_ports) == 2:
            connections[_fmt(inst_ports[0])] = _fmt(inst_ports[1])

        elif len(ext_ports) == 0 and len(inst_ports) > 2:
            for i in range(len(inst_ports) - 1):
                connections[_fmt(inst_ports[i])] = _fmt(inst_ports[i + 1])

        elif len(ext_ports) >= 1 and len(inst_ports) >= 1:
            for ep in ext_ports:
                ports[ep["name"]] = _fmt(inst_ports[0])
            for i in range(len(inst_ports) - 1):
                connections[_fmt(inst_ports[i])] = _fmt(inst_ports[i + 1])

    result: sax.Netlist = {"instances": instances}
    if connections:
        result["connections"] = connections
    if ports:
        result["ports"] = ports
    return result
