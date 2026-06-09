"""Read a Mosaic schematic as sax netlist."""

from itertools import combinations
from pathlib import Path

import yaml

import sax


def parse_mosaic(  # noqa: C901
    nyancir: dict | Path | str,
    *,
    models: sax.Models | None = None,
) -> sax.Netlist:
    """Parse mosaic schematic."""
    netlist: sax.Netlist = {
        "instances": {},
        "nets": [],
        "ports": {},
    }
    nets = {}
    nyandct = _to_dict(nyancir)
    name_map = _name_mapping(nyandct)
    comp_map = _get_component_map()
    models = models or {}

    for nyanname, nyanobj in nyandct.items():
        clean_nyanname = sax.clean_string(nyanname)
        name = name_map.get(clean_nyanname, clean_nyanname)
        nyantype = nyanobj.get("type", "")
        if nyantype in ["", "port"]:
            continue  # TODO: do something better?
        component_fqn = nyanobj.get("model", "")
        if not component_fqn:
            continue
        if component_fqn in models:
            component = component_fqn
        else:
            component = _to_modelname(comp_map, component_fqn)
        settings = nyanobj.get("props", {})
        netlist["instances"][name] = {
            "component": component,
            "settings": settings,
        }
        for p, net in nyanobj.get("nets", {}).items():
            if net not in nets:
                nets[net] = []
            nets[net].append(f"{name},{p}")

    for net_name, ports in nets.items():
        len_ports = len(ports)
        if len_ports == 0:
            continue
        if len_ports == 1:
            netlist["ports"][net_name] = ports[0]
            continue

        for p1, p2 in combinations(ports, 2):
            netlist["nets"].append({"p1": p1, "p2": p2})
    return netlist


def _to_dict(nyancir: dict | Path | str) -> dict:
    if isinstance(nyancir, dict):
        return nyancir
    if isinstance(nyancir, str) and "\n" in nyancir:
        return yaml.safe_load(nyancir)
    path = Path(nyancir).resolve()
    content = path.read_text()
    return yaml.safe_load(content)


def _name_mapping(nyancir: dict) -> dict[str, str]:
    name_map1: dict[str, list[str]] = {}
    for nyanname, nyanobj in nyancir.items():
        if not nyanname:
            continue
        if nyanname not in name_map1:
            name_map1[nyanname] = []
        name = nyanobj.get("name", "")
        if not name:
            name = sax.clean_string(nyanname)
        name_map1[nyanname].append(name)

    name_map: dict[str, str] = {}
    for nyanname, names in name_map1.items():
        len_names = len(names)
        if len_names == 0:
            continue
        if len_names == 1:
            name_map[sax.clean_string(nyanname)] = sax.clean_string(names[0])
        for name in names:
            clean_name = sax.clean_string(name)
            clean_nyanname = sax.clean_string(nyanname)
            name_map[clean_nyanname] = clean_name
    return name_map


def _to_modelname(component_map: dict[str, str], component_fqn: str) -> str:
    fallback = component_fqn.rsplit(".", maxsplit=1)[-1]
    return component_map.get(component_fqn, fallback)


def _get_component_map() -> dict[str, str]:
    try:
        import gdsfactory as gf

        pdk = gf.get_active_pdk()
    except Exception:  # noqa: BLE001
        return {}
    component_map: dict[str, str] = {}
    for k, v in pdk.cells.items():
        qualname = getattr(v, "__qualname__", "")
        if not qualname:
            continue
        component_map[qualname] = k
    return component_map
