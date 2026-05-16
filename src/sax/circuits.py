"""SAX Circuit Definition."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Iterator
from functools import partial
from typing import Any, Literal, cast, overload

import networkx as nx
import numpy as np

import sax

from .backends import circuit_backends
from .models.probes import ideal_probe
from .netlists import (
    _connections_to_nets,
    expand_probes,
    extract_port_probes,
    remove_unused_instances,
)
from .netlists import netlist as into_recnet
from .s import get_ports, scoo, sdense, sdict
from .utils import get_settings, merge_dicts, replace_kwargs, update_settings

__all__ = ["circuit", "draw_dag", "get_required_circuit_models"]


def _filter_portless_subnets(
    netlist: sax.RecursiveNetlist,
) -> sax.RecursiveNetlist:
    """Remove sub-netlists that have no ports (except the top-level entry)."""
    top_level_name = next(iter(netlist))
    return {
        name: flatnet
        for name, flatnet in netlist.items()
        if name == top_level_name or flatnet.get("ports", {})
    }


@overload
def circuit(
    netlist: dict[str, Any],
    models: sax.Models | None = None,
    *,
    backend: sax.BackendLike = "default",
    top_level_name: str = "top_level",
    ignore_impossible_connections: bool = False,
    probes: dict[str, str] | None = None,
    on_internal_port: Literal["warn", "ignore", "as_probes"] = "warn",
) -> tuple[sax.SDictModel, sax.CircuitInfo]: ...


@overload
def circuit(
    netlist: dict[str, Any],
    models: sax.Models | None = None,
    *,
    backend: sax.BackendLike = "default",
    return_type: Literal["SDict"],
    top_level_name: str = "top_level",
    ignore_impossible_connections: bool = False,
    probes: dict[str, str] | None = None,
    on_internal_port: Literal["warn", "ignore", "as_probes"] = "warn",
) -> tuple[sax.SDictModel, sax.CircuitInfo]: ...


@overload
def circuit(
    netlist: dict[str, Any],
    models: sax.Models | None = None,
    *,
    backend: sax.BackendLike = "default",
    return_type: Literal["SDense"],
    top_level_name: str = "top_level",
    ignore_impossible_connections: bool = False,
    probes: dict[str, str] | None = None,
    on_internal_port: Literal["warn", "ignore", "as_probes"] = "warn",
) -> tuple[sax.SDenseModel, sax.CircuitInfo]: ...


@overload
def circuit(
    netlist: dict[str, Any],
    models: sax.Models | None = None,
    *,
    backend: sax.BackendLike = "default",
    return_type: Literal["SCoo"],
    top_level_name: str = "top_level",
    ignore_impossible_connections: bool = False,
    probes: dict[str, str] | None = None,
    on_internal_port: Literal["warn", "ignore", "as_probes"] = "warn",
) -> tuple[sax.SCooModel, sax.CircuitInfo]: ...


def circuit(
    netlist: dict[str, Any],
    models: sax.Models | None = None,
    *,
    backend: sax.BackendLike = "default",
    return_type: Literal["SDict", "SDense", "SCoo"] = "SDict",
    top_level_name: str = "top_level",
    ignore_impossible_connections: bool = False,
    probes: dict[str, str] | None = None,
    on_internal_port: Literal["warn", "ignore", "as_probes"] = "warn",
) -> tuple[sax.Model, sax.CircuitInfo]:
    """Create a circuit function for a given netlist.

    Constructs a circuit model from a netlist description by connecting component
    models according to the specified connections. The resulting circuit function
    can be called with parameters to evaluate the overall S-matrix.

    Args:
        netlist: Circuit netlist specifying instances, connections, and ports.
            Can be a flat netlist or recursive netlist dictionary.
        models: Dictionary mapping component names to their model functions.
            If None, models must be provided in the netlist itself.
        backend: Circuit analysis backend to use. Options include "default",
            "klu", "filipsson_gunnar", "additive", "forward". Defaults to "default".
        return_type: Format of the returned S-matrix. Options: "SDict", "SDense",
            "SCoo". Defaults to "SDict".
        top_level_name: Name of the top-level circuit in recursive netlists.
            Defaults to "top_level".
        ignore_impossible_connections: If True, ignore connections to missing
            instance ports instead of raising an error. Defaults to False.
        probes: Optional dictionary mapping probe names to instance ports where
            measurement probes should be inserted. Each probe intercepts a
            connection and exposes forward and backward traveling wave ports.
            For a probe named "X" at instance port "inst,port", two new circuit
            ports are created: "X_fwd" and "X_bwd". Defaults to None.
        on_internal_port: How to handle top-level ports that map to internal
            connection nodes. ``"warn"`` (default) drops them with a warning,
            ``"ignore"`` drops them silently, ``"as_probes"`` converts them
            to measurement probes (legacy behaviour).

    Returns:
        Tuple containing:
            - Circuit model function that accepts parameters and returns S-matrix
            - CircuitInfo object with DAG, models, and backend information

    Raises:
        RuntimeError: If circuit construction fails.
        ValueError: If netlist or models are invalid.
        KeyError: If required models are missing.

    Example:
        ```python
        # Define component models
        def waveguide(length=10.0, neff=2.4, wl=1.55):
            phase = 2 * np.pi * neff * length / wl
            return {("in", "out"): np.exp(1j * phase)}


        # Create netlist
        netlist = {
            "instances": {
                "wg1": {"component": "waveguide", "settings": {"length": 20.0}}
            },
            "connections": {},
            "ports": {"in": "wg1,in", "out": "wg1,out"},
        }

        # Create circuit
        models = {"waveguide": waveguide}
        circuit_func, info = circuit(netlist, models)

        # Evaluate circuit
        s_matrix = circuit_func(wl=1.55)
        ```
    """
    _backend = sax.into[sax.Backend](backend)

    instance_models = _extract_instance_models(netlist)
    recnet = into_recnet(
        netlist,
        top_level_name=top_level_name,
    )
    patch_netlist_array_instances(recnet)
    recnet = sax.into[sax.RecursiveNetlist](recnet)
    recnet, auto_probes = extract_port_probes(recnet, on_internal_port)
    if auto_probes:
        probes = {**(probes or {}), **auto_probes}
    if probes:
        recnet = expand_probes(recnet, probes)
        models = {"_ideal_probe": ideal_probe, **(models or {})}
    recnet = resolve_array_instances(recnet)
    recnet = _filter_portless_subnets(recnet)
    recnet = remove_unused_instances(recnet)
    _validate_netlist_ports(recnet)
    dependency_dag = _create_dag(recnet, models, validate=True)
    models = _validate_models(
        models or {}, dependency_dag, extra_models=instance_models
    )

    circuit = None
    new_models = {}
    current_models = {}
    model_names = list(nx.topological_sort(dependency_dag))[::-1]
    for model_name in model_names:
        if model_name in models:
            new_models[model_name] = models[model_name]
            continue

        flatnet = recnet[model_name]
        current_models |= new_models
        new_models = {}

        current_models[model_name] = circuit = _flat_circuit(
            flatnet["instances"],
            flatnet.get("connections", {}),
            flatnet.get("nets", []),
            flatnet.get("ports", {}),
            flatnet.get("placements", {}),
            current_models,
            _backend,
            ignore_impossible_connections=ignore_impossible_connections,
        )

    if circuit is None:
        msg = "Could not construct circuit (unknown reason)"
        raise RuntimeError(msg)
    circuit = _enforce_return_type(circuit, return_type)
    return circuit, sax.CircuitInfo(
        dag=dependency_dag,
        models=current_models,
        backend=_backend,
    )


def _create_dag(
    netlist: sax.RecursiveNetlist,
    models: sax.Models | None = None,
    *,
    validate: bool = False,
) -> nx.DiGraph:
    if models is None:
        models = {}

    all_models = {}
    g = nx.DiGraph()

    for model_name, subnetlist in netlist.items():
        if model_name not in all_models:
            all_models[model_name] = models.get(model_name, subnetlist)
            g.add_node(model_name)
        if model_name in models:
            continue
        for instance in subnetlist["instances"].values():
            component = instance["component"]
            if component not in all_models:
                all_models[component] = models.get(component)
                g.add_node(component)
            g.add_edge(model_name, component)

    # we only need the nodes that depend on the parent...
    parent_node = next(iter(netlist.keys()))
    nodes = [parent_node, *nx.descendants(g, parent_node)]
    g = cast(nx.DiGraph, nx.induced_subgraph(g, nodes))
    if validate:
        g = _validate_dag(g)
    return g


def draw_dag(dag: nx.DiGraph, *, with_labels: bool = True, **kwargs: Any) -> None:  # noqa: ANN401
    """Draw a directed acyclic graph (DAG) representing circuit dependencies.

    Visualizes the dependency graph of a circuit using networkx. If pydot/graphviz
    is available, uses hierarchical layout; otherwise falls back to a custom layout.

    Args:
        dag: Directed acyclic graph representing circuit component dependencies.
        with_labels: Whether to display node labels. Defaults to True.
        **kwargs: Additional keyword arguments passed to networkx draw function.

    Example:
        ```python
        import matplotlib.pyplot as plt

        # Assuming you have a circuit with dependencies
        _, info = circuit(netlist, models)
        draw_dag(info.dag)
        plt.show()
        ```
    """
    if shutil.which("dot"):
        return nx.draw(
            dag,
            nx.nx_pydot.pydot_layout(dag, prog="dot"),
            with_labels=with_labels,
            **kwargs,
        )
    return nx.draw(dag, _my_dag_pos(dag), with_labels=with_labels, **kwargs)


def get_required_circuit_models(
    netlist: sax.AnyNetlist,
    models: dict[str, sax.Model] | None = None,
) -> list[str]:
    """Determine which component models are required for a given netlist.

    Analyzes a netlist to identify all component types that need model functions.
    This is useful for validating that all required models are available before
    circuit construction.

    Args:
        netlist: Circuit netlist to analyze for component dependencies.
        models: Optional dictionary of available models. Used to filter out
            models that are already provided.

    Returns:
        List of component names that require model functions.

    Example:
        ```python
        netlist = {
            "instances": {
                "wg1": {"component": "waveguide"},
                "dc1": {"component": "directional_coupler"},
            },
            "ports": {"in": "wg1,in", "out": "dc1,out"},
        }
        required = get_required_circuit_models(netlist)
        # Result: ["waveguide", "directional_coupler"]

        # With some models already available
        models = {"waveguide": my_waveguide_model}
        required = get_required_circuit_models(netlist, models)
        # Result: ["directional_coupler"]
        ```
    """
    instance_models = _extract_instance_models(netlist)
    recnet = into_recnet(
        netlist,
    )
    recnet = remove_unused_instances(recnet)
    dependency_dag = _create_dag(recnet, models, validate=True)
    _, required, _ = _find_missing_models(
        models,
        dependency_dag,
        extra_models=instance_models,
    )
    return required


def _flat_circuit(
    instances: sax.Instances,
    connections: sax.Connections,
    nets: sax.Nets,
    ports: sax.Ports,
    placements: sax.Placements,
    models: sax.Models,
    backend: sax.Backend,
    *,
    ignore_impossible_connections: bool = False,
) -> sax.Model:
    analyze_insts_fn, analyze_fn, evaluate_fn = circuit_backends[backend]
    dummy_instances = analyze_insts_fn(instances, models)
    inst_port_mode = {
        k: _port_modes_dict(get_ports(s)) for k, s in dummy_instances.items()
    }
    all_nets = _connections_to_nets(connections) + list(nets)
    all_nets = _get_multimode_nets(
        all_nets,
        inst_port_mode,
        ignore_impossible_connections=ignore_impossible_connections,
    )
    ports = _get_multimode_ports(
        ports,
        inst_port_mode,
        ignore_impossible_connections=ignore_impossible_connections,
    )

    inst2model = {}
    for k, inst in instances.items():
        inst2model[k] = models[inst["component"]]

    model_settings = {name: get_settings(model) for name, model in inst2model.items()}
    netlist_settings = {
        name: {
            k: v
            for k, v in (inst.get("settings") or {}).items()
            if k in model_settings[name]
        }
        for name, inst in instances.items()
    }
    for name, settings in netlist_settings.items():
        if "placement" in settings:
            settings["placement"] = sax.into[sax.Placement](placements.get(name, {}))
    default_settings = merge_dicts(model_settings, netlist_settings)
    default_settings = {_strip_array_index(k): v for k, v in default_settings.items()}
    analyzed = analyze_fn(dummy_instances, all_nets, ports)

    def _circuit(**settings: sax.SettingsValue) -> sax.SType:
        full_settings = merge_dicts(default_settings, settings)
        full_settings = _forward_global_settings(inst2model, full_settings)
        full_settings = merge_dicts(full_settings, settings)

        instances: dict[str, sax.SType] = {}
        for inst_name, model in inst2model.items():
            inst_settings = full_settings.get(_strip_array_index(inst_name), {})
            instances[inst_name] = model(**inst_settings)

        return evaluate_fn(analyzed, instances)

    replace_kwargs(_circuit, **default_settings)

    return cast(sax.Model, _circuit)


def _in_degree(dag: nx.DiGraph) -> Iterator[tuple[str, int]]:
    return cast(Iterator[tuple[str, int]], dag.in_degree())


def _out_degree(dag: nx.DiGraph) -> Iterator[tuple[str, int]]:
    return cast(Iterator[tuple[str, int]], dag.out_degree())


def _my_dag_pos(dag: nx.DiGraph) -> dict:
    # inferior to pydot
    in_degree = {}
    for k, v in _in_degree(dag):
        if v not in in_degree:
            in_degree[v] = []
        in_degree[v].append(k)

    widths = {k: len(vs) for k, vs in in_degree.items()}
    width = max(widths.values())

    horizontal_pos = {
        k: np.linspace(0, 1, w + 2)[1:-1] * width for k, w in widths.items()
    }

    pos = {}
    for k, vs in in_degree.items():
        for x, v in zip(horizontal_pos[k], vs, strict=False):
            pos[v] = (x, -k)
    return pos


def _find_root(g: nx.DiGraph) -> list[str]:
    return [n for n, d in _in_degree(g) if d == 0]


def _find_leaves(g: nx.DiGraph) -> list[str]:
    return [n for n, d in _out_degree(g) if d == 0]


def _find_missing_models(
    models: dict | None,
    dag: nx.DiGraph,
    extra_models: dict | None = None,
) -> tuple[sax.Models, list[str], list[str]]:
    if extra_models is None:
        extra_models = {}
    if models is None:
        models = {}
    models = {**models, **extra_models}
    required_models = _find_leaves(dag)
    missing_models = [m for m in required_models if m not in models]
    return models, required_models, missing_models


def _validate_models(
    models: sax.Models,
    dag: nx.DiGraph,
    extra_models: dict | None = None,
) -> sax.Models:
    models, required_models, missing_models = _find_missing_models(
        models,
        dag,
        extra_models,
    )
    if missing_models:
        model_diff = {
            "Missing Models": missing_models,
            "Given Models": list(models),
            "Required Models": required_models,
        }
        model_diff_str = json.dumps(model_diff, indent=4)
        msg = (
            "Missing models. The following models are still missing to build "
            f"the circuit:\n{model_diff_str}"
        )
        raise ValueError(msg)
    return models


def _forward_global_settings(
    instances: sax.Instances, settings: sax.Settings
) -> sax.Settings:
    global_settings = {
        k: settings.pop(k) for k in list(settings.keys()) if k not in instances
    }
    if global_settings:
        settings = update_settings(settings, **global_settings)
    return settings


def _port_modes_dict(
    port_modes: Iterable[sax.PortMode],
) -> dict[sax.Port, set[sax.Mode]]:
    result = {}
    for port_mode in port_modes:
        port, mode = port_mode.split("@") if "@" in port_mode else (port_mode, None)
        if port not in result:
            result[port] = set()
        if mode is not None:
            result[port].add(mode)
    return result


def _get_multimode_nets(
    nets: sax.Nets,
    inst_port_mode: dict[sax.InstanceName, dict[sax.Port, set[sax.Mode]]],
    *,
    ignore_impossible_connections: bool = False,
) -> sax.Nets:
    mm_nets: sax.Nets = []
    for net in nets:
        inst1, port1 = net["p1"].split(",")
        inst2, port2 = net["p2"].split(",")
        try:
            modes1 = inst_port_mode[inst1][port1]
        except KeyError as e:
            if ignore_impossible_connections:
                continue
            msg = (
                f"Instance {inst1} does not contain port {port1}. "
                f"Available ports: {list(inst_port_mode[inst1])}."
            )
            raise KeyError(msg) from e
        try:
            modes2 = inst_port_mode[inst2][port2]
        except KeyError as e:
            if ignore_impossible_connections:
                continue
            msg = (
                f"Instance {inst2} does not contain port {port2}. "
                f"Available ports: {list(inst_port_mode[inst2])}."
            )
            raise KeyError(msg) from e
        if not modes1 and not modes2:
            mm_nets.append({"p1": net["p1"], "p2": net["p2"]})
        elif (not modes1) or (not modes2):
            msg = (
                "trying to connect a multimode model to single mode model.\n"
                "Please update your models dictionary.\n"
                f"Problematic connection: '{net['p1']}':'{net['p2']}'"
            )
            raise ValueError(msg)
        else:
            common_modes = modes1.intersection(modes2)
            for mode in sorted(common_modes):
                mm_nets.append(
                    {
                        "p1": f"{inst1},{port1}@{mode}",
                        "p2": f"{inst2},{port2}@{mode}",
                    }
                )
    return mm_nets


def _get_multimode_ports(
    ports: sax.Ports,
    inst_port_mode: dict[sax.InstanceName, dict[sax.Port, set[sax.Mode]]],
    *,
    ignore_impossible_connections: bool = False,
) -> sax.Ports:
    mm_ports = {}
    for port, inst_port2 in ports.items():
        inst2, port2 = inst_port2.split(",")
        try:
            modes2 = inst_port_mode[inst2][port2]
        except KeyError as e:
            if ignore_impossible_connections:
                continue
            msg = (
                f"Instance {inst2} does not contain port {port2}. "
                f"Available ports: {list(inst_port_mode[inst2])}"
            )
            raise KeyError(msg) from e
        if not modes2:
            mm_ports[port] = f"{inst2},{port2}"
        else:
            for mode in sorted(modes2):
                mm_ports[f"{port}@{mode}"] = f"{inst2},{port2}@{mode}"
    return mm_ports


def _enforce_return_type(model: sax.Model, return_type: Any) -> sax.Model:  # noqa: ANN401
    stypes = {
        "sdict": sdict,
        "scoo": scoo,
        "sdense": sdense,
        sax.SDict: sdict,
        sax.SDense: sdense,
        sax.SCoo: scoo,
        sax.SDictModel: sdict,
        sax.SDenseModel: sdense,
        sax.SCooModel: scoo,
    }
    if isinstance(return_type, str):
        return_type = return_type.lower()
    stype = stypes.get(return_type)
    if stype is None:
        return model
    return stype(model)


def _extract_instance_models(netlist: sax.AnyNetlist) -> sax.Models:
    if _is_netlist(netlist):
        callable_instances = [f for f in netlist["instances"].values() if callable(f)]
        models = {}
        for f in callable_instances:
            while isinstance(f, partial):
                f = f.func
            models[f.__name__] = f
        return models

    if _is_recursive_netlist(netlist):
        models = {}
        for net in netlist.values():
            models.update(_extract_instance_models(cast(sax.Netlist, net)))
        return models

    return {}


def _is_netlist(netlist: sax.AnyNetlist) -> bool:
    return isinstance(netlist, dict) and "instances" in netlist and "ports" in netlist


def _is_recursive_netlist(netlist: sax.AnyNetlist) -> bool:
    return isinstance(netlist, dict) and not _is_netlist(netlist)


def _validate_dag(dag: nx.DiGraph) -> nx.DiGraph:
    nodes = _find_root(dag)
    if len(nodes) > 1:
        msg = f"Multiple top_levels found in netlist: {nodes}"
        raise ValueError(msg)
    if len(nodes) < 1:
        msg = "Netlist does not contain any nodes."
        raise ValueError(msg)
    if not dag.is_directed():
        msg = "Netlist dependency cycles detected!"
        raise ValueError(msg)
    return dag


def _validate_netlist_ports(netlist: sax.RecursiveNetlist) -> None:
    top_level_name = next(iter(netlist))
    top_level = netlist[top_level_name]
    ports_str = ", ".join(list(top_level.get("ports", {})))
    if not ports_str:
        ports_str = "no ports given"
    if len(top_level.get("ports", {})) < 1:
        msg = (
            "Cannot create circuit: "
            f"at least 1 port needs to be defined. Got {ports_str}."
        )
        raise ValueError(msg)


def _strip_array_index(s: sax.InstanceName) -> sax.Name:
    return s.split("<")[0]


def resolve_array_instance(name: sax.Name, inst: sax.Instance) -> sax.Instances:
    if "array" not in inst:
        return {name: inst}
    ret = {}
    for i in range(inst["array"]["columns"]):
        for j in range(inst["array"]["rows"]):
            ret[f"{name}<{i}.{j}>"] = {"component": inst["component"]}
            if "settings" in inst:
                ret[f"{name}<{i}.{j}>"]["settings"] = inst["settings"]
    return ret


@overload
def resolve_array_instances(netlist: sax.RecursiveNetlist) -> sax.RecursiveNetlist: ...


@overload
def resolve_array_instances(netlist: sax.Netlist) -> sax.Netlist: ...


def resolve_array_instances(netlist: sax.AnyNetlist) -> sax.AnyNetlist:
    if _is_netlist(net := cast(sax.Netlist, netlist)):
        net = {**net}  # shallow copy
        instances = {}
        for name, inst in net["instances"].items():
            instances.update(resolve_array_instance(name, inst))
        net["instances"] = instances
        return net
    if _is_recursive_netlist(recnet := cast(sax.RecursiveNetlist, netlist)):
        return {k: resolve_array_instances(v) for k, v in recnet.items()}
    return netlist


def patch_netlist_array_instances(netlist: sax.RecursiveNetlist) -> None:  # noqa: C901
    """Patch array instances into netlist."""

    def patch_flatnet(flatnet: sax.Netlist) -> None:
        array_insts = {}

        def inner() -> None:
            i1 = p.split(",")[0]
            i2 = q.split(",")[0]
            for i in [i1, i2]:
                if "<" in i:
                    i0 = i.split("<")[0]
                    c, r = [int(x) for x in i.split("<")[1].split(">")[0].split(".")]
                    if i0 not in array_insts:
                        array_insts[i0] = {"columns": 0, "rows": 0}
                    array_insts[i0]["columns"] = max(array_insts[i0]["columns"], c + 1)
                    array_insts[i0]["rows"] = max(array_insts[i0]["rows"], r + 1)

        for net in flatnet.get("nets", []):
            p, q = net["p1"], net["p2"]
            inner()
        for p, q in flatnet.get("connections", {}).items():  # noqa: B007
            inner()
        for p in flatnet.get("ports", {}).values():
            q = p
            inner()

        for k, v in array_insts.items():
            flatnet["instances"][k]["array"] = v

    for v in netlist.values():
        patch_flatnet(v)
