"""SAX Circuit Definition."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, Literal, cast, overload

import networkx as nx
import numpy as np
from kfnetlist import HierarchicalNetlist, Netlist

import sax

from . import _circuit_compiler as compiler
from .backends import circuit_backends
from .models.probes import ideal_probe
from .s import get_ports, scoo, sdense, sdict
from .utils import get_settings, merge_dicts, replace_kwargs, update_settings

__all__ = ["circuit", "draw_dag", "get_required_circuit_models"]


@overload
def circuit(
    netlist: Netlist | HierarchicalNetlist,
    models: Mapping[str, sax.Model] | None = None,
    *,
    backend: sax.BackendLike = "default",
    top_level_name: str | None = None,
    ignore_impossible_connections: bool = False,
    probes: dict[str, str] | None = None,
    on_internal_port: Literal["warn", "ignore", "as_probes"] = "warn",
) -> tuple[sax.SDictModel, sax.CircuitInfo]: ...


@overload
def circuit(
    netlist: Netlist | HierarchicalNetlist,
    models: Mapping[str, sax.Model] | None = None,
    *,
    backend: sax.BackendLike = "default",
    return_type: Literal["SDict"],
    top_level_name: str | None = None,
    ignore_impossible_connections: bool = False,
    probes: dict[str, str] | None = None,
    on_internal_port: Literal["warn", "ignore", "as_probes"] = "warn",
) -> tuple[sax.SDictModel, sax.CircuitInfo]: ...


@overload
def circuit(
    netlist: Netlist | HierarchicalNetlist,
    models: Mapping[str, sax.Model] | None = None,
    *,
    backend: sax.BackendLike = "default",
    return_type: Literal["SDense"],
    top_level_name: str | None = None,
    ignore_impossible_connections: bool = False,
    probes: dict[str, str] | None = None,
    on_internal_port: Literal["warn", "ignore", "as_probes"] = "warn",
) -> tuple[sax.SDenseModel, sax.CircuitInfo]: ...


@overload
def circuit(
    netlist: Netlist | HierarchicalNetlist,
    models: Mapping[str, sax.Model] | None = None,
    *,
    backend: sax.BackendLike = "default",
    return_type: Literal["SCoo"],
    top_level_name: str | None = None,
    ignore_impossible_connections: bool = False,
    probes: dict[str, str] | None = None,
    on_internal_port: Literal["warn", "ignore", "as_probes"] = "warn",
) -> tuple[sax.SCooModel, sax.CircuitInfo]: ...


def circuit(
    netlist: Netlist | HierarchicalNetlist,
    models: Mapping[str, sax.Model] | None = None,
    *,
    backend: sax.BackendLike = "default",
    return_type: Literal["SDict", "SDense", "SCoo"] = "SDict",
    top_level_name: str | None = None,
    ignore_impossible_connections: bool = False,
    probes: dict[str, str] | None = None,
    on_internal_port: Literal["warn", "ignore", "as_probes"] = "warn",
) -> tuple[sax.Model, sax.CircuitInfo]:
    """Create a circuit function for a given netlist.

    Constructs a circuit model from a netlist description by connecting component
    models according to the specified connections. The resulting circuit function
    can be called with parameters to evaluate the overall S-matrix.

    Args:
        netlist: A kfnetlist Netlist or HierarchicalNetlist.
        models: Dictionary mapping factory, library-qualified, or reference IDs
            to model functions. Leaf instances need a matching model.
        backend: Circuit analysis backend to use. Options include "default",
            "klu", "filipsson_gunnar", "additive". Defaults to "default".
        return_type: Format of the returned S-matrix. Options: "SDict", "SDense",
            "SCoo". Defaults to "SDict".
        top_level_name: Name of the root in a hierarchy. Defaults to
            ``top_level`` when present, then the first document entry.
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
            to measurement probes.

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


        from kfnetlist import Netlist, NetlistPort, PortRef

        netlist = Netlist()
        netlist.create_inst("wg1", "pdk", "waveguide", {"length": 20.0})
        for port in ("in", "out"):
            netlist.create_port(port)
            netlist.create_net(NetlistPort(name=port), PortRef("wg1", port))

        # Create circuit
        models = {"waveguide": waveguide}
        circuit_func, info = circuit(netlist, models)

        # Evaluate circuit
        s_matrix = circuit_func(wl=1.55)
        ```
    """
    _backend = sax.into[sax.Backend](backend)

    merged_models: sax.Models = dict(models or {})
    return _compile_circuit(
        netlist,
        merged_models,
        backend=_backend,
        return_type=return_type,
        top_level_name=top_level_name,
        ignore_impossible_connections=ignore_impossible_connections,
        probes=probes,
        on_internal_port=on_internal_port,
    )


def _document(
    netlist: Netlist | HierarchicalNetlist, root_name: str | None
) -> tuple[dict[str, Netlist], str]:
    """Select a validated kfnetlist document and its simulation root."""
    if type(netlist) is Netlist:
        root = root_name or "top_level"
        document = HierarchicalNetlist({root: netlist})
    elif isinstance(netlist, HierarchicalNetlist):
        document = netlist
        document.validate()
        root = root_name or (
            "top_level" if "top_level" in document else next(iter(document), None)
        )
    else:
        msg = "Expected a kfnetlist.Netlist or kfnetlist.HierarchicalNetlist."
        raise TypeError(msg)
    if root is None or root not in document:
        msg = f"Unknown top-level netlist {root!r}; available IDs: {list(document)!r}."
        raise ValueError(msg)
    return dict(document.items()), root


def _definition_dag(
    cells: dict[str, Netlist],
    root: str,
    models: sax.Models,
    *,
    require_models: bool = False,
) -> nx.DiGraph:
    g = nx.DiGraph()
    pending = [(root, root)]
    visited: set[str] = set()
    while pending:
        cell_name, path = pending.pop()
        g.add_node(cell_name)
        if cell_name in visited or cell_name in models:
            continue
        visited.add(cell_name)
        for name, inst in cells[cell_name].instances.items():
            key = compiler.resolve(inst, models, cells)
            if key is None:
                if require_models:
                    raise ValueError(
                        compiler.missing_model_message(inst, f"{path}.{name}")
                    )
                key = inst.component
            g.add_edge(cell_name, key)
            if key in cells and key not in models:
                pending.append((key, f"{path}.{name}"))
    return _validate_dag(g)


def _prepare_circuit(
    netlist: Netlist | HierarchicalNetlist,
    models: sax.Models,
    top_level_name: str | None,
    probes: dict[str, str] | None,
) -> tuple[
    dict[str, Netlist],
    str,
    dict[str, str],
    dict[str, dict[str, str]],
    dict[str, list[tuple[str, str]]],
]:
    cells, root = _document(netlist, top_level_name)

    top_probes, per_cell_probes, probe_paths = compiler.plan_hierarchical_probes(
        cells, root, models, probes or {}
    )
    keep = {cell: tuple(targets.values()) for cell, targets in per_cell_probes.items()}
    keep[root] = (*keep.get(root, ()), *top_probes.values())
    for path in probe_paths.values():
        for instance, parent in path:
            keep[parent] = (*keep.get(parent, ()), instance)
    cells = {
        name: compiler.prune_unconnected_instances(nl, keep=keep.get(name, ()))
        for name, nl in cells.items()
    }
    return cells, root, top_probes, per_cell_probes, probe_paths


def _prepare_cell_tables(
    nl: Netlist,
    model_name: str,
    root: str,
    models: sax.Models,
    cells: dict[str, Netlist],
    extra_ports: dict[str, dict[str, str]],
    top_probes: dict[str, str],
    per_cell_probes: dict[str, dict[str, str]],
    on_internal_port: str,
) -> tuple[sax.Instances, sax.Nets, sax.Ports, bool]:
    instances, nets, ports, bindings = compiler.lower_bindings(nl, models, cells)
    for name, inst in instances.items():
        key = bindings.get(name)
        if key is None:
            msg = (
                f"Could not resolve model for instance {name!r} "
                f"(component {inst['component']!r}) in {model_name!r}."
            )
            raise ValueError(msg)
        inst["component"] = key

    for port_name, endpoint in extra_ports.get(model_name, {}).items():
        if port_name in ports:
            msg = (
                f"Hierarchical probe port {port_name!r} conflicts with "
                f"an existing port in {model_name!r}."
            )
            raise ValueError(msg)
        ports[port_name] = endpoint

    probe_here: dict[str, str] = {}
    if model_name == root:
        ports, auto_probes = compiler.handle_internal_ports(
            nets, ports, on_internal_port
        )
        probe_here.update(top_probes)
        probe_here.update(auto_probes)
    probe_here.update(per_cell_probes.get(model_name, {}))

    if probe_here:
        instances, nets, ports = compiler.expand_probes_tables(
            instances, nets, ports, probe_here
        )

    if model_name == root and not ports:
        msg = (
            "Cannot create circuit: at least 1 port needs to be defined. "
            "Got no ports given."
        )
        raise ValueError(msg)

    return instances, nets, ports, bool(probe_here)


def _compile_circuit(
    netlist: Netlist | HierarchicalNetlist,
    models: sax.Models | None,
    *,
    backend: sax.Backend,
    return_type: Literal["SDict", "SDense", "SCoo"],
    top_level_name: str | None,
    ignore_impossible_connections: bool,
    probes: dict[str, str] | None,
    on_internal_port: Literal["warn", "ignore", "as_probes"],
) -> tuple[sax.Model, sax.CircuitInfo]:
    models = dict(models or {})
    cells, root, top_probes, per_cell_probes, probe_paths = _prepare_circuit(
        netlist, models, top_level_name, probes
    )
    dependency_dag = _definition_dag(cells, root, models, require_models=True)
    models = _validate_models(models, dependency_dag)

    extra_ports: dict[str, dict[str, str]] = {}
    for probe_name, path in probe_paths.items():
        for instance_name, parent in reversed(path):
            extra_ports.setdefault(parent, {})[f"{probe_name}_fwd"] = (
                f"{instance_name},{probe_name}_fwd"
            )
            extra_ports.setdefault(parent, {})[f"{probe_name}_bwd"] = (
                f"{instance_name},{probe_name}_bwd"
            )

    circuit = None
    new_models: sax.Models = {}
    current_models: sax.Models = {}
    model_names = list(nx.topological_sort(dependency_dag))[::-1]
    for model_name in model_names:
        if model_name in models:
            new_models[model_name] = models[model_name]
            continue

        current_models |= new_models
        new_models = {}
        nl = cells[model_name]
        instances, nets, ports, has_probes = _prepare_cell_tables(
            nl,
            model_name,
            root,
            models,
            cells,
            extra_ports,
            top_probes,
            per_cell_probes,
            on_internal_port,
        )

        available: sax.Models = {**models, **current_models}
        if has_probes:
            available["_ideal_probe"] = ideal_probe
        current_models[model_name] = circuit = _flat_circuit(
            instances,
            nets,
            ports,
            available,
            backend,
            ignore_impossible_connections=ignore_impossible_connections,
        )

    if circuit is None:
        msg = "Could not construct circuit (unknown reason)"
        raise RuntimeError(msg)
    circuit = _enforce_return_type(circuit, return_type)
    return circuit, sax.CircuitInfo(
        dag=dependency_dag,
        models=current_models,
        backend=backend,
    )


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
    netlist: Netlist | HierarchicalNetlist,
    models: Mapping[str, sax.Model] | None = None,
    *,
    top_level_name: str | None = None,
) -> list[str]:
    """Determine which component models are required for a given netlist.

    Analyzes a netlist to identify all component types that need model functions.
    This is useful for validating that all required models are available before
    circuit construction.

    Args:
        netlist: Circuit netlist to analyze for component dependencies.
        models: Optional model bindings used to determine analytical boundaries.
            The result includes required primitive models even when supplied.
        top_level_name: Explicit root; otherwise use document/default root selection.

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
        # Result: ["directional_coupler", "waveguide"] (order unspecified)
        ```
    """
    merged: sax.Models = dict(models or {})
    cells, root = _document(netlist, top_level_name)
    cells = {
        name: compiler.prune_unconnected_instances(nl) for name, nl in cells.items()
    }
    dependency_dag = _definition_dag(cells, root, merged)
    _, required, _ = _find_missing_models(merged, dependency_dag)
    return required


def _flat_circuit(
    instances: sax.Instances,
    nets: sax.Nets,
    ports: sax.Ports,
    models: sax.Models,
    backend: sax.Backend,
    *,
    ignore_impossible_connections: bool = False,
) -> sax.Model:
    analyze_insts_fn, analyze_fn, evaluate_fn = circuit_backends[backend]
    # Backend discovery validates Python identifiers. Model keys may contain
    # library separators, so give that boundary local IDs.
    model_ids = {
        component: f"_model_{i}"
        for i, component in enumerate(
            dict.fromkeys(inst["component"] for inst in instances.values())
        )
    }
    analysis_instances = {
        name: {**inst, "component": model_ids[inst["component"]]}
        for name, inst in instances.items()
    }
    analysis_models = {model_ids[key]: models[key] for key in model_ids}
    dummy_instances = analyze_insts_fn(analysis_instances, analysis_models)
    inst_port_mode = {
        k: _port_modes_dict(get_ports(s)) for k, s in dummy_instances.items()
    }
    expanded_nets = _get_multimode_nets(
        nets,
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
    default_settings = merge_dicts(model_settings, netlist_settings)
    default_settings = {_strip_array_index(k): v for k, v in default_settings.items()}
    analyzed = analyze_fn(dummy_instances, expanded_nets, ports)

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
    instance_names = {_strip_array_index(name) for name in instances}
    global_settings = {
        k: settings.pop(k) for k in list(settings.keys()) if k not in instance_names
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
        msg = f"Invalid return_type {return_type!r}; expected SDict, SCoo, or SDense."
        raise ValueError(msg)
    return stype(model)


def _validate_dag(dag: nx.DiGraph) -> nx.DiGraph:
    if not nx.is_directed_acyclic_graph(dag):
        msg = "Netlist dependency cycles detected!"
        raise ValueError(msg)
    nodes = _find_root(dag)
    if len(nodes) > 1:
        msg = f"Multiple top_levels found in netlist: {nodes}"
        raise ValueError(msg)
    if len(nodes) < 1:
        msg = "Netlist does not contain any nodes."
        raise ValueError(msg)
    return dag


def _strip_array_index(s: sax.InstanceName) -> sax.Name:
    return s.split("<")[0]
