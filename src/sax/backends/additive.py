"""SAX Additive Backend."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import networkx as nx

import sax
from sax.netlists import _nets_to_connections_strict

__all__ = [
    "analyze_circuit_additive",
    "analyze_instances_additive",
    "evaluate_circuit_additive",
]


def analyze_instances_additive(
    instances: sax.Instances,
    models: sax.Models,
) -> dict[sax.InstanceName, sax.SDict]:
    """Analyze circuit instances for the additive backend.

    Prepares instance S-matrices for the additive backend by converting all
    component models to SDict format. The additive backend uses a graph-based
    approach with path finding to compute circuit responses.

    Args:
        instances: Dictionary mapping instance names to instance definitions
            containing component names and settings.
        models: Dictionary mapping component names to their model functions.

    Returns:
        Dictionary mapping instance names to their S-matrices in SDict format.

    Example:
        ```python
        instances = {
            "wg1": {"component": "waveguide", "settings": {"length": 10.0}},
            "dc1": {"component": "coupler", "settings": {"coupling": 0.1}},
        }
        models = {"waveguide": waveguide_model, "coupler": coupler_model}
        analyzed = analyze_instances_additive(instances, models)
        ```
    """
    instances = sax.into[sax.Instances](instances)
    models = sax.into[sax.Models](models)
    model_names = set()
    for i in instances.values():
        model_names.add(i["component"])
    dummy_models = {k: sax.sdict(models[k]()) for k in model_names}
    dummy_instances = {}
    for k, i in instances.items():
        dummy_instances[k] = dummy_models[i["component"]]
    return dummy_instances


def analyze_circuit_additive(
    analyzed_instances: dict[sax.InstanceName, sax.SDict],  # noqa: ARG001
    nets: sax.Nets,
    ports: sax.Ports,
) -> Any:  # noqa: ANN401
    """Analyze circuit topology for the additive backend.

    Prepares the circuit connection information for the additive backend
    evaluation. This backend uses graph theory to find all possible paths
    between circuit ports and sums their contributions.

    Args:
        analyzed_instances: Instance S-matrices from analyze_instances_additive.
            Not used in this analysis step but required for interface consistency.
        nets: List of net dictionaries with "p1" and "p2" keys defining
            internal circuit connections.
        ports: Dictionary mapping external port names to instance ports.

    Returns:
        Tuple containing connections and ports information for circuit evaluation.

    Example:
        ```python
        nets = [{"p1": "wg1,out", "p2": "dc1,in1"}, {"p1": "dc1,out1", "p2": "wg2,in"}]
        ports = {"in": "wg1,in", "out": "wg2,out"}
        analyzed = analyze_circuit_additive(analyzed_instances, nets, ports)
        ```
    """
    connections = _nets_to_connections_strict(nets)
    return connections, ports


def evaluate_circuit_additive(
    analyzed: Any,  # noqa: ANN401
    instances: dict[sax.InstanceName, sax.SDict],
) -> sax.SDict:
    """Evaluate circuit S-matrix using additive path-based method.

    Computes the circuit S-matrix by finding all possible signal paths between
    external ports and additively combining their contributions. This approach
    works well for circuits with multiple parallel paths.

    The algorithm:
    1. Creates a graph representation of the circuit
    2. Finds all simple paths between each pair of external ports
    3. Calculates the transmission/reflection for each path
    4. Sums contributions from all paths

    Args:
        analyzed: Circuit analysis data from analyze_circuit_additive containing
            connections and ports information.
        instances: Dictionary mapping instance names to their evaluated S-matrices
            in SDict format.

    Returns:
        Circuit S-matrix in SDict format with external port combinations as keys.

    Example:
        ```python
        # Evaluated instance S-matrices
        instances = {
            "wg1": {("in", "out"): 0.95 * jnp.exp(1j * 0.1)},
            "dc1": {("in1", "out1"): 0.9, ("in1", "out2"): 0.1},
        }
        circuit_s = evaluate_circuit_additive(analyzed, instances)
        # Result contains S-parameters between external ports
        ```
    """
    connections, ports = analyzed
    edges = _graph_edges(instances, connections, ports)

    graph = nx.Graph()
    graph.add_edges_from(edges)
    _prune_internal_output_nodes(graph)

    sdict = {}
    for source in ports:
        for target in ports:
            paths = _get_possible_paths(graph, source=("", source), target=("", target))
            if not paths:
                continue
            sdict[source, target] = _path_lengths(graph, paths)

    return sdict


def _split_port(port: sax.Port) -> tuple[sax.InstanceName, sax.Name]:
    try:
        instance, port = port.split(",")
    except ValueError:
        (port,) = port.split(",")
        instance = ""
    return instance, port


def _graph_edges(
    instances: dict[sax.InstanceName, sax.SDict],
    connections: sax.Connections,
    ports: sax.Ports,
) -> list[tuple[tuple[str, str], tuple[str, str], dict[str, Any]]]:
    zero = jnp.asarray([0.0])
    edges = {}
    edges.update({_split_port(k): _split_port(v) for k, v in connections.items()})
    edges.update({_split_port(v): _split_port(k) for k, v in connections.items()})
    edges.update({_split_port(k): _split_port(v) for k, v in ports.items()})
    edges.update({_split_port(v): _split_port(k) for k, v in ports.items()})
    edges = [(n1, n2, {"type": "C", "length": zero}) for n1, n2 in edges.items()]

    _instances = {
        **{i1: None for (i1, _), (_, _), _ in edges},
        **{i2: None for (_, _), (i2, _), _ in edges},
    }

    _instances.pop("", None)  # external ports don't belong to an instance

    for instance in _instances:
        s = instances[instance]
        edges += [
            (
                (instance, p1),
                (instance, p2),
                {"type": "S", "length": jnp.asarray(length).ravel()},
            )
            for (p1, p2), length in sax.sdict(s).items()  # type: ignore[reportAttributeAccessIssue]
        ]

    return edges


def _prune_internal_output_nodes(graph: nx.Graph) -> nx.Graph:
    broken = True
    while broken:
        broken = False
        for (i, p), dic in list(graph.adjacency()):
            if (
                i != ""
                and len(dic) == 2
                and all(prop.get("type", "C") == "C" for prop in dic.values())
            ):
                graph.remove_node((i, p))
                graph.add_edge(*dic.keys(), type="C", length=0.0)
                broken = True
                break
    return graph


def _get_possible_paths(
    graph: nx.Graph, source: tuple[str, str], target: tuple[str, str]
) -> list[list[tuple[tuple[str, str], tuple[str, str]]]]:
    paths = []
    default_props = {"type": "C", "length": 0.0}
    for path in nx.all_simple_edge_paths(graph, source, target):
        prevtype = "C"
        for n1, n2 in path:
            curtype = graph.get_edge_data(n1, n2, default_props)["type"]
            if curtype == prevtype == "S":
                break
            prevtype = curtype
        else:
            paths.append(path)
    return paths


def _path_lengths(
    graph: nx.Graph, paths: list[list[tuple[tuple[str, str], tuple[str, str]]]]
) -> list[Any]:
    lengths = []
    for path in paths:
        length = zero = jnp.array([0.0])
        default_edge_data = {"type": "C", "length": zero}
        for edge in path:
            edge_data = graph.get_edge_data(*edge, default_edge_data)
            length = (length[None, :] + edge_data.get("length", zero)[:, None]).ravel()
        lengths.append(length)
    return lengths
