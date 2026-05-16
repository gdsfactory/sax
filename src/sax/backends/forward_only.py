"""Forward-only backend for SAX circuit evaluation."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import networkx as nx

import sax
from sax.netlists import _nets_to_connections_strict

__all__ = [
    "analyze_circuit_forward",
    "analyze_instances_forward",
    "evaluate_circuit_forward",
]


def analyze_instances_forward(
    instances: sax.Instances,
    models: sax.Models,
) -> dict[sax.InstanceName, sax.SCoo]:
    """Analyze circuit instances for the forward-only backend.

    Prepares instance S-matrices for the forward-only backend by converting all
    component models to SCoo (coordinate) format. This backend is specialized
    for feed-forward circuits without feedback loops.

    Args:
        instances: Dictionary mapping instance names to instance definitions
            containing component names and settings.
        models: Dictionary mapping component names to their model functions.

    Returns:
        Dictionary mapping instance names to their S-matrices in SCoo format.

    Note:
        The forward-only backend is designed for circuits with unidirectional
        signal flow and no feedback paths. It uses a simplified approach that
        may not be accurate for circuits with reflections or loops.

    Example:
        ```python
        instances = {
            "wg1": {"component": "waveguide", "settings": {"length": 10.0}},
            "amp1": {"component": "amplifier", "settings": {"gain": 20.0}},
        }
        models = {"waveguide": waveguide_model, "amplifier": amplifier_model}
        analyzed = analyze_instances_forward(instances, models)
        ```
    """
    instances = sax.into[sax.Instances](instances)
    models = sax.into[sax.Models](models)
    model_names = set()
    for i in instances.values():
        model_names.add(i["component"])
    dummy_models = {k: sax.scoo(models[k]()) for k in model_names}
    dummy_instances = {}
    for k, i in instances.items():
        dummy_instances[k] = dummy_models[i["component"]]
    return dummy_instances


def analyze_circuit_forward(
    analyzed_instances: dict[sax.InstanceName, sax.SDict],  # noqa: ARG001
    nets: sax.Nets,
    ports: sax.Ports,
) -> Any:  # noqa: ANN401
    """Analyze circuit topology for the forward-only backend.

    Prepares the circuit connection information for the forward-only backend
    evaluation. This backend assumes unidirectional signal flow and does not
    account for reflections or bidirectional coupling.

    Args:
        analyzed_instances: Instance S-matrices from analyze_instances_forward.
            Not used in this analysis step but required for interface consistency.
        nets: List of net dictionaries with "p1" and "p2" keys defining
            internal circuit connections.
        ports: Dictionary mapping external port names to instance ports.

    Returns:
        Tuple containing connections and ports information for circuit evaluation.

    Example:
        ```python
        nets = [{"p1": "wg1,out", "p2": "amp1,in"}, {"p1": "amp1,out", "p2": "wg2,in"}]
        ports = {"in": "wg1,in", "out": "wg2,out"}
        analyzed = analyze_circuit_forward(analyzed_instances, nets, ports)
        ```
    """
    connections = _nets_to_connections_strict(nets)
    return connections, ports


def evaluate_circuit_forward(
    analyzed: Any,  # noqa: ANN401
    instances: dict[str, sax.SDict],
) -> sax.SDict:
    """Evaluate circuit S-matrix using forward-only propagation.

    Computes the circuit response using a simplified forward propagation approach.
    This method assumes unidirectional signal flow and uses breadth-first search
    to propagate signals through the circuit without considering reflections.

    The algorithm:
    1. Creates a directed graph representation of the circuit
    2. For each input port, injects a unit signal
    3. Uses BFS to propagate signals through the circuit
    4. Records the signal levels at output ports

    Args:
        analyzed: Circuit analysis data from analyze_circuit_forward containing
            connections and ports information.
        instances: Dictionary mapping instance names to their evaluated S-matrices
            in SDict format.

    Returns:
        Circuit S-matrix in SDict format, typically containing only forward
        transmission terms (input to output ports).

    Warning:
        This backend is only accurate for feed-forward circuits without
        reflections or feedback paths. For circuits with bidirectional coupling
        or reflections, use the Filipsson-Gunnar or KLU backends instead.

    Example:
        ```python
        # Circuit analysis and instances (feed-forward only)
        analyzed = (connections, ports)
        instances = {
            "wg1": {("in", "out"): 0.95},  # Low-loss waveguide
            "amp1": {("in", "out"): 10.0},  # 20dB amplifier
        }
        circuit_s = evaluate_circuit_forward(analyzed, instances)
        # Result contains only forward transmission terms
        ```
    """
    connections, ports = analyzed
    edges = _graph_edges_directed(instances, connections, ports)

    graph = nx.DiGraph()
    graph.add_edges_from(edges)

    # Dictionary to store signals at each node
    circuit_sdict = {}
    for in_port in ports:
        if in_port.startswith("in"):
            node_signals = {("", in_port): 1}
            bfs_output = nx.bfs_layers(graph, ("", in_port))
            for layer in bfs_output:
                layer_signals = {}
                for node in layer:
                    if node in node_signals:
                        signal = node_signals[node]
                        for neighbor in graph.successors(node):
                            transmission = graph[node][neighbor]["transmission"]
                            if neighbor in layer_signals:
                                layer_signals[neighbor] += signal * transmission
                            else:
                                layer_signals[neighbor] = signal * transmission
                node_signals.update(layer_signals)
            sdict = {
                (in_port, p2): v
                for (p1, p2), v in node_signals.items()
                if p1 == "" and p2.startswith("out")
            }
            circuit_sdict.update(sdict)
    return circuit_sdict


def _split_port(port: str) -> tuple[str, str]:
    try:
        instance, port = port.split(",")
    except ValueError:
        (port,) = port.split(",")
        instance = ""
    return instance, port


def _graph_edges_directed(
    instances: dict[str, sax.SDict],
    connections: dict[str, str],
    ports: dict[str, str],
) -> list[tuple[tuple[str, str], tuple[str, str], dict[str, Any]]]:
    one = jnp.array([1.0], dtype=float)
    edges_dict = {}
    edges_dict.update({_split_port(k): _split_port(v) for k, v in connections.items()})
    edges_dict.update({_split_port(k): _split_port(v) for k, v in ports.items()})
    edges = []
    for n1, n2 in edges_dict.items():
        if n1[0] == "" and n1[1].startswith("out"):
            edges += [(n2, n1, {"transmission": one})]
        else:
            edges += [(n1, n2, {"transmission": one})]

    for instance, s in instances.items():
        for (p1, p2), w in sax.sdict(s).items():
            if p1.startswith("in") and p2.startswith("out"):
                edges += [
                    (
                        (instance, p1),
                        (instance, p2),
                        {"transmission": jnp.asarray(w, dtype=complex).ravel()},
                    )
                ]
    return edges
