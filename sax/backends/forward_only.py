"""SAX forward_only Backend"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import networkx as nx

from ..netlist import Component
from ..saxtypes import Model, SCoo, SDict, scoo, sdict


def analyze_instances_forward(
    instances: dict[str, Component],
    models: dict[str, Model],
) -> dict[str, SCoo]:
    instances, instances_old = {}, instances
    for k, v in instances_old.items():
        if not isinstance(v, Component):
            v = Component(**v)
        instances[k] = v
    model_names = set()
    for i in instances.values():
        model_names.add(i.component)
    dummy_models = {k: scoo(models[k]()) for k in model_names}
    dummy_instances = {}
    for k, i in instances.items():
        dummy_instances[k] = dummy_models[i.component]
    return dummy_instances


def analyze_circuit_forward(
    analyzed_instances: dict[str, SDict],
    connections: dict[str, str],
    ports: dict[str, str],
) -> Any:
    return connections, ports


# import matplotlib.pyplot as plt
def evaluate_circuit_forward(
    analyzed: Any,
    instances: dict[str, SDict],
) -> SDict:
    """Evaluate a circuit for the given sdicts using simple matrix multiplication."""
    connections, ports = analyzed
    edges = _graph_edges_directed(instances, connections, ports)
    graph = nx.DiGraph()
    graph.add_edges_from(edges)

    circuit_sdict = {}
    one_nodes = [p for p in ports if p.startswith("in")]
    # Preallocate layer-by-layer signals to minimize copy
    for in_port in one_nodes:
        orig_node = ("", in_port)
        node_signals = {orig_node: 1}
        # Preallocate current and next layer dicts
        bfs_layers = nx.bfs_layers(graph, orig_node)
        for layer in bfs_layers:
            layer_signals = {}
            for node in layer:
                signal = node_signals.get(node)
                if signal is not None:
                    for neighbor in graph.successors(node):
                        transmission = graph[node][neighbor]["transmission"]
                        # Accumulate to existing value if present
                        prev = layer_signals.get(neighbor)
                        s = signal * transmission
                        if prev is not None:
                            layer_signals[neighbor] = prev + s
                        else:
                            layer_signals[neighbor] = s
            node_signals.update(layer_signals)
        # Use a small local dict, then update once after collecting all key/vals
        local_sdict = {}
        for (p1, p2), v in node_signals.items():
            if p1 == "" and p2.startswith("out"):
                local_sdict[(in_port, p2)] = v
        circuit_sdict.update(local_sdict)
    return circuit_sdict


def _split_port(port: str) -> Tuple[str, str]:
    try:
        instance, port = port.split(",")
    except ValueError:
        (port,) = port.split(",")
        instance = ""
    return instance, port


def _graph_edges_directed(
    instances: dict[str, SDict],
    connections: dict[str, str],
    ports: dict[str, str],
):
    from sax.backends.forward_only import _split_port

    one = jnp.array([1.0], dtype=float)
    # Avoid creating two dicts and successive update: merge both generators up front
    edges_dict = dict(
        (_split_port(k), _split_port(v))
        for k, v in list(connections.items()) + list(ports.items())
    )
    edges = []
    for n1, n2 in edges_dict.items():
        # Avoid list addition by using append
        if n1[0] == "" and n1[1].startswith("out"):
            edges.append((n2, n1, {"transmission": one}))
        else:
            edges.append((n1, n2, {"transmission": one}))

    # sdict conversion done once per instance (ensure fast lookup)
    for instance, s in instances.items():
        instance_sdict = sdict(s)
        for (p1, p2), w in instance_sdict.items():
            if p1.startswith("in") and p2.startswith("out"):
                # If w is already JAX array and ravelled, skip conversion
                rv = w
                if not (hasattr(w, "shape") and len(getattr(w, "shape", ())) == 1):
                    rv = jnp.asarray(w, dtype=complex).ravel()
                edges.append(
                    (
                        (instance, p1),
                        (instance, p2),
                        {"transmission": rv},
                    )
                )
    return edges
