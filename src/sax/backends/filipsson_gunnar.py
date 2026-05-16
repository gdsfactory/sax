"""SAX Filipsson Gunnar Backend."""

from __future__ import annotations

from typing import Any

import jax
from jaxtyping import Array

import sax
from sax.netlists import _nets_to_connections_strict

__all__ = [
    "analyze_circuit_fg",
    "analyze_instances_fg",
    "evaluate_circuit_fg",
]


def analyze_instances_fg(
    instances: sax.Instances,
    models: sax.Models,
) -> dict[sax.InstanceName, sax.SDict]:
    """Analyze circuit instances for the Filipsson-Gunnar backend.

    Prepares instance S-matrices for the Filipsson-Gunnar backend by converting
    all component models to SDict format. This backend implements the classic
    Filipsson-Gunnar algorithm for S-matrix interconnection of multiports.

    Args:
        instances: Dictionary mapping instance names to instance definitions
            containing component names and settings.
        models: Dictionary mapping component names to their model functions.

    Returns:
        Dictionary mapping instance names to their S-matrices in SDict format.

    Note:
        The Filipsson-Gunnar algorithm is a systematic method for computing
        the overall S-matrix of interconnected multiport networks described in:
        Filipsson, Gunnar. "A new general computer algorithm for S-matrix
        calculation of interconnected multiports." 11th European Microwave
        Conference. IEEE, 1981.

    Example:
        ```python
        instances = {
            "wg1": {"component": "waveguide", "settings": {"length": 10.0}},
            "dc1": {"component": "coupler", "settings": {"coupling": 0.1}},
        }
        models = {"waveguide": waveguide_model, "coupler": coupler_model}
        analyzed = analyze_instances_fg(instances, models)
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


def analyze_circuit_fg(
    analyzed_instances: dict[str, sax.SDict],  # noqa: ARG001
    nets: sax.Nets,
    ports: sax.Ports,
) -> Any:  # noqa: ANN401
    """Analyze circuit topology for the Filipsson-Gunnar backend.

    Prepares the circuit connection information for the Filipsson-Gunnar backend
    evaluation. This implementation currently skips detailed analysis and passes
    the connection information directly to the evaluation phase.

    Args:
        analyzed_instances: Instance S-matrices from analyze_instances_fg.
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
        analyzed = analyze_circuit_fg(analyzed_instances, nets, ports)
        ```
    """
    connections = _nets_to_connections_strict(nets)
    return connections, ports  # skip analysis for now


def evaluate_circuit_fg(
    analyzed: Any,  # noqa: ANN401
    instances: dict[str, sax.SType],
) -> sax.SDict:
    """Evaluate circuit S-matrix using the Filipsson-Gunnar algorithm.

    Computes the overall circuit S-matrix by systematically interconnecting
    multiport networks using the Filipsson-Gunnar algorithm. This method
    iteratively applies equation 6 from the original paper to connect ports.

    The algorithm:
    1. Creates a block diagonal S-matrix from all component S-matrices
    2. Iteratively interconnects ports according to the connections
    3. Applies the Filipsson-Gunnar interconnection formula for each connection
    4. Extracts the final S-matrix for external ports

    Args:
        analyzed: Circuit analysis data from analyze_circuit_fg containing
            connections and ports information.
        instances: Dictionary mapping instance names to their evaluated S-matrices
            in any SAX format (will be converted to SDict).

    Returns:
        Circuit S-matrix in SDict format with external port combinations as keys.

    Note:
        The interconnection formula used is equation 6 from:
        Filipsson, Gunnar. "A new general computer algorithm for S-matrix
        calculation of interconnected multiports." 11th European Microwave
        Conference. IEEE, 1981.

    Example:
        ```python
        # Circuit analysis and instances
        analyzed = (connections, ports)
        instances = {
            "wg1": {("in", "out"): 0.95 * jnp.exp(1j * 0.1)},
            "dc1": {("in1", "out1"): 0.9, ("in1", "out2"): 0.1},
        }
        circuit_s = evaluate_circuit_fg(analyzed, instances)
        ```
    """
    connections, ports = analyzed

    # it's actually easier working w reverse:
    reversed_ports = {v: k for k, v in ports.items()}

    block_diag = {}
    for name, S in instances.items():
        block_diag.update(
            {
                (f"{name},{p1}", f"{name},{p2}"): v
                for (p1, p2), v in sax.sdict(S).items()
            }
        )

    sorted_connections = sorted(connections.items(), key=_connections_sort_key)
    all_connected_instances = {k: {k} for k in instances}

    for k, l in sorted_connections:
        name1, _ = k.split(",")
        name2, _ = l.split(",")

        connected_instances = (
            all_connected_instances[name1] | all_connected_instances[name2]
        )
        for name in connected_instances:
            all_connected_instances[name] = connected_instances

        current_ports = tuple(
            p
            for instance in connected_instances
            for p in set([p for p, _ in block_diag] + [p for _, p in block_diag])
            if p.startswith(f"{instance},")
        )

        block_diag.update(_interconnect_ports(block_diag, current_ports, k, l))

        for i, j in list(block_diag.keys()):
            is_connected = j in (k, l) or i in (k, l)
            is_in_output_ports = i in reversed_ports and j in reversed_ports
            if is_connected and not is_in_output_ports:
                del block_diag[
                    i, j
                ]  # we're no longer interested in these port combinations

    circuit_sdict: sax.SDict = {
        (reversed_ports[i], reversed_ports[j]): v
        for (i, j), v in block_diag.items()
        if i in reversed_ports and j in reversed_ports
    }
    return circuit_sdict


def _connections_sort_key(connection: tuple[str, str]) -> tuple[str, str]:
    """Sort key for sorting a connection dictionary."""
    part1, part2 = connection
    name1, _ = part1.split(",")
    name2, _ = part2.split(",")
    return (min(name1, name2), max(name1, name2))


def _interconnect_ports(
    block_diag: dict[tuple[str, str], Any],
    current_ports: tuple[str, ...],
    k: str,
    l: str,
) -> dict[tuple[str, str], Any]:
    """Interconnect two ports in a given model.

    > the interconnect algorithm is based on equation 6 of 'Filipsson, Gunnar.
    > "A new general computer algorithm for S-matrix calculation of interconnected
    > multiports." 11th European Microwave Conference. IEEE, 1981.'

    """
    current_block_diag = {}
    for i in current_ports:
        for j in current_ports:
            vij = _calculate_interconnected_value(
                vij=block_diag.get((i, j), 0.0),
                vik=block_diag.get((i, k), 0.0),
                vil=block_diag.get((i, l), 0.0),
                vkj=block_diag.get((k, j), 0.0),
                vkk=block_diag.get((k, k), 0.0),
                vkl=block_diag.get((k, l), 0.0),
                vlj=block_diag.get((l, j), 0.0),
                vlk=block_diag.get((l, k), 0.0),
                vll=block_diag.get((l, l), 0.0),
            )
            current_block_diag[i, j] = vij
    return current_block_diag


@jax.jit
def _calculate_interconnected_value(
    vij: Array,
    vik: Array,
    vil: Array,
    vkj: Array,
    vkk: Array,
    vkl: Array,
    vlj: Array,
    vlk: Array,
    vll: Array,
) -> Array:
    """Calculate an interconnected S-parameter value.

    > The interconnect algorithm is based on equation 6 in the paper below
    > Filipsson, Gunnar. "A new general computer algorithm for S-matrix calculation
    > of interconnected multiports." 11th European Microwave Conference. IEEE, 1981.

    """
    result = vij + (
        vkj * vil * (1 - vlk)
        + vlj * vik * (1 - vkl)
        + vkj * vll * vik
        + vlj * vkk * vil
    ) / ((1 - vkl) * (1 - vlk) - vkk * vll)
    return result
