"""Connection conversion for backends that require paired endpoints."""

from sax.saxtypes.netlist import Connections, Nets


def nets_to_connections_strict(nets: Nets) -> Connections:
    """Reject repeated endpoints for backends that require paired connections."""
    connections: Connections = {}
    seen: set[str] = set()
    for net in nets:
        for endpoint in (net["p1"], net["p2"]):
            if endpoint in seen:
                msg = (
                    "Multiply connected ports are only supported with the 'klu' "
                    f"backend. Port {endpoint!r} appears in multiple connections."
                )
                raise ValueError(msg)
            seen.add(endpoint)
        connections[net["p1"]] = net["p2"]
    return connections
