"""Derive solver wiring from kfnetlist without another netlist schema."""

from kfnetlist import Netlist, NetlistPort, PortRef


def _endpoint(ref: PortRef) -> str:
    return f"{ref.instance},{ref.port}"


def solver_wiring(netlist: Netlist) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """Return only the endpoint indices needed by SAX's numerical solvers."""
    pairs: list[tuple[str, str]] = []
    attached: dict[str, str] = {}
    for net in netlist.nets:
        refs = [member for member in net if isinstance(member, PortRef)]
        external = [member for member in net if isinstance(member, NetlistPort)]
        if external:
            for port in external:
                attached[port.name] = _endpoint(refs[0])
        elif len(refs) == 2:
            pairs.append((_endpoint(refs[0]), _endpoint(refs[1])))
    ports = {port.name: attached[port.name] for port in netlist.ports}
    return pairs, ports


def pairwise_connections_strict(
    pairs: list[tuple[str, str]],
) -> dict[str, str]:
    """Reject repeated endpoints for backends that require paired connections."""
    connections: dict[str, str] = {}
    seen: set[str] = set()
    for left, right in pairs:
        for endpoint in (left, right):
            if endpoint in seen:
                msg = (
                    "Multiply connected ports are only supported with the 'klu' "
                    f"backend. Port {endpoint!r} appears in multiple connections."
                )
                raise ValueError(msg)
            seen.add(endpoint)
        connections[left] = right
    return connections
