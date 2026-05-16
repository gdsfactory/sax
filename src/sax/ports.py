"""Port naming strategies."""

from __future__ import annotations

import re
from typing import Literal, TypeAlias

import sax

PortNamingStrategy: TypeAlias = Literal["optical", "inout"]

PORT_NAMING_STRATEGY: PortNamingStrategy = "inout"


def set_port_naming_strategy(strategy: PortNamingStrategy) -> PortNamingStrategy:
    """Set the port naming strategy for the default SAX models.

    This function configures the global port naming convention used by SAX model
    functions. The strategy affects how ports are named in generated S-matrices
    and impacts netlist compatibility with different simulation tools.

    Args:
        strategy: Port naming strategy to use. Options are:
            - "optical": Uses optical convention (o1, o2, o3, ...)
            - "inout": Uses input/output convention (in0, in1, out0, out1, ...)

    Returns:
        The strategy that was set (for confirmation).

    Raises:
        ValueError: If the strategy is not one of the valid options.

    Examples:
        Switch to optical naming:

        ```python
        import sax.models

        sax.models.set_port_naming_strategy("optical")
        # Now models will use o1, o2, o3, ... port names
        ```

        Switch to input/output naming:

        ```python
        sax.models.set_port_naming_strategy("inout")
        # Now models will use in0, in1, out0, out1, ... port names
        ```

        Check current strategy:

        ```python
        current = sax.models.get_port_naming_strategy()
        print(f"Current strategy: {current}")
        ```

    Note:
        This is a global setting that affects all subsequently created model
        instances. Existing model instances retain their original port naming.
        The "inout" convention is more intuitive for circuit simulation, while
        "optical" follows traditional optical component naming.
    """
    global PORT_NAMING_STRATEGY  # noqa: PLW0603
    _strategy = str(strategy).lower()
    if _strategy not in ["optical", "inout"]:
        msg = f"Invalid port naming strategy: {strategy}"
        raise ValueError(msg)
    PORT_NAMING_STRATEGY = strategy
    return strategy


def get_port_naming_strategy() -> PortNamingStrategy:
    """Get the current port naming strategy.

    Returns the currently active port naming strategy that determines how
    ports are named in SAX model functions.

    Returns:
        Current port naming strategy ("optical" or "inout").

    Examples:
        Check current strategy:

        ```python
        import sax.models

        strategy = sax.models.get_port_naming_strategy()
        print(f"Current strategy: {strategy}")
        ```

        Use in conditional logic:

        ```python
        if sax.models.get_port_naming_strategy() == "optical":
            print("Using optical port naming (o1, o2, ...)")
        else:
            print("Using input/output port naming (in0, out0, ...)")
        ```
    """
    return PORT_NAMING_STRATEGY


class PortNamer:
    """Port naming class to encapsulate port naming logic.

    This class provides a unified interface for generating consistent port names
    across SAX model functions. It supports different naming strategies and
    handles the mapping between logical port indices and string names.

    The PortNamer automatically applies the current global naming strategy or
    allows override with a specific strategy. It provides both attribute-based
    access (e.g., p.in0, p.out1) and index-based access (e.g., p[0], p[1]).

    Attributes:
        num_inputs: Number of input ports.
        num_outputs: Number of output ports.
        num_ports: Total number of ports (inputs + outputs).
        strategy: Port naming strategy being used.
    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        strategy: PortNamingStrategy | None = None,
    ) -> None:
        """Initialize the PortNamer with the number of ports.

        Args:
            num_inputs: Number of input ports for the device.
            num_outputs: Number of output ports for the device.
            strategy: Optional port naming strategy override. If None, uses
                the current global strategy from get_port_naming_strategy().

        Examples:
            Create a 2x2 device (e.g., directional coupler):

            ```python
            p = PortNamer(2, 2)
            print(p.in0, p.in1, p.out0, p.out1)  # inout strategy
            ```

            Create a 1x2 device (e.g., splitter):

            ```python
            p = PortNamer(1, 2)
            print(p.in0, p.out0, p.out1)
            ```

            Force optical naming:

            ```python
            p = PortNamer(2, 2, strategy="optical")
            print(p.o1, p.o2, p.o3, p.o4)
            ```
        """
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_ports = num_inputs + num_outputs
        self.strategy = (strategy or get_port_naming_strategy()).lower()

    def __repr__(self) -> str:
        return f"PortNamer({self.num_inputs}, {self.num_outputs}, '{self.strategy}')"

    def __getattr__(self, attr: str) -> sax.Name:
        idx = int(re.sub("[a-zA-Z_-]", "", attr))
        if attr.startswith("in"):
            if idx >= self.num_inputs:
                msg = (
                    f"{self.num_inputs}x{self.num_outputs} PortNamer does not "
                    f"have an input port {attr}."
                )
                raise AttributeError(msg)
                msg = f"Index {idx} out of range for {self.num_ports} ports."
                raise IndexError(msg)
            return self[idx]
        if attr.startswith("out"):
            if idx >= self.num_outputs:
                msg = (
                    f"{self.num_inputs}x{self.num_outputs} PortNamer does not "
                    f"have an output port {attr}."
                )
                raise AttributeError(msg)
            return self[self.num_ports - idx - 1]
        return self[idx - 1]

    def __getitem__(self, idx: int) -> sax.Name:
        """Get port name by index.

        Args:
            idx: Port index (0-based). Input ports come first (0 to num_inputs-1),
                followed by output ports (num_inputs to num_ports-1).

        Returns:
            Port name string according to the current naming strategy.

        Raises:
            IndexError: If the index is out of range.

        Examples:
            Index-based access:

            ```python
            p = PortNamer(2, 2)  # 2 inputs, 2 outputs
            print(p[0], p[1])  # First two ports (inputs)
            print(p[2], p[3])  # Last two ports (outputs)
            ```
        """
        if idx < 0:
            msg = "Index must be a non-negative integer."
            raise IndexError(msg)

        if idx >= self.num_ports:
            msg = f"Index {idx} out of range for {self.num_ports} ports."
            raise IndexError(msg)

        if self.strategy == "inout":
            if idx < self.num_inputs:
                return f"in{idx}"
            return f"out{self.num_ports - idx - 1}"

        return f"o{idx + 1:.0f}"

    def is_input_port(self, port: sax.Name) -> bool:
        """Check if a port is an input port.

        Args:
            port: Port name to check.

        Returns:
            True if the port is an input port, False otherwise.

        Examples:
            Check port direction:

            ```python
            p = PortNamer(2, 2)
            print(p.is_input_port("in0"))  # True
            print(p.is_input_port("out0"))  # False
            ```
        """
        port = port.lower()
        if port.startswith("in"):
            return True
        idx = int(re.sub("[a-zA-Z_-]", "", port))
        return idx < self.num_inputs

    def is_output_port(self, port: sax.Name) -> bool:
        """Check if a port is an output port.

        Args:
            port: Port name to check.

        Returns:
            True if the port is an output port, False otherwise.

        Examples:
            Check port direction:

            ```python
            p = PortNamer(2, 2)
            print(p.is_output_port("out0"))  # True
            print(p.is_output_port("in0"))  # False
            ```
        """
        return not self.is_input_port(port)

    def is_input_port_idx(self, idx: int) -> bool:
        """Check if a port index is an input port.

        Args:
            idx: Port index to check.

        Returns:
            True if the port index corresponds to an input port, False otherwise.

        Examples:
            Check port direction by index:

            ```python
            p = PortNamer(2, 2)
            print(p.is_input_port_idx(0))  # True (first input)
            print(p.is_input_port_idx(2))  # False (first output)
            ```
        """
        return self.is_input_port(self[idx])

    def is_output_port_idx(self, idx: int) -> bool:
        """Check if a port index is an output port.

        Args:
            idx: Port index to check.

        Returns:
            True if the port index corresponds to an output port, False otherwise.

        Examples:
            Check port direction by index:

            ```python
            p = PortNamer(2, 2)
            print(p.is_output_port_idx(2))  # True (first output)
            print(p.is_output_port_idx(0))  # False (first input)
            ```
        """
        return self.is_output_port(self[idx])
