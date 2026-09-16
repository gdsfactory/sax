"""SAX Backends."""

from __future__ import annotations

import warnings
from collections.abc import Callable

import sax

from .additive import (
    analyze_circuit_additive,
    analyze_instances_additive,
    evaluate_circuit_additive,
)
from .filipsson_gunnar import (
    analyze_circuit_fg,
    analyze_instances_fg,
    evaluate_circuit_fg,
)
from .forward_only import (
    analyze_circuit_forward,
    analyze_instances_forward,
    evaluate_circuit_forward,
)
from .klu import (
    analyze_circuit_klu,
    analyze_instances_klu,
    evaluate_circuit_klu,
)

circuit_backends: dict[sax.Backend, tuple[Callable, Callable, Callable]] = {
    "filipsson_gunnar": (
        analyze_instances_fg,
        analyze_circuit_fg,
        evaluate_circuit_fg,
    ),
    "additive": (
        analyze_instances_additive,
        analyze_circuit_additive,
        evaluate_circuit_additive,
    ),
    "forward": (
        analyze_instances_forward,
        analyze_circuit_forward,
        evaluate_circuit_forward,
    ),
    "klu": (
        analyze_instances_klu,
        analyze_circuit_klu,
        evaluate_circuit_klu,
    ),
}

analyze_instances = analyze_instances_klu
analyze_circuit = analyze_circuit_klu
evaluate_circuit = evaluate_circuit_klu
default_backend = "klu"

try:
    import klujax  # noqa: F401

except ImportError:
    analyze_instances = analyze_instances_fg
    analyze_circuit = analyze_circuit_fg
    evaluate_circuit = evaluate_circuit_fg
    default_backend = "filipsson_gunnar"
    warnings.warn(
        "klujax not found. Please install klujax for "
        "better performance during circuit evaluation!",
        stacklevel=2,
    )

try:
    from .cuda import (
        analyze_circuit_cuda,
        analyze_instances_cuda,
        evaluate_circuit_cuda,
    )

    circuit_backends["cuda"] = (
        analyze_instances_cuda,
        analyze_circuit_cuda,
        evaluate_circuit_cuda,
    )
except ImportError:
    pass


__all__ = [
    "analyze_circuit",
    "analyze_circuit_additive",
    "analyze_circuit_fg",
    "analyze_circuit_forward",
    "analyze_instances",
    "analyze_instances_additive",
    "analyze_instances_fg",
    "analyze_instances_forward",
    "circuit_backends",
    "evaluate_circuit",
    "evaluate_circuit_additive",
    "evaluate_circuit_fg",
    "evaluate_circuit_forward",
]
