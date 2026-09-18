"""SAX backends. KLU is the default and klujax is a required dependency."""

from __future__ import annotations

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


__all__ = [
    "analyze_circuit",
    "analyze_circuit_additive",
    "analyze_circuit_fg",
    "analyze_instances",
    "analyze_instances_additive",
    "analyze_instances_fg",
    "circuit_backends",
    "evaluate_circuit",
    "evaluate_circuit_additive",
    "evaluate_circuit_fg",
]
