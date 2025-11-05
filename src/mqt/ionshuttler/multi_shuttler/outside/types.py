from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


Node = tuple[float, float]
Edge = tuple[Node, Node]


@dataclass(frozen=True, slots=True)
class GateInfo:
    """Immutable metadata carried for each parsed gate."""

    qubits: Tuple[int, ...]
    qasm: str


@dataclass(slots=True)
class ParsedCircuit:
    """Container for the parsed circuit sequence and gate metadata."""

    sequence: List[int]
    gate_info: Dict[int, GateInfo]
