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


@dataclass(slots=True)
class SlicePlan:
    """Plan describing the qubits and gates assigned to each processing zone for one slice."""

    qubits_by_pz: Dict[str, List[int]]
    gates_by_pz: Dict[str, List[int]]

    def __repr__(self) -> str:
        qubits_str = [qubits for pz, qubits in self.qubits_by_pz.items()]
        gates_str = [gates for pz, gates in self.gates_by_pz.items()]
        return f"qubits:{qubits_str}, gates:{gates_str}"