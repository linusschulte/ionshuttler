from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

from .graph_utils import create_idc_dictionary

if TYPE_CHECKING:
    from .processing_zone import ProcessingZone
    from .types import Edge, Node
    from mqt.ionshuttler.multi_shuttler.outside.types import GateInfo


class Graph(nx.Graph):  # type: ignore [type-arg]
    @property
    def junction_nodes(self) -> list[Node]:
        return self._junction_nodes

    @junction_nodes.setter
    def junction_nodes(self, value: list[Node]) -> None:
        self._junction_nodes = value

    @property
    def pzs(self) -> list[ProcessingZone]:
        return self._pzs

    @pzs.setter
    def pzs(self, value: list[ProcessingZone]) -> None:
        self._pzs_name_map = {pz.name: pz for pz in value}
        self._pzs = value

    @property
    def pzs_name_map(self) -> dict[str, ProcessingZone]:
        return self._pzs_name_map

    @property
    def locked_gates(self) -> dict[int, str]:
        if not hasattr(self, "_locked_gates"):
            self._locked_gates = {}
        return self._locked_gates

    @locked_gates.setter
    def locked_gates(self, value: dict[int, str]) -> None:
        self._locked_gates = value

    @property
    def state(self) -> dict[int, Edge]:
        return self._state

    @state.setter
    def state(self, value: dict[int, Edge]) -> None:
        self._state = value

    @property
    def max_ions_per_pz(self) -> int:
        return self._max_ions_per_pz

    @max_ions_per_pz.setter
    def max_ions_per_pz(self, value: int) -> None:
        self._max_ions_per_pz = value

    @property
    def in_process(self) -> list[int]:
        return self._in_process

    @in_process.setter
    def in_process(self, value: list[int]) -> None:
        self._in_process = value

    @property
    def arch(self) -> str:
        return self._arch

    @arch.setter
    def arch(self, value: str) -> None:
        self._arch = value

    @property
    def sequence(self) -> list[int]:
        return self._sequence

    @sequence.setter
    def sequence(self, value: list[int]) -> None:
        self._sequence = value

    @property
    def plot(self) -> bool:
        return self._plot

    @plot.setter
    def plot(self, value: bool) -> None:
        self._plot = value

    @property
    def save(self) -> bool:
        return self._save

    @save.setter
    def save(self, value: bool) -> None:
        self._save = value

    @property
    def stop_moves(self) -> list[int]:
        return self._stop_moves

    @stop_moves.setter
    def stop_moves(self, value: list[int]) -> None:
        self._stop_moves = value

    @property
    def idc_dict(self) -> dict[int, Edge]:
        if not hasattr(self, "_idc_dict"):
            self._idc_dict = create_idc_dictionary(self)
        return self._idc_dict

    @property
    def map_to_pz(self) -> dict[int, str]:
        return self._map_to_pz

    @map_to_pz.setter
    def map_to_pz(self, value: dict[int, str]) -> None:
        self._map_to_pz = value

    @property
    def gate_info(self) -> dict[int, "GateInfo"]:
        return self._gate_info

    @gate_info.setter
    def gate_info(self, value: dict[int, "GateInfo"]) -> None:
        self._gate_info = value

    @property
    def gate_pz_assignment(self) -> dict[int, str]:
        if not hasattr(self, "_gate_pz_assignment"):
            self._gate_pz_assignment = {}
        return self._gate_pz_assignment

    @gate_pz_assignment.setter
    def gate_pz_assignment(self, value: dict[int, str]) -> None:
        self._gate_pz_assignment = value

    def gate_qubits(self, gate_id: int) -> tuple[int, ...]:
        return self._gate_info[gate_id].qubits

    def preferred_pz_for_gate(self, gate_id: int) -> str | None:
        return self.gate_pz_assignment.get(gate_id)
