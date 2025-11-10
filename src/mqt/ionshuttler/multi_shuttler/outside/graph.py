from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

from .graph_utils import create_dist_dict, create_idc_dictionary, get_idx_from_idc

if TYPE_CHECKING:
    from .processing_zone import ProcessingZone
    from .types import Edge, GateInfo, Node, SlicePlan


class Graph(nx.Graph):  # type: ignore [type-arg]
    @property
    def mz_graph(self) -> Graph:
        return self._mz_graph

    @mz_graph.setter
    def mz_graph(self, value: Graph) -> None:
        self._mz_graph = value

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, value: int) -> None:
        self._seed = value

    @property
    def idc_dict(self) -> dict[Edge, int]:
        if not hasattr(self, "_idc_dict"):
            self._idc_dict = create_idc_dictionary(self)
        return self._idc_dict

    @property
    def max_num_parking(self) -> int:
        return self._max_num_parking

    @max_num_parking.setter
    def max_num_parking(self, value: int) -> None:
        self._max_num_parking = value

    @property
    def pzs(self) -> list[ProcessingZone]:
        return self._pzs

    @pzs.setter
    def pzs(self, value: list[ProcessingZone]) -> None:
        parking_edges_idxs = []
        pzs_name_map = {}
        edge_to_pz_map = {}

        for pz in value:
            pz.max_num_parking = self.max_num_parking
            parking_idx = get_idx_from_idc(self.idc_dict, pz.parking_edge)
            parking_edges_idxs.append(parking_idx)
            pzs_name_map[pz.name] = pz
            # Populate edge_to_pz_map for edges belonging to this PZ's structure
            for edge_idx in pz.pz_edges_idx:
                edge_to_pz_map[edge_idx] = pz

        self._parking_edges_idxs = parking_edges_idxs
        self._pzs_name_map = pzs_name_map
        self._edge_to_pz_map = edge_to_pz_map
        self._pzs = value

    @property
    def parking_edges_idxs(self) -> list[int]:
        return self._parking_edges_idxs

    @property
    def pzs_name_map(self) -> dict[str, ProcessingZone]:
        return self._pzs_name_map

    @property
    def edge_to_pz_map(self) -> dict[int, ProcessingZone]:
        return self._edge_to_pz_map

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
    def state(self) -> dict[int, Edge]:
        return self._state

    @state.setter
    def state(self, value: dict[int, Edge]) -> None:
        self._state = value

    @property
    def sequence(self) -> list[int]:
        return self._sequence

    @sequence.setter
    def sequence(self, value: list[int]) -> None:
        self._sequence = value

    @property
    def gate_info(self) -> dict[int, GateInfo]:
        return self._gate_info

    @gate_info.setter
    def gate_info(self, value: dict[int, GateInfo]) -> None:
        self._gate_info = value

    @property
    def gate_pz_assignment(self) -> dict[int, str]:
        if not hasattr(self, "_gate_pz_assignment"):
            self._gate_pz_assignment = {}
        return self._gate_pz_assignment

    @gate_pz_assignment.setter
    def gate_pz_assignment(self, value: dict[int, str]) -> None:
        self._gate_pz_assignment = value

    @property
    def dag_gate_id_lookup(self) -> dict[int, int]:
        return self._dag_gate_id_lookup

    @dag_gate_id_lookup.setter
    def dag_gate_id_lookup(self, value: dict[int, int]) -> None:
        self._dag_gate_id_lookup = value

    @property
    def locked_gates(self) -> dict[int, str]:
        return self._locked_gates

    @locked_gates.setter
    def locked_gates(self, value: dict[int, str]) -> None:
        self._locked_gates = value

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
    def map_to_pz(self) -> dict[int, str]:
        return self._map_to_pz

    @map_to_pz.setter
    def map_to_pz(self, value: dict[int, str]) -> None:
        self._map_to_pz = value

    @property
    def next_gate_at_pz(self) -> dict[str, int | None]:
        return self._next_gate_at_pz

    @next_gate_at_pz.setter
    def next_gate_at_pz(self, value: dict[str, int | None]) -> None:
        self._next_gate_at_pz = value

    @property
    def dist_dict(self) -> dict[str, dict[Edge, list[Node]]]:
        if not hasattr(self, "_dist_dict"):
            self._dist_dict = create_dist_dict(self)
        return self._dist_dict

    @dist_dict.setter
    def dist_dict(self, value: dict[str, dict[Edge, list[Node]]]) -> None:
        self._dist_dict = value

    @property
    def junction_nodes(self) -> list[Node]:
        return self._junction_nodes

    @junction_nodes.setter
    def junction_nodes(self, value: list[Node]) -> None:
        self._junction_nodes = value

    def gate_qubits(self, gate_id: int) -> tuple[int, ...]:
        return self._gate_info[gate_id].qubits

    def gate_qasm(self, gate_id: int) -> str:
        return self._gate_info[gate_id].qasm

    def next_gate_qubits(self, pz_name: str) -> tuple[int, ...]:
        gate_id = self._next_gate_at_pz.get(pz_name)
        if gate_id is None:
            return ()
        return self.gate_qubits(gate_id)

    def preferred_pz_for_gate(self, gate_id: int) -> str | None:
        return self.gate_pz_assignment.get(gate_id)

    @property
    def current_gate_by_pz(self) -> dict[str, int]:
        if not hasattr(self, "_current_gate_by_pz"):
            self._current_gate_by_pz = {}
        return self._current_gate_by_pz

    @current_gate_by_pz.setter
    def current_gate_by_pz(self, value: dict[str, int]) -> None:
        self._current_gate_by_pz = value

    def initialize_slice_plan(self, plan: list[SlicePlan] | None) -> None:
        self.slice_plan = plan
        if plan is None:
            self.current_slice_index = 0
            self.slice_remaining_gates = []
            return
        self.current_slice_index = 0
        self.slice_remaining_gates = [
            {gate for gates in slice_info.gates_by_pz.values() for gate in gates}
            for slice_info in plan
        ]
        ordered_gates: list[int] = []
        seen: set[int] = set()
        for slice_info in plan:
            for gates in slice_info.gates_by_pz.values():
                for gate_id in gates:
                    if gate_id not in seen:
                        ordered_gates.append(gate_id)
                        seen.add(gate_id)
        if ordered_gates:
            remaining = [gate_id for gate_id in self.sequence if gate_id not in seen]
            self.sequence = ordered_gates + remaining

    @property
    def slice_plan(self) -> list[SlicePlan] | None:
        return getattr(self, "_slice_plan", None)

    @slice_plan.setter
    def slice_plan(self, value: list[SlicePlan] | None) -> None:
        self._slice_plan = value

    @property
    def current_slice_index(self) -> int:
        return getattr(self, "_current_slice_index", 0)

    @current_slice_index.setter
    def current_slice_index(self, value: int) -> None:
        self._current_slice_index = value

    @property
    def slice_remaining_gates(self) -> list[set[int]]:
        return getattr(self, "_slice_remaining_gates", [])

    @slice_remaining_gates.setter
    def slice_remaining_gates(self, value: list[set[int]]) -> None:
        self._slice_remaining_gates = value
