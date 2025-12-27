from __future__ import annotations

import itertools
import math
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.patches as patches
import numpy as np
from scipy.spatial import ConvexHull

from .compilation import create_initial_sequence
from .cycles import shortest_path_to_node
from .fgp_roee import FGPResult, _build_time_slices
from .graph import Graph
from .processing_zone import ProcessingZone
from .types import GateInfo, SlicePlan

DEBUG_FLAG = bool(int(os.getenv("IONSHUTTLER_DEBUG_FGP_TABU", "0")))

@dataclass(slots=True)
class Supernode:
    """Collapsed component that groups qubits that share a multi-qubit gate."""

    id: int
    qubits: tuple[int, ...]
    load: int


@dataclass(slots=True)
class ContractionResult:
    """Diagnostic container for the supernode contraction preview."""

    supernodes: list[Supernode]
    qubit_to_supernode: dict[int, int]
    required_edges: dict[tuple[int, int], float]
    required_unary: set[int]
    assignment: list[int] | None
    cluster_loads: list[int] | None


@dataclass(slots=True)
class Partition:
    processing_zone: set[int]
    memory_zone: set[int]
    tbd: set[int]



def fgp_tabu_global(
    graph: Graph,
    *,
    num_pzs: int | None = None,
    capacity: int | None = None,
    capacity_weight: float = 1.0,
    distance_weight: float = 1.0,
    max_iterations: int | None = None,
    tabu_list_length: int | None = None,
    graph_based_distance: bool = False,
    **legacy_params: object,
) -> FGPResult:
    """Public entry point mirroring compute_gate_partition but using tabu refinement."""

    if DEBUG_FLAG:
        print("=== FGP Tabu Parameters ===")
        print(f"num_ions: {_infer_num_qubits(graph.gate_info)}")
        print(f"num_pzs: {num_pzs}")
        print(f"capacity: {capacity}")
        print(f"capacity_weight: {capacity_weight}")
        print(f"distance_weight: {distance_weight}")
        print(f"max_iterations: {max_iterations}")
        print(f"tabu_list_length: {tabu_list_length}")
        print(f"graph_based_distance: {graph_based_distance}")
        if legacy_params:
            ignored = ", ".join(sorted(legacy_params))
            print(f"Ignoring legacy parameters: {ignored}")
        print()

    if not graph.sequence:
        gate_partition_by_pz = {pz.name: [] for pz in graph.pzs}
        return FGPResult([], gate_partition_by_pz, {}, [], [], time_slices=[])

    gate_info = graph.gate_info
    num_pzs = num_pzs or len(graph.pzs)
    if num_pzs <= 0:
        raise ValueError("Number of processing zones must be positive.")


    num_qubits = _infer_num_qubits(gate_info)
    capacity = max(capacity or math.ceil(num_qubits / num_pzs), 1)
    pz_names = [pz.name for pz in graph.pzs]
    pz_positions: Sequence[ProcessingZone | None] = [graph.pzs_name_map.get(name) for name in pz_names]
    pz_distance_map = _build_pz_distance_map(graph, pz_positions, graph_based_distance=graph_based_distance)
    if not max_iterations or max_iterations <= 0:
        max_iterations = 50
    if not tabu_list_length or tabu_list_length <= 0:
        tabu_list_length = 20
    

    partition_output = _run_fgp_tabu(
        graph.sequence,
        gate_info,
        num_qubits=num_qubits,
        num_pzs=num_pzs,
        capacity=capacity,
        pz_names=pz_names,
        max_iterations=max_iterations,
        tabu_list_length=tabu_list_length,
        capacity_weight=capacity_weight,
        distance_weight=distance_weight,
        pz_positions=pz_positions,
        pz_distance_map=pz_distance_map,
    )

    if DEBUG_FLAG:
        print("Overview:")
        for idx, slice in enumerate(partition_output["peeled_subslices"]):
            print(f"Slice {idx+1}:", partition_output["slice_plan"][idx])
            for subslice in slice[:1]:
                all_qubits = []
                for partition in subslice["partitions"]:
                    all_qubits_partition = []
                    for v in partition.values():
                        all_qubits_partition.extend(v)
                    all_qubits.append(all_qubits_partition)
                print(f"hidden partitioning: {all_qubits}")

        for idx, result in []: #enumerate(partition_output["partition_results"]):
            print(f"\n=== Slice {idx+1} ===")
            print(f"Gates in slice: {partition_output['time_slices'][idx]}")
            
            if result.assignment is not None:
                print("\nQubit assignments by PZ:")
                qubit_to_pz = {}
                for sn in result.supernodes:
                    pz = result.assignment[sn.id]
                    for q in sn.qubits:
                        qubit_to_pz[q] = pz
                
                for pz_idx in range(num_pzs):
                    qubits_in_pz = sorted([q for q, p in qubit_to_pz.items() if p == pz_idx])
                    print(f"  PZ{pz_idx}: {qubits_in_pz}")
                
                print("\nGate assignments by PZ:")
                for pz_idx in range(num_pzs):
                    gates_in_pz = []
                    for gate_id in partition_output["time_slices"][idx]:
                        gate_qubits = gate_info[gate_id].qubits
                        if gate_qubits and qubit_to_pz.get(gate_qubits[0]) == pz_idx:
                            gates_in_pz.append(gate_id)
                    print(f"  PZ{pz_idx}: {gates_in_pz}")
                
                if result.cluster_loads:
                    print(f"\nCluster loads: {result.cluster_loads}")

    assignments: list[list[int]] = partition_output["qubit_assignments"]  # type: ignore[assignment]
    slice_plan: list[SlicePlan] = partition_output["slice_plan"]  # type: ignore[assignment]
    gate_partition_by_pz: dict[str, list[int]] = partition_output["gate_partition_by_pz"]  # type: ignore[assignment]
    gate_assignment: dict[int, str] = partition_output["gate_assignment"]  # type: ignore[assignment]
    time_slices: list[list[int]] = partition_output.get("time_slices", [])  # type: ignore[assignment]
    pre_cost: float = sum(partition_output["pre_partitioning_costs"])  # type: ignore[assignment]
    post_cost: float = sum(partition_output["post_partitioning_costs"])  # type: ignore[assignment]

    moves = _compute_moves(assignments)
    move_distance_total = _compute_total_move_distance(moves, pz_positions, pz_distance_map)

    return FGPResult(
        assignments,
        gate_partition_by_pz,
        gate_assignment,
        moves,
        slice_plan,
        pre_cost,
        post_cost,
        time_slices=time_slices,
        move_distance_total=move_distance_total,
    )


def partition_slice(
    gate_info: dict[int, GateInfo],
    slice_gate_ids: Sequence[int],
    num_qubits: int | None = None,
    *,
    num_pzs: int | None = None,
    previous_qubit_assignment: list[int] | None = None,
) -> ContractionResult:
    """Contract the slice graph and produce a greedy initial assignment."""

    if not slice_gate_ids:
        raise ValueError("Slice must contain at least one gate to contract.")

    required_edges = _build_edge_weights(slice_gate_ids, gate_info)
    required_unary = {
        gate_info[gate_id].qubits[0]
        for gate_id in slice_gate_ids
        if len(gate_info[gate_id].qubits) == 1
    }

    if not num_qubits:
        raise ValueError("num_qubits must be provided or inferable from gate_info")
    qubits = list(range(num_qubits))
    

    # perform contraction of qubits sharing multi-qubit gates into supernodes
    supernodes, qubit_to_supernode = _contract_supernodes(qubits, required_edges)
    

    # greedy initial partitioning
    assignment: list[int] | None = None
    cluster_loads: list[int] | None = None
    if previous_qubit_assignment is not None:
        assignment, cluster_loads = _seed_assignment_from_previous(
            supernodes,
            previous_qubit_assignment,
            num_pzs,
        )

    if assignment is None:
        assignment, cluster_loads = _greedy_initial_partition(supernodes, num_pzs)

    result = ContractionResult(
        supernodes=supernodes,
        qubit_to_supernode=qubit_to_supernode,
        required_edges=required_edges,
        required_unary=required_unary,
        assignment=assignment,
        cluster_loads=cluster_loads,
    )
        
    return result


def peel_slice(
    result: ContractionResult,
    gate_info: dict[int, GateInfo],
    slice_gate_ids: Sequence[int],
    capacities: Sequence[int],
) -> list[dict[str, object]]:
    """Peel an overfull slice into capacity-respecting subslices.

    Returns a list of subslices, each containing a snapshot of partition buckets and
    the gates that can be performed in that subslice.
    """
    if result.assignment is None:
        raise ValueError("Cannot peel without an initial assignment.")
    num_pzs = len(capacities)
    partitions: list[Partition] = [
        Partition(processing_zone=set(), memory_zone=set(), tbd=set()) for _ in range(num_pzs)
    ]

    required_qubits = {q for gid in slice_gate_ids for q in gate_info[gid].qubits}
    # Initialize TBD/memory from assignment
    for sn in result.supernodes:
        cluster = result.assignment[sn.id]
        if cluster < 0 or cluster >= num_pzs:
            raise ValueError(f"Supernode {sn.id} assigned to invalid cluster {cluster}")
        for q in sn.qubits:
            if q in required_qubits:
                partitions[cluster].tbd.add(q)
            else:
                partitions[cluster].memory_zone.add(q)

    required_gates: set[int] = set(slice_gate_ids)
    subslices: list[dict[str, object]] = []

    while required_gates:
        progress = False

        # Select qubits into processing_zone up to capacity, lowest weight first
        for pz_idx, partition in enumerate(partitions):
            if capacities[pz_idx] <= 0:
                continue
            candidates: dict[int, set[int]] = defaultdict(set)
            for q in partition.tbd:
                sn_id = result.qubit_to_supernode[q]
                candidates[sn_id].add(q)
            ordered = sorted(candidates.items(), key=lambda kv: len(kv[1]))
            remaining_cap = max(capacities[pz_idx] - len(partition.processing_zone), 0)

            for sn_id, qs in ordered:
                if remaining_cap < len(qs):
                    continue
                take = min(len(qs), remaining_cap)
                selected = set(list(qs)[:take])
                partition.processing_zone.update(selected)
                partition.tbd.difference_update(selected)
                remaining_cap -= take
                progress = progress or bool(selected)

        performed: set[int] = set()
        performed_by_pz: dict[int, list[int]] = defaultdict(list)
        for gate_id in list(required_gates):
            qubits = gate_info[gate_id].qubits
            hosting_pz = None
            for pz_idx, partition in enumerate(partitions):
                if qubits[0] in partition.processing_zone:
                    hosting_pz = pz_idx
                    break
            if hosting_pz is None:
                continue
            if all(q in partitions[hosting_pz].processing_zone for q in qubits):
                performed.add(gate_id)
                performed_by_pz[hosting_pz].append(gate_id)

        subslices.append(
            {
                "partitions": [
                    {
                        "processing_zone": sorted(p.processing_zone),
                        "memory_zone": sorted(p.memory_zone),
                        "tbd": sorted(p.tbd),
                    }
                    for p in partitions
                ],
                "gates_by_pz": {
                    idx: sorted(performed_by_pz.get(idx, [])) for idx in range(num_pzs)
                },
            }
        )

        required_gates.difference_update(performed)
        for p in partitions:
            p.memory_zone.update(p.processing_zone)
            p.processing_zone.clear()

        if not progress and not performed and required_gates:
            raise RuntimeError("Peeling stalled: no progress but required gates remain.")

    return subslices


def tabu_optimize_global(
    partition_results: Sequence[ContractionResult],
    *,
    num_pzs: int,
    capacity: int | None,
    capacity_weight: float,
    distance_weight: float,
    max_iterations: int,
    tabu_list_length: int,
    num_qubits: int,
    pz_positions: Sequence[ProcessingZone | None] | None,
    pz_distance_map: dict[tuple[str, str], float] | None,
) -> tuple[list[list[int]], float, float]:
    """Run a single tabu search over all slices simultaneously."""

    if not partition_results or num_pzs <= 0:
        return [], 0.0, 0.0

    assignments_by_slice: list[list[int]] = []
    slice_loads: list[list[int]] = []
    qubit_assignments_by_slice: list[list[int]] = [
        [-1] * num_qubits for _ in partition_results
    ]

    for slice_idx, result in enumerate(partition_results):
        if result.assignment is None:
            raise ValueError("Partition result missing initial assignment.")
        assignment_copy = result.assignment.copy()
        assignments_by_slice.append(assignment_copy)
        loads = [0] * num_pzs
        for sn in result.supernodes:
            cluster = assignment_copy[sn.id]
            if cluster < 0 or cluster >= num_pzs:
                raise ValueError(f"Supernode {sn.id} assigned to invalid cluster {cluster}.")
            loads[cluster] += sn.load
            for qubit in sn.qubits:
                if 0 <= qubit < num_qubits:
                    qubit_assignments_by_slice[slice_idx][qubit] = cluster
        slice_loads.append(loads)

    capacity_cost = _compute_capacity_cost(slice_loads, capacity)
    distance_cost = _compute_distance_cost(
        qubit_assignments_by_slice,
        pz_positions,
        pz_distance_map,
    )
    current_cost = capacity_weight * capacity_cost + distance_weight * distance_cost
    initial_cost = current_cost
    best_cost = current_cost
    best_assignments = [assignment.copy() for assignment in assignments_by_slice]

    tabu_list: list[tuple[int, int, int]] = []
    move_counts_per_iteration: list[int] = []
    move_histograms_per_iteration: list[dict[float, int]] = []
    for _ in range(max_iterations):
        best_move: tuple[int, Supernode, int] | None = None
        best_move_score = math.inf
        best_move_capacity_delta = 0.0
        best_move_distance_delta = 0.0
        move_histogram: dict[float, int] = {}
        total_moves = 0

        for slice_idx, result in enumerate(partition_results):
            for sn in result.supernodes:
                current_cluster = assignments_by_slice[slice_idx][sn.id]
                for target_cluster in range(num_pzs):
                    if target_cluster == current_cluster:
                        continue
                    delta_capacity = _capacity_delta(
                        slice_loads[slice_idx],
                        current_cluster,
                        target_cluster,
                        sn.load,
                        capacity,
                    )
                    delta_distance = _distance_delta(
                        slice_idx,
                        sn.qubits,
                        current_cluster,
                        target_cluster,
                        qubit_assignments_by_slice,
                        pz_positions,
                        pz_distance_map,
                    )
                    move_delta = capacity_weight * delta_capacity + distance_weight * delta_distance
                    candidate_cost = current_cost + move_delta
                    bucket = round(move_delta, 6)
                    move_histogram[bucket] = move_histogram.get(bucket, 0) + 1
                    total_moves += 1
                    move_key = (slice_idx, sn.id, target_cluster)
                    if move_key in tabu_list and candidate_cost >= best_cost:
                        continue
                    if candidate_cost < best_move_score:
                        best_move_score = candidate_cost
                        best_move = (slice_idx, sn, target_cluster)
                        best_move_capacity_delta = delta_capacity
                        best_move_distance_delta = delta_distance

        move_counts_per_iteration.append(total_moves)
        move_histograms_per_iteration.append(move_histogram)
        print(f"Iteration {len(move_counts_per_iteration)}: total_moves={total_moves}")
        for bucket in sorted(move_histogram):
            print(f"  delta={bucket:.6f} count={move_histogram[bucket]}")

        if best_move is None:
            break

        slice_idx, sn, target_cluster = best_move
        previous_cluster = assignments_by_slice[slice_idx][sn.id]
        assignments_by_slice[slice_idx][sn.id] = target_cluster
        slice_loads[slice_idx][previous_cluster] -= sn.load
        slice_loads[slice_idx][target_cluster] += sn.load
        for qubit in sn.qubits:
            if 0 <= qubit < num_qubits:
                qubit_assignments_by_slice[slice_idx][qubit] = target_cluster

        capacity_cost += best_move_capacity_delta
        distance_cost += best_move_distance_delta
        current_cost = best_move_score

        tabu_list.append((slice_idx, sn.id, previous_cluster))
        if len(tabu_list) > tabu_list_length:
            tabu_list.pop(0)

        if current_cost < best_cost:
            best_cost = current_cost
            best_assignments = [assignment.copy() for assignment in assignments_by_slice]

    final_assignments = best_assignments if best_assignments else assignments_by_slice
    return final_assignments, initial_cost, best_cost


def _capacity_delta(
    load_vector: Sequence[int],
    src_cluster: int,
    dst_cluster: int,
    supernode_load: int,
    capacity: int | None,
) -> float:
    if capacity is None:
        return 0.0
    if not (0 <= src_cluster < len(load_vector) and 0 <= dst_cluster < len(load_vector)):
        return 0.0
    before = max(0, load_vector[src_cluster] - capacity) + max(0, load_vector[dst_cluster] - capacity)
    after = max(0, load_vector[src_cluster] - supernode_load - capacity) + max(
        0, load_vector[dst_cluster] + supernode_load - capacity
    )
    return float(after - before)


def _distance_delta(
    slice_idx: int,
    qubits: Sequence[int],
    current_cluster: int,
    target_cluster: int,
    qubit_assignments_by_slice: Sequence[Sequence[int]],
    pz_positions: Sequence[ProcessingZone | None] | None,
    pz_distance_map: dict[tuple[str, str], float] | None,
) -> float:
    delta = 0.0
    num_slices = len(qubit_assignments_by_slice)
    if num_slices <= 1:
        return 0.0
    current_slice_assignments = qubit_assignments_by_slice[slice_idx]
    if slice_idx > 0:
        previous_assignments = qubit_assignments_by_slice[slice_idx - 1]
        for q in qubits:
            if q < 0 or q >= len(current_slice_assignments):
                continue
            prev_cluster = previous_assignments[q]
            curr_cluster = current_slice_assignments[q]
            delta += _distance_between_clusters(
                prev_cluster,
                target_cluster,
                pz_positions,
                pz_distance_map,
            ) - _distance_between_clusters(
                prev_cluster,
                curr_cluster,
                pz_positions,
                pz_distance_map,
            )
    if slice_idx < num_slices - 1:
        next_assignments = qubit_assignments_by_slice[slice_idx + 1]
        for q in qubits:
            if q < 0 or q >= len(current_slice_assignments):
                continue
            next_cluster = next_assignments[q]
            curr_cluster = current_slice_assignments[q]
            delta += _distance_between_clusters(
                target_cluster,
                next_cluster,
                pz_positions,
                pz_distance_map,
            ) - _distance_between_clusters(
                curr_cluster,
                next_cluster,
                pz_positions,
                pz_distance_map,
            )
    return delta


def _compute_capacity_cost(
    slice_loads: Sequence[Sequence[int]],
    capacity: int | None,
) -> float:
    if capacity is None:
        return 0.0
    total = 0.0
    for loads in slice_loads:
        for load in loads:
            total += max(0, load - capacity)
    return float(total)


def _compute_distance_cost(
    qubit_assignments_by_slice: Sequence[Sequence[int]],
    pz_positions: Sequence[ProcessingZone | None] | None,
    pz_distance_map: dict[tuple[str, str], float] | None,
) -> float:
    if len(qubit_assignments_by_slice) <= 1:
        return 0.0
    total = 0.0
    for idx in range(len(qubit_assignments_by_slice) - 1):
        current = qubit_assignments_by_slice[idx]
        nxt = qubit_assignments_by_slice[idx + 1]
        for qubit, current_cluster in enumerate(current):
            next_cluster = nxt[qubit] if qubit < len(nxt) else -1
            total += _distance_between_clusters(
                current_cluster,
                next_cluster,
                pz_positions,
                pz_distance_map,
            )
    return float(total)


def _distance_between_clusters(
    src_cluster: int,
    dst_cluster: int,
    pz_positions: Sequence[ProcessingZone | None] | None,
    pz_distance_map: dict[tuple[str, str], float] | None,
) -> float:
    if src_cluster == dst_cluster or src_cluster < 0 or dst_cluster < 0:
        return 0.0
    if (
        pz_positions
        and 0 <= src_cluster < len(pz_positions)
        and 0 <= dst_cluster < len(pz_positions)
    ):
        src_pz = pz_positions[src_cluster]
        dst_pz = pz_positions[dst_cluster]
        if src_pz is not None and dst_pz is not None:
            return get_pz_distance(src_pz, dst_pz, pz_distance_map)
    return 1.0


def _compute_moves(assignments: list[list[int]]) -> list[list[tuple[int, int, int]]]:
    """Identify qubits that change clusters between consecutive time slices."""
    moves: list[list[tuple[int, int, int]]] = []
    for previous, current in zip(assignments, assignments[1:]):
        slice_moves: list[tuple[int, int, int]] = []
        for qubit, (src, dst) in enumerate(zip(previous, current)):
            if src != dst:
                slice_moves.append((qubit, src, dst))
        moves.append(slice_moves)
    return moves


def _compute_total_move_distance(
    moves: list[list[tuple[int, int, int]]],
    pz_positions: Sequence[ProcessingZone | None] | None,
    pz_distance_map: dict[tuple[str, str], float] | None,
) -> float:
    """Sum distance of all moves between consecutive slices."""
    if not moves:
        return 0.0
    total = 0.0
    for slice_moves in moves:
        for _, src, dst in slice_moves:
            dist = 1.0
            if pz_positions and 0 <= src < len(pz_positions) and 0 <= dst < len(pz_positions):
                src_pz = pz_positions[src]
                dst_pz = pz_positions[dst]
                if src_pz and dst_pz:
                    dist = get_pz_distance(src_pz, dst_pz, pz_distance_map)
            total += dist
    return total


def _compute_cluster_loads(result: ContractionResult, num_pzs: int) -> list[int]:
    loads = [0] * num_pzs
    if result.assignment is None:
        return loads
    for sn in result.supernodes:
        cluster = result.assignment[sn.id]
        if 0 <= cluster < num_pzs:
            loads[cluster] += sn.load
    return loads


def _build_qubit_assignment(
    result: ContractionResult,
    num_qubits: int,
    num_clusters: int,
) -> list[int] | None:
    if result.assignment is None:
        return None
    qubit_assignment = [-1] * num_qubits
    for sn in result.supernodes:
        cluster = result.assignment[sn.id]
        if cluster < 0 or cluster >= num_clusters:
            continue
        for q in sn.qubits:
            if 0 <= q < num_qubits:
                qubit_assignment[q] = cluster
    return qubit_assignment


def plot_partition_outputs(
    num_qubits: int,
    result: ContractionResult,
    gate_info: dict[int, GateInfo],
    slice_gate_ids: Sequence[int],
    output_dir: Path,
) -> None:
    
    """Write before/after/partition plots for a single slice."""
    qubits = range(num_qubits)
    required_unary_qubits = {
        gate_info[gid].qubits[0]
        for gid in slice_gate_ids
        if len(gate_info[gid].qubits) == 1
    }
    required_edge_qubits = {
        qubit 
        for gid in slice_gate_ids 
        if len(gate_info[gid].qubits) == 2
        for qubit in gate_info[gid].qubits
    }

    required_qubits = required_unary_qubits | required_edge_qubits

    prefix = f"slice_{slice_gate_ids[0]}_{slice_gate_ids[-1]}"
    output_dir.mkdir(parents=True, exist_ok=True)

    before_path = output_dir / f"{prefix}_before.png"
    _plot_interaction_graph(
        qubits,
        result.required_edges,
        {},
        before_path,
        node_label="q",
        highlighted_nodes=required_unary_qubits,
        node_weights={},
    )
    after_path = output_dir / f"{prefix}_contracted.png"
    _plot_interaction_graph(
        result.supernodes,
        {},
        {},
        after_path,
        node_label="S",
        highlighted_nodes=required_qubits,
        node_weights={},
    )

    if result.assignment is not None:
        assign_path = output_dir / f"{prefix}_partition.png"
        
        _plot_interaction_graph(
            result.supernodes,
            {},
            {},
            assign_path,
            node_label="S",
            highlighted_nodes=required_qubits,
            node_weights={},
            assignment=result.assignment
        )

class _UnionFind:
    def __init__(self, elements: Iterable[int]) -> None:
        self.parent: dict[int, int] = {}
        self.rank: dict[int, int] = {}
        for element in elements:
            self.parent[element] = element
            self.rank[element] = 0

    def find(self, item: int) -> int:
        parent = self.parent.setdefault(item, item)
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: int, b: int) -> bool:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return False
        rank_a = self.rank.setdefault(root_a, 0)
        rank_b = self.rank.setdefault(root_b, 0)
        if rank_a < rank_b:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if rank_a == rank_b:
            self.rank[root_a] = rank_a + 1
        return True
    


def _run_fgp_tabu(
    sequence: Sequence[int],
    gate_info: dict[int, GateInfo],
    *,
    num_qubits: int,
    num_pzs: int | None,
    capacity: int | None,
    pz_names: Sequence[str] | None,
    enable_plots: bool = False,
    output_dir: Path | None = Path("outputs/fgp_tabu"),
    max_iterations: int = 50,
    tabu_list_length: int = 20,
    capacity_weight: float = 1.0,
    distance_weight: float = 1.0,
    pz_positions: Sequence[ProcessingZone | None] | None = None,
    pz_distance_map: dict[tuple[str, str], float] | None = None,

) -> dict[str, object]:
    """Core partition routine shared by CLI and API entry points."""

    time_slices = _build_time_slices(sequence, gate_info, num_qubits)
    partitioning_results: list[ContractionResult] = []
    prev_qubit_assignment: list[int] | None = None


    for idx, current_slice in enumerate(time_slices):
        result = partition_slice(
            gate_info,
            current_slice,
            num_qubits=num_qubits,
            num_pzs=num_pzs,
            previous_qubit_assignment=prev_qubit_assignment,
        )
        partitioning_results.append(result)
        if num_pzs:
            prev_qubit_assignment = _build_qubit_assignment(result, num_qubits, num_pzs)

    pre_cost = 0.0
    post_cost = 0.0
    peeled_subslices: list[list[dict[str, object]] | None] = [None] * len(time_slices)

    if num_pzs:
        (
            optimized_assignments,
            initial_cost,
            best_cost,
        ) = tabu_optimize_global(
            partitioning_results,
            num_pzs=num_pzs,
            capacity=capacity,
            capacity_weight=capacity_weight,
            distance_weight=distance_weight,
            max_iterations=max_iterations,
            tabu_list_length=tabu_list_length,
            num_qubits=num_qubits,
            pz_positions=pz_positions,
            pz_distance_map=pz_distance_map,
        )
        for slice_idx, result in enumerate(partitioning_results):
            result.assignment = optimized_assignments[slice_idx]
            result.cluster_loads = _compute_cluster_loads(result, num_pzs)
        pre_cost = initial_cost
        post_cost = best_cost

    if enable_plots and output_dir is not None:
        # Clear the entire output directory before generating new plots
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for idx, current_slice in enumerate(time_slices):
        result = partitioning_results[idx]
        if enable_plots and output_dir is not None:
            plot_partition_outputs(
                num_qubits,
                result,
                gate_info,
                current_slice,
                output_dir,
            )
        if capacity and num_pzs:
            capacities = [capacity] * num_pzs
            subslices = peel_slice(result, gate_info, current_slice, capacities)
            peeled_subslices[idx] = subslices

    assignments: list[list[int]] = []
    gate_partition_by_pz: dict[str, list[int]] = {}
    gate_assignment: dict[int, str] = {}
    slice_plans: list[SlicePlan] = []
    if num_pzs:
        resolved_pz_names = list(pz_names) if pz_names else [f"PZ{i}" for i in range(num_pzs)]
        peeled_for_helper: Sequence[Sequence[dict[str, object]]] | None = None
        if any(peeled_subslices):
            peeled_for_helper = [
                subslice if subslice is not None else []
                for subslice in peeled_subslices
            ]        
        (
            assignments,
            slice_plans,
            gate_partition_by_pz,
            gate_assignment,
        ) = _build_slice_plans_from_results(
            partitioning_results,
            time_slices,
            gate_info,
            resolved_pz_names,
            num_qubits=num_qubits,
            peeled_subslices=peeled_for_helper,
        )


    return {
        "partition_results": partitioning_results,
        "qubit_assignments": assignments,
        "gate_partition_by_pz": gate_partition_by_pz,
        "gate_assignment": gate_assignment,
        "slice_plan": slice_plans,
        "peeled_subslices": [
            subslice if subslice is not None else [] for subslice in peeled_subslices
        ],
        "time_slices": time_slices,
        "pre_partitioning_costs": [pre_cost],
        "post_partitioning_costs": [post_cost],
    }


def _align_clusters_to_previous(
    partitioning_results: Sequence[ContractionResult],
    time_slices: Sequence[Sequence[int]],
    gate_info: dict[int, GateInfo],
    num_qubits: int,
    num_pzs: int,
    *,
    processing_only: bool = False,
) -> Sequence[ContractionResult]:
    prev_qubit_assignment = [-1] * num_qubits

    for slice_idx, result in enumerate(partitioning_results):
        if result.assignment is None:
            prev_qubit_assignment = [-1] * num_qubits
            continue

        if processing_only:
            active_qubits = {
                q for gate_id in time_slices[slice_idx] for q in gate_info[gate_id].qubits
            }
        else:
            active_qubits = None

        counts = [[0] * num_pzs for _ in range(num_pzs)]
        for sn in result.supernodes:
            cluster = result.assignment[sn.id]
            if cluster < 0 or cluster >= num_pzs:
                continue
            for q in sn.qubits:
                if active_qubits is not None and q not in active_qubits:
                    continue
                prev = prev_qubit_assignment[q]
                if 0 <= prev < num_pzs:
                    counts[cluster][prev] += 1

        best_perm = list(range(num_pzs))
        best_score = sum(counts[i][best_perm[i]] for i in range(num_pzs))
        
        # TODO: Improve this, because its combinatorically expensive in num_pzs
        for perm in itertools.permutations(range(num_pzs)):
            score = sum(counts[i][perm[i]] for i in range(num_pzs))
            if score > best_score:
                best_score = score
                best_perm = list(perm)

        if best_perm != list(range(num_pzs)):
            new_assignment = result.assignment.copy()
            for sn in result.supernodes:
                cluster = result.assignment[sn.id]
                if 0 <= cluster < num_pzs:
                    new_assignment[sn.id] = best_perm[cluster]
            result.assignment = new_assignment

        cluster_loads = [0] * num_pzs
        for sn in result.supernodes:
            cluster = result.assignment[sn.id]
            if 0 <= cluster < num_pzs:
                cluster_loads[cluster] += sn.load
            for q in sn.qubits:
                if active_qubits is not None and q not in active_qubits:
                    continue
                prev_qubit_assignment[q] = cluster
        result.cluster_loads = cluster_loads

    return partitioning_results  


def _load_gate_metadata(qasm_path: Path) -> tuple[list[int], dict[int, GateInfo]]:
    parsed = create_initial_sequence(qasm_path)
    return parsed.sequence, parsed.gate_info


def _infer_num_qubits(gate_info: dict[int, GateInfo]) -> int:
    max_qubit = -1
    for info in gate_info.values():
        if info.qubits:
            max_qubit = max(max_qubit, max(info.qubits))
    if max_qubit < 0:
        raise ValueError("Unable to infer qubit count from gate metadata.")

    return max_qubit + 1


def _build_edge_weights(
    gate_ids: Sequence[int],
    gate_info: dict[int, GateInfo],
) -> dict[tuple[int, int], float]:
    weights: dict[tuple[int, int], float] = {}
    for gate_id in gate_ids:
        qubits = gate_info[gate_id].qubits
        if len(qubits) != 2:
            continue
        edge = tuple(sorted(qubits))
        weights[edge] = weights.get(edge, 0.0) + 1.0
    return weights


def _contract_supernodes(
    qubits: Sequence[int],
    required_edges: dict[tuple[int, int], float],
) -> tuple[list[Supernode], dict[int, int]]:
    uf = _UnionFind(qubits)
    for u, v in required_edges:
        uf.union(u, v)

    components: dict[int, list[int]] = {}
    for qubit in qubits:
        root = uf.find(qubit)
        components.setdefault(root, []).append(qubit)

    supernodes: list[Supernode] = []
    qubit_to_supernode: dict[int, int] = {}
    for idx, nodes in enumerate(components.values()):
        nodes_sorted = tuple(sorted(nodes))
        supernodes.append(Supernode(id=idx, qubits=nodes_sorted, load=len(nodes_sorted)))
        for qubit in nodes_sorted:
            qubit_to_supernode[qubit] = idx
    return supernodes, qubit_to_supernode


def _seed_assignment_from_previous(
    supernodes: list[Supernode],
    previous_qubit_assignment: list[int],
    num_pzs: int,
) -> tuple[list[int] | None, list[int] | None]:
    if not previous_qubit_assignment:
        return None, None

    assignment = [-1] * len(supernodes)
    cluster_loads = [0] * num_pzs

    for sn in supernodes:
        prev_clusters: dict[int, int] = defaultdict(int)
        for q in sn.qubits:
            prev_cluster = previous_qubit_assignment[q] if 0 <= q < len(previous_qubit_assignment) else -1
            if 0 <= prev_cluster < num_pzs:
                prev_clusters[prev_cluster] += 1
        if not prev_clusters:
            return None, None
        target_cluster = max(prev_clusters.items(), key=lambda kv: kv[1])[0]
        assignment[sn.id] = target_cluster
        cluster_loads[target_cluster] += sn.load

    return assignment, cluster_loads


def _greedy_initial_partition(
    supernodes: list[Supernode],
    num_pzs: int,
) -> tuple[list[int], list[int]]:
    """Assign supernodes to PZs by balancing load."""
    if num_pzs <= 0:
        raise ValueError("Number of processing zones must be positive for initial partitioning.")
    assignment = [-1] * len(supernodes)
    cluster_loads = [0] * num_pzs

    for sn in sorted(supernodes, key=lambda node: node.load, reverse=True):
        target_cluster = min(range(num_pzs), key=lambda idx: cluster_loads[idx])
        assignment[sn.id] = target_cluster
        cluster_loads[target_cluster] += sn.load

    return assignment, cluster_loads

def get_pz_distance(
    pz1: ProcessingZone,
    pz2: ProcessingZone,
    pz_distance_map: dict[tuple[str, str], float] | None = None,
) -> float:
    if pz1.name == pz2.name:
        return 0.0
    if pz_distance_map:
        key = (pz1.name, pz2.name)
        if key in pz_distance_map:
            return pz_distance_map[key]
        key_rev = (pz2.name, pz1.name)
        if key_rev in pz_distance_map:
            return pz_distance_map[key_rev]
    return math.dist(pz1.processing_zone, pz2.processing_zone)


def _build_pz_distance_map(
    graph: Graph,
    pz_positions: Sequence[ProcessingZone | None],
    *,
    graph_based_distance: bool = True,
) -> dict[tuple[str, str], float]:
    distance_map: dict[tuple[str, str], float] = {}
    if not graph_based_distance:
        for i, pz_i in enumerate(pz_positions):
            if pz_i is None:
                continue
            for j in range(i + 1, len(pz_positions)):
                pz_j = pz_positions[j]
                if pz_j is None:
                    continue
                key = (pz_i.name, pz_j.name)
                distance_map[key] = math.dist(pz_i.processing_zone, pz_j.processing_zone)
        return distance_map

    for i, pz_i in enumerate(pz_positions):
        if pz_i is None:
            continue
        for j in range(i + 1, len(pz_positions)):
            pz_j = pz_positions[j]
            if pz_j is None:
                continue
            key = (pz_i.name, pz_j.name)
            try:
                path = shortest_path_to_node(
                    graph,
                    pz_i.processing_zone,
                    pz_j.processing_zone,
                    exclude_first_entry_connection=True,
                )
                distance_map[key] = max(len(path) - 1, 0) if path else math.dist(
                    pz_i.processing_zone, pz_j.processing_zone
                )
            except Exception:
                distance_map[key] = math.dist(pz_i.processing_zone, pz_j.processing_zone)
    return distance_map


def _build_slice_plans_from_results(
    partition_results: Sequence[ContractionResult],
    time_slices: Sequence[Sequence[int]],
    gate_info: dict[int, GateInfo],
    pz_names: Sequence[str],
    *,
    num_qubits: int,
    peeled_subslices: Sequence[Sequence[dict[str, object]]] | None = None,
) -> tuple[list[list[int]], list[SlicePlan], dict[str, list[int]], dict[int, str]]:
    """Convert contraction outputs into slice plans and gate assignments."""

    if not pz_names:
        msg = "No processing zone names provided to build slice plans. Defaulting to 'pz0, pz1, ...'."
        pz_names = [f"pz{i}" for i in range(len(partition_results))]
    if len(partition_results) != len(time_slices):
        msg = "Partition results and time slices must have matching length."
        raise ValueError(msg)

    slice_plans: list[SlicePlan] = []
    gate_partition_by_pz: dict[str, list[int]] = {pz: [] for pz in pz_names}
    gate_assignment: dict[int, str] = {}
    per_slice_qubit_assignments: list[list[int]] = []
    num_clusters = len(pz_names)

    peeled_iter = peeled_subslices if peeled_subslices is not None else [None] * len(partition_results)

    def _append_slice_plan(
        qubits_by_pz_local: dict[str, list[int]],
        gates_by_pz_local: dict[str, list[int]],
    ) -> None:
        slice_plans.append(SlicePlan(qubits_by_pz=qubits_by_pz_local, gates_by_pz=gates_by_pz_local))
        for pz_name, gates in gates_by_pz_local.items():
            gate_partition_by_pz[pz_name].extend(gates)
            for gate_id in gates:
                gate_assignment[gate_id] = pz_name

    for idx, (result, slice_gate_ids) in enumerate(zip(partition_results, time_slices)):
        if result.assignment is None:
            msg = "Partition result is missing a finalized assignment."
            raise ValueError(msg)
        if len(result.assignment) != len(result.supernodes):
            msg = "Assignment length mismatch when building slice plans."
            raise ValueError(msg)

        qubit_assignment = [-1] * num_qubits
        for sn in result.supernodes:
            cluster = result.assignment[sn.id]
            if cluster < 0 or cluster >= num_clusters:
                msg = f"Supernode {sn.id} assigned to invalid cluster {cluster}."
                raise ValueError(msg)
            pz_name = pz_names[cluster]
            for qubit in sn.qubits:
                qubit_assignment[qubit] = cluster

        per_slice_qubit_assignments.append(qubit_assignment)

        subslices_for_slice = peeled_iter[idx] if idx < len(peeled_iter) else None
        # If present handle subslices ...
        if subslices_for_slice:
            for subslice in subslices_for_slice:
                partitions_state = subslice.get("partitions", [])
                if len(partitions_state) != num_clusters:
                    raise ValueError("Subslice partition state mismatch with processing zones.")
                qubits_by_pz_sub: dict[str, list[int]] = {}
                for pz_idx, partition_state in enumerate(partitions_state):
                    proc = sorted(partition_state.get("processing_zone", []))
                    qubits_by_pz_sub[pz_names[pz_idx]] = proc

                gates_by_idx: dict[int, list[int]] = subslice.get("gates_by_pz", {})  # type: ignore[arg-type]
                gates_by_pz_named: dict[str, list[int]] = {}
                for pz_idx, pz_name in enumerate(pz_names):
                    gates = list(gates_by_idx.get(pz_idx, []))
                    gates_by_pz_named[pz_name] = gates

                _append_slice_plan(qubits_by_pz_sub, gates_by_pz_named)
            continue

        # ... otherwise treat individual slice
        required_qubits = {
            q for gate_id in slice_gate_ids for q in gate_info[gate_id].qubits
        }
        qubits_by_pz: dict[str, list[int]] = {pz: [] for pz in pz_names}
        gates_by_pz: dict[str, list[int]] = {pz: [] for pz in pz_names}
        for qubit in sorted(required_qubits):
            cluster = qubit_assignment[qubit]
            if cluster < 0 or cluster >= num_clusters:
                msg = f"Qubit {qubit} from slice assigned to invalid cluster {cluster}."
                raise ValueError(msg)
            qubits_by_pz[pz_names[cluster]].append(qubit)
        for gate_id in slice_gate_ids:
            qubits = gate_info[gate_id].qubits
            if not qubits:
                continue
            cluster = qubit_assignment[qubits[0]]
            if cluster < 0 or cluster >= num_clusters:
                msg = f"Gate {gate_id} references unassigned qubit {qubits[0]}."
                raise ValueError(msg)
            if any(qubit_assignment[q] != cluster for q in qubits[1:]):
                msg = f"Gate {gate_id} spans multiple clusters after refinement."
                raise ValueError(msg)
            pz_name = pz_names[cluster]
            gates_by_pz[pz_name].append(gate_id)

        _append_slice_plan(qubits_by_pz, gates_by_pz)

    return per_slice_qubit_assignments, slice_plans, gate_partition_by_pz, gate_assignment


def _plot_interaction_graph(
    nodes: Sequence[int] | Sequence[Supernode],
    required_edges: dict[tuple[int, int], float],
    auxiliary_edges: dict[tuple[int, int], float],
    out_path: Path,
    *,
    node_label: str,
    highlighted_nodes: set[int] | None = None,
    node_weights: dict[int, float] | None = None,
    assignment: list[int] | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:  # pragma: no cover - optional diagnostics
        return

   

    G = nx.Graph()
    highlighted_nodes = highlighted_nodes or set()
    node_weights = node_weights or {}

    for node in nodes:
        if isinstance(node, Supernode):
            if len(node.qubits) == 1:
                label = f"q{node.qubits[0]}"
            else:
                label = f"{node_label}{node.id} ({len(node.qubits)})"
            highlight = any(q in highlighted_nodes for q in node.qubits)
            G.add_node(
                node.id,
                qubits=node.qubits,
                label=label,
                highlighted=highlight,
            )
        else:
            G.add_node(
                node,
                qubits=(node,),
                label=f"{node_label}{node}",
                highlighted=node in highlighted_nodes,
            )

    for (u, v), weight in auxiliary_edges.items():
        if weight <= 0:
            continue
        G.add_edge(u, v, weight=weight, edge_type="aux")
    for (u, v), weight in required_edges.items():
        if G.has_edge(u, v):
            G[u][v]["weight"] += weight
            G[u][v]["edge_type"] = "required"
        else:
            G.add_edge(u, v, weight=weight, edge_type="required")

    pos = nx.circular_layout(G)
    plt.figure(figsize=(8, 6))

     # Plot polygons for the partitions in the background:
    partitions = set(assignment or [])
    partition_colors = plt.cm.Set3(np.linspace(0, 1, len(partitions)))
    
    for partition_idx, partition in enumerate(partitions):
        qubits_in_partition = [node for node, assign in enumerate(assignment) if assign == partition]

        if len(qubits_in_partition) <= 0:
            continue

        if len(qubits_in_partition) == 1:
            # Draw a circle around the single qubit
            circle_center = pos[qubits_in_partition[0]]
            circle = patches.Circle(
                circle_center,
                radius=0.15,
                alpha=0.2,
                facecolor=partition_colors[partition_idx],
                edgecolor=partition_colors[partition_idx],
            )

            plt.gca().add_patch(circle)
            continue
        
        # Get positions of qubits in this partition
        partition_pos = np.array([pos[qubit] for qubit in qubits_in_partition])
        
        # Calculate convex hull to draw a shape around the partition
        if len(qubits_in_partition) >= 3:
            try:
                hull = ConvexHull(partition_pos)
                hull_points = partition_pos[hull.vertices]
                # Add some padding around the convex hull
                center = np.mean(partition_pos, axis=0)
                hull_points = center + 1.2 * (hull_points - center)
                
                polygon = patches.Polygon(
                    hull_points, 
                    closed=True, 
                    alpha=0.2, 
                    facecolor=partition_colors[partition_idx],
                    edgecolor=partition_colors[partition_idx],
                    linewidth=2,
                    linestyle='--'
                )
                plt.gca().add_patch(polygon)
            except:
                # Fallback for degenerate cases
                pass
        elif len(qubits_in_partition) == 2:
            # For 2 qubits, draw a line with some thickness
            p1, p2 = partition_pos
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                    color=partition_colors[partition_idx], 
                    linewidth=8, alpha=0.3, linestyle='--')

    labels = nx.get_node_attributes(G, "label")
    highlighted = [node for node, data in G.nodes(data=True) if data.get("highlighted", False)]
    regular = [node for node in G.nodes if node not in highlighted]
    node_list = list(G.nodes)

    node_weights = node_weights or {}
    colors = []
    for node in node_list:
        data = G.nodes[node]
        qubits = data.get("qubits", (node,))
        colors.append(sum(node_weights.get(q, 0.0) for q in qubits))
    vmax = max(colors) if colors else 0.0
    cmap = None
    try:
        import matplotlib.pyplot as plt  # type: ignore  # noqa: PLC0415
        cmap = plt.cm.YlOrRd
    except Exception:
        pass
    vmin = 0
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=regular,
        node_color=[colors[node_list.index(n)] for n in regular] if regular else "#1f77b4",
        node_size=600,
        edgecolors="gray",
        linewidths=0.8,
        cmap=cmap,
        vmin=vmin if cmap else None,
        vmax=vmax*2 if cmap and vmax and vmax > 0 else None,
    )
    if highlighted:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=highlighted,
            node_color=[colors[node_list.index(n)] for n in highlighted] if highlighted else "#1f77b4",
            cmap=cmap,
            edgecolors = "red",
            linewidths=2,
            node_size=750,
            vmin=0,
            vmax=vmax*2 if vmax > 0 else None,
        )
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

    required_edges_list = [(u, v) for (u, v, d) in G.edges(data=True) if d.get("edge_type") == "required"]
    aux_edges_list = [(u, v) for (u, v, d) in G.edges(data=True) if d.get("edge_type") == "aux"]

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=aux_edges_list,
        edge_color="#7f7f7f",
        style="dashed",
        alpha=0.7,
        width=1.0,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=required_edges_list,
        edge_color="#d62728",
        width=2.0,
    )

    weights = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels={edge: f"{w:.1f}" for edge, w in weights.items()})
        
        

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _print_summary(result: ContractionResult) -> None:
    print(f"Supernodes: {len(result.supernodes)}")
    for node in result.supernodes:
        print(f"  S{node.id}: qubits={node.qubits}, load={node.load}")
    print(f"Required edges: {len(result.required_edges)}")
    for (u, v), weight in sorted(result.required_edges.items()):
        print(f"  q{u} -- q{v}: weight={weight:.2f}")
    if result.assignment is not None:
        print("Greedy assignment:", result.assignment)
        if result.cluster_loads:
            print("Cluster loads:", result.cluster_loads)

__all__ = ["fgp_tabu_global"]


def main() -> None:    
    from pathlib import Path
    import argparse
    parser = argparse.ArgumentParser(description="Partition preview for the global tabu optimizer.")
    parser.add_argument("qasm", type=Path, help="Path to the QASM file.")
    parser.add_argument(
        "--num-pzs",
        type=int,
        default=None,
        help="Number of processing zones to seed an assignment (optional).",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=None,
        help="Uniform capacity per processing zone for peeling (optional).",
    )
    parser.add_argument(
        "--capacity-weight",
        type=float,
        default=1.0,
        help="Weight for penalizing capacity overflow in the global objective.",
    )
    parser.add_argument(
        "--distance-weight",
        type=float,
        default=1.0,
        help="Weight for penalizing shuttling distance between slices.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable diagnostic plotting.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/fgp_tabu"),
        help="Directory that receives the generated plots.",
    )
    args = parser.parse_args()

    sequence, gate_info = _load_gate_metadata(args.qasm)
    num_qubits = _infer_num_qubits(gate_info)

    partition_output = _run_fgp_tabu(
        sequence,
        gate_info,
        num_qubits=num_qubits,
        num_pzs=args.num_pzs,
        capacity=args.capacity,
        capacity_weight=args.capacity_weight,
        distance_weight=args.distance_weight,
        pz_names=None,
        enable_plots=not args.no_plot,
        output_dir=args.output_dir,
        pz_positions=None,
        pz_distance_map=None,
    )

    for result in partition_output["partition_results"]:
        _print_summary(result)


if __name__ == "__main__":
    main()
