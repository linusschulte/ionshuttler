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
    lookahead_edges: dict[tuple[int, int], float]
    lookahead_unary: dict[int, float]
    assignment: list[int] | None
    cluster_loads: list[int] | None
    pre_tabu_cost: float | None = None
    post_tabu_cost: float | None = None


@dataclass(slots=True)
class Partition:
    processing_zone: set[int]
    memory_zone: set[int]
    tbd: set[int]



def fgp_tabu(
    graph: Graph,
    *,
    num_pzs: int | None = None,
    capacity: int | None = None,
    sigma: float = 1.0,
    sigma_single: float | None = None,
    balance_penalty: float = 1.0,
    max_iterations: int = 50,
    tabu_list_length: int = 20,
    lookahead_weight_factor: float = 1.0,
    lookahead_slices: int | float = math.inf,
    distance_weight_factor: float = 1.0,
    graph_based_distance: bool = False,
) -> FGPResult:
    """Public entry point mirroring compute_gate_partition but using tabu refinement."""

    if DEBUG_FLAG:
        print("=== FGP Tabu Parameters ===")
        print(f"num_ions: {_infer_num_qubits(graph.gate_info)}")
        print(f"num_pzs: {num_pzs}")
        print(f"capacity: {capacity}")
        print(f"sigma: {sigma}")
        print(f"sigma_single: {sigma_single}")
        print(f"balance_penalty: {balance_penalty}")
        print(f"max_iterations: {max_iterations}")
        print(f"tabu_list_length: {tabu_list_length}")
        print(f"lookahead_weight_factor: {lookahead_weight_factor}")
        print(f"lookahead_slices: {lookahead_slices}")
        print(f"distance_weight_factor: {distance_weight_factor}")
        print(f"graph_based_distance: {graph_based_distance}")
        print()

    if not graph.sequence:
        gate_partition_by_pz = {pz.name: [] for pz in graph.pzs}
        return FGPResult([], gate_partition_by_pz, {}, [], [])

    gate_info = graph.gate_info
    num_pzs = num_pzs or len(graph.pzs)
    if num_pzs <= 0:
        raise ValueError("Number of processing zones must be positive.")


    num_qubits = _infer_num_qubits(gate_info)
    capacity = max(capacity or math.ceil(num_qubits / num_pzs), 1)
    pz_names = [pz.name for pz in graph.pzs]
    pz_positions: Sequence[ProcessingZone | None] = [graph.pzs_name_map.get(name) for name in pz_names]
    pz_distance_map = _build_pz_distance_map(graph, pz_positions, graph_based_distance=graph_based_distance)

    

    partition_output = _run_fgp_tabu(
        graph.sequence,
        gate_info,
        num_qubits=num_qubits,
        num_pzs=num_pzs,
        sigma_edges=sigma,
        sigma_single=sigma_single,
        capacity=capacity,
        pz_names=pz_names,
        max_iterations=max_iterations,
        tabu_list_length=tabu_list_length,
        balance_penalty=balance_penalty,
        lookahead_weight_factor=lookahead_weight_factor,
        lookahead_slices=lookahead_slices,
        distance_weight_factor=distance_weight_factor,
        pz_positions=pz_positions,
        pz_distance_map=pz_distance_map,
    )

    if DEBUG_FLAG:
        print("Overview:")
        for idx, slice in enumerate(partition_output["slice_plan"]):
            print(f"Slice {idx+1}", slice)
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

    moves = _compute_moves(assignments)

    return FGPResult(assignments, gate_partition_by_pz, gate_assignment, moves, slice_plan)


def partition_slice(
    gate_info: dict[int, GateInfo],
    slice_gate_ids: Sequence[int],
    future_slices: Sequence[Sequence[int]],
    num_qubits: int | None = None,
    *,
    sigma_edges: float = 1.0,
    sigma_single: float | None = None,
    num_pzs: int | None = None,
    balance_penalty: float = 1.0,
    max_iterations: int = 50,
    tabu_list_length: int = 20,
    lookahead_weight_factor: float = 1.0,
    previous_qubit_assignment: list[int] | None = None,
    pz_positions: Sequence[ProcessingZone | None] | None = None,
    distance_weight_factor: float = 0.0,
    pz_distance_map: dict[tuple[str, str], float] | None = None,
) -> ContractionResult:
    """Contract the slice graph and produce a greedy initial assignment."""

    if not slice_gate_ids:
        raise ValueError("Slice must contain at least one gate to contract.")

    sigma_single = sigma_single if sigma_single is not None else sigma_edges


    # Build lookahead connectivity graph
    required_edges = _build_edge_weights(slice_gate_ids, gate_info)
    lookahead_edges = _build_decayed_edge_weights(future_slices, gate_info, sigma_edges)
    required_unary = {
        gate_info[gate_id].qubits[0]
        for gate_id in slice_gate_ids
        if len(gate_info[gate_id].qubits) == 1
    }
    unary_weights = _build_unary_weights(future_slices, gate_info, sigma_single)

    if not num_qubits:
        raise ValueError("num_qubits must be provided or inferable from gate_info")
    qubits = list(range(num_qubits))
    

    # perform contraction of qubits sharing multi-qubit gates into supernodes
    supernodes, qubit_to_supernode = _contract_supernodes(qubits, required_edges)
    aggregated_lookahead_edges = _aggregate_lookahead_edges(lookahead_edges, qubit_to_supernode)
    

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
        assignment, cluster_loads = _greedy_initial_partition(
            supernodes,
            aggregated_lookahead_edges,
            unary_weights,
            num_pzs,
            balance_penalty=balance_penalty,
        )

    result = ContractionResult(
        supernodes=supernodes,
        qubit_to_supernode=qubit_to_supernode,
        required_edges=required_edges,
        required_unary=required_unary,
        lookahead_edges=aggregated_lookahead_edges,
        lookahead_unary=unary_weights,
        assignment=assignment,
        cluster_loads=cluster_loads,
    )

    if assignment is not None:
        result.pre_tabu_cost = _compute_cost(
            result,
            assignment,
            num_pzs,
            lookahead_weight_factor=lookahead_weight_factor,
            balance_penalty=balance_penalty,
            pz_positions=pz_positions,
            previous_qubit_assignment=previous_qubit_assignment,
            distance_weight_factor=distance_weight_factor,
            pz_distance_map=pz_distance_map,
        )

    # tabu search to optimized partitioning
    if assignment is not None:
        result.assignment, result.cluster_loads = tabu_optimize_partition(
            result,
            num_pzs,
            lookahead_weight_factor=lookahead_weight_factor,
            balance_penalty=balance_penalty,
            max_iterations=max_iterations,
            tabu_list_length=tabu_list_length,
            pz_positions=pz_positions,
            previous_qubit_assignment=previous_qubit_assignment,
            distance_weight_factor=distance_weight_factor,
            pz_distance_map=pz_distance_map,
        )

    if result.assignment is not None:
        result.post_tabu_cost = _compute_cost(
            result,
            result.assignment,
            num_pzs,
            lookahead_weight_factor=lookahead_weight_factor,
            balance_penalty=balance_penalty,
            pz_positions=pz_positions,
            previous_qubit_assignment=previous_qubit_assignment,
            distance_weight_factor=distance_weight_factor,
            pz_distance_map=pz_distance_map,
        )
        
    return result


def tabu_optimize_partition(
    result: ContractionResult,
    num_pzs: int,
    *,
    max_iterations: int = 50,
    tabu_list_length: int = 20,
    lookahead_weight_factor: float = 1.0,
    balance_penalty: float = 0.1,
    pz_positions: Sequence[ProcessingZone | None] | None = None,
    previous_qubit_assignment: list[int] | None = None,
    distance_weight_factor: float = 0.0,
    pz_distance_map: dict[tuple[str, str], float] | None = None,
) -> tuple[list[int], list[int]]:
    if result.assignment is None:
        raise ValueError("Refinement requires an initial assignment.")
    assignment = result.assignment.copy()
    best_assignment = assignment.copy()
    best_cost = _compute_cost(
        result,
        assignment,
        num_pzs,
        lookahead_weight_factor=lookahead_weight_factor,
        balance_penalty=balance_penalty,
        pz_positions=pz_positions,
        previous_qubit_assignment=previous_qubit_assignment,
        distance_weight_factor=distance_weight_factor,
        pz_distance_map=pz_distance_map,
    )
    tabu_list: list[tuple[int, int]] = []

    for _ in range(max_iterations):
        current_cost = _compute_cost(
            result,
            assignment,
            num_pzs,
            lookahead_weight_factor=lookahead_weight_factor,
            balance_penalty=balance_penalty,
            pz_positions=pz_positions,
            previous_qubit_assignment=previous_qubit_assignment,
            distance_weight_factor=distance_weight_factor,
            pz_distance_map=pz_distance_map,
        )
        best_move = None
        best_move_cost = current_cost

        for sn in range(len(assignment)):
            current_cluster = assignment[sn]
            for target in range(num_pzs):
                if target == current_cluster:
                    continue
                move = (sn, target)
                if move in tabu_list:
                    continue
                assignment[sn] = target
                # TODO: change cost calculation to local gain/loss in order to avoid global cost recomputation for every move
                cost = _compute_cost(
                    result,
                    assignment,
                    num_pzs,
                    lookahead_weight_factor=lookahead_weight_factor,
                    balance_penalty=balance_penalty,
                    pz_positions=pz_positions,
                    previous_qubit_assignment=previous_qubit_assignment,
                    distance_weight_factor=distance_weight_factor,
                    pz_distance_map=pz_distance_map,
                )
                if cost < best_move_cost:
                    best_move_cost = cost
                    best_move = move
                assignment[sn] = current_cluster

        if best_move is None:
            break

        sn, target = best_move
        prev_cluster = assignment[sn]
        assignment[sn] = target
        tabu_list.append((sn, prev_cluster))
        if len(tabu_list) > tabu_list_length:
            tabu_list.pop(0)

        if best_move_cost < best_cost:
            best_cost = best_move_cost
            best_assignment = assignment.copy()

    cluster_loads = [0] * num_pzs
    for sn_id, cluster in enumerate(best_assignment):
        if 0 <= cluster < num_pzs:
            cluster_loads[cluster] += result.supernodes[sn_id].load

    return best_assignment, cluster_loads


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

    # Weight components per supernode
    supernode_unary: dict[int, float] = {
        sn.id: sum(result.lookahead_unary.get(q, 0.0) for q in sn.qubits) for sn in result.supernodes
    }
    supernode_internal: dict[int, float] = {}
    for sn in result.supernodes:
        internal_edge_sum = 0.0
        for (u, v), weight in result.lookahead_edges.items():
            if result.qubit_to_supernode.get(u) == sn.id and result.qubit_to_supernode.get(v) == sn.id:
                internal_edge_sum += weight
        supernode_internal[sn.id] = internal_edge_sum

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
            # Score supernodes: unary + internal + connections to already selected qubits in this PZ
            pz_selected = set(partition.processing_zone)
            edge_map = result.lookahead_edges

            def peel_score(sn_id: int) -> float:
                unary = supernode_unary.get(sn_id, 0.0)
                internal = supernode_internal.get(sn_id, 0.0)
                external = 0.0
                for q in candidates[sn_id]:
                    for sel in pz_selected:
                        external += edge_map.get(tuple(sorted((q, sel))), 0.0)
                return unary + internal + external

            ordered = sorted(candidates.items(), key=lambda kv: peel_score(kv[0]))
            remaining_cap = max(capacities[pz_idx] - len(partition.processing_zone), 0)

            for sn_id, qs in ordered:
                if remaining_cap < len(qs): # Only add a supernode if it fits entirely!
                    break
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
    future_slices: Sequence[Sequence[int]],
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
        result.lookahead_edges,
        before_path,
        node_label="q",
        highlighted_nodes=required_unary_qubits,
        node_weights=result.lookahead_unary,
    )
    after_path = output_dir / f"{prefix}_contracted.png"
    supernode_weights = {
        sn.id: sum(result.lookahead_unary.get(q, 0.0) for q in sn.qubits) for sn in result.supernodes
    }
    _plot_interaction_graph(
        result.supernodes,
        {},
        result.lookahead_edges,
        after_path,
        node_label="S",
        highlighted_nodes=required_qubits,
        node_weights=supernode_weights,
    )

    if result.assignment is not None:
        assign_path = output_dir / f"{prefix}_partition.png"
        
        _plot_interaction_graph(
            result.supernodes,
            {},
            result.lookahead_edges,
            assign_path,
            node_label="S",
            highlighted_nodes=required_qubits,
            node_weights=supernode_weights,
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
    sigma_edges: float,
    sigma_single: float | None,
    
    capacity: int | None,
    pz_names: Sequence[str] | None,
    enable_plots: bool = False,
    output_dir: Path | None = Path("outputs/fgp_tabu"),
    balance_penalty: float,
    max_iterations: int = 50,
    tabu_list_length: int = 20,
    lookahead_weight_factor: float,
    lookahead_slices: int | float,
    distance_weight_factor: float = 0.0,
    pz_positions: Sequence[ProcessingZone | None] | None = None,
    pz_distance_map: dict[tuple[str, str], float] | None = None,

) -> dict[str, object]:
    """Core partition routine shared by CLI and API entry points."""

    time_slices = _build_time_slices(sequence, gate_info, num_qubits)
    partitioning_results: list[ContractionResult] = []
    future_windows: list[Sequence[Sequence[int]]] = []
    prev_qubit_assignment: list[int] | None = None


    for idx, current_slice in enumerate(time_slices):
        if lookahead_slices == math.inf:
            future_slice_window = time_slices[idx + 1 :]
        else:
            future_slice_window = time_slices[idx + 1 : idx + 1 + int(lookahead_slices)]
        future_windows.append(future_slice_window)

        result = partition_slice(
            gate_info,
            current_slice,
            future_slice_window,
            num_qubits=num_qubits,
            sigma_edges=sigma_edges,
            sigma_single=sigma_single,
            num_pzs=num_pzs,
            max_iterations=max_iterations,
            tabu_list_length=tabu_list_length,
            balance_penalty=balance_penalty,
            lookahead_weight_factor=lookahead_weight_factor,
            previous_qubit_assignment=prev_qubit_assignment,
            pz_positions=pz_positions,
            distance_weight_factor=distance_weight_factor,
            pz_distance_map=pz_distance_map,
        )
        partitioning_results.append(result)
        if num_pzs:
            prev_qubit_assignment = _build_qubit_assignment(result, num_qubits, num_pzs)

    pre_tabu_costs = [res.pre_tabu_cost for res in partitioning_results if res.pre_tabu_cost is not None]
    post_tabu_costs = [res.post_tabu_cost for res in partitioning_results if res.post_tabu_cost is not None]
    if DEBUG_FLAG:
        print(f"Total improvement: {sum(pre_tabu_costs) - sum(post_tabu_costs):.4f} ({((sum(pre_tabu_costs) - sum(post_tabu_costs)) / sum(pre_tabu_costs) * 100) if sum(pre_tabu_costs) > 0 else 0.0:.2f}%)\n")


    peeled_subslices: list[list[dict[str, object]] | None] = [None] * len(time_slices)

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
                future_windows[idx],
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


def _build_decayed_edge_weights(
    future_slices: Sequence[Sequence[int]],
    gate_info: dict[int, GateInfo],
    sigma: float,
    threshold: float = 0.05,
) -> dict[tuple[int, int], float]:
    weights: dict[tuple[int, int], float] = {}
    if sigma == 0:
        decay_fn = lambda depth: 1.0
    else:
        decay_fn = lambda depth: math.pow(2.0, -depth / sigma)

    for depth, slice_gate_ids in enumerate(future_slices, start=1):
        decay = decay_fn(depth)
        for gate_id in slice_gate_ids:
            qubits = gate_info[gate_id].qubits
            if len(qubits) != 2:
                continue
            edge = tuple(sorted(qubits))
            weights[edge] = weights.get(edge, 0.0) + decay
    weights = {edge: weight for edge, weight in weights.items() if weight >= threshold}
    return weights


def _build_unary_weights(
    future_slices: Sequence[Sequence[int]],
    gate_info: dict[int, GateInfo],
    sigma: float,
    threshold: float = 0.0,
) -> dict[int, float]:
    """Accumulate per-qubit weights from single-qubit gates with exponential decay (lookahead only)."""
    weights: dict[int, float] = defaultdict(float)

    if sigma == 0:
        decay_fn = lambda depth: 1.0
    else:
        decay_fn = lambda depth: math.pow(2.0, -depth / sigma)

    for depth, slice_gate_ids in enumerate(future_slices, start=1):
        decay = decay_fn(depth)
        for gate_id in slice_gate_ids:
            qubits = gate_info[gate_id].qubits
            if len(qubits) != 1:
                continue
            weights[qubits[0]] += decay

    return {qubit: weight for qubit, weight in weights.items() if weight >= threshold}


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


def _aggregate_lookahead_edges(
    lookahead_edges: dict[tuple[int, int], float],
    qubit_to_supernode: dict[int, int],
) -> dict[tuple[int, int], float]:
    aggregated: dict[tuple[int, int], float] = {}
    for (u, v), weight in lookahead_edges.items():
        if u not in qubit_to_supernode or v not in qubit_to_supernode:
            continue
        super_u = qubit_to_supernode[u]
        super_v = qubit_to_supernode[v]
        if super_u == super_v:
            continue
        edge = tuple(sorted((super_u, super_v)))
        aggregated[edge] = aggregated.get(edge, 0.0) + weight
    return aggregated


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
    lookahead_edges: dict[tuple[int, int], float],
    unary_weights: dict[int, float],
    num_pzs: int,
    *,
    balance_penalty: float = 0.0,
) -> tuple[list[int], list[int]]:
    """Assign supernodes to PZs greedily using edge and unary weights."""
    total_clusters = num_pzs
    assignment = [-1] * len(supernodes)
    cluster_loads = [0] * total_clusters

    # Precompute adjacency by supernode id
    adjacency: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for (u, v), weight in lookahead_edges.items():
        adjacency[u].append((v, weight))
        adjacency[v].append((u, weight))

    # Order supernodes by degree + unary weight
    def node_weight(sn: Supernode) -> float:
        edge_w = sum(w for _, w in adjacency.get(sn.id, []))
        unary_w = sum(unary_weights.get(q, 0.0) for q in sn.qubits)
        return edge_w + unary_w

    for sn in sorted(supernodes, key=node_weight, reverse=True):
        best_cluster = None
        best_score = -math.inf
        for cluster_idx in range(total_clusters):
            # internal connectivity gain
            internal = sum(
                weight
                for neighbor, weight in adjacency.get(sn.id, [])
                if assignment[neighbor] == cluster_idx
            )
            unary_gain = sum(unary_weights.get(q, 0.0) for q in sn.qubits)
            balance = -balance_penalty * cluster_loads[cluster_idx]
            score = internal + unary_gain + balance
            if score > best_score:
                best_score = score
                best_cluster = cluster_idx
        assignment[sn.id] = best_cluster if best_cluster is not None else 0
        cluster_loads[assignment[sn.id]] += sn.load

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


def _compute_cost(
    result: ContractionResult,
    assignment: list[int],
    num_pzs: int,
    *,
    lookahead_weight_factor: float = 1.0, # relative weight of lookahead vs current slice
    balance_penalty: float = 0.5,
    pz_positions: Sequence[ProcessingZone | None] | None = None,
    previous_qubit_assignment: list[int] | None = None,
    distance_weight_factor: float = 0.0,
    pz_distance_map: dict[tuple[str, str], float] | None = None,
) -> float:
    if len(assignment) != len(result.supernodes):
        raise ValueError("Assignment length mismatch.")

    required_cut_penalty = 0.0
    for (u, v), weight in result.required_edges.items():
        su = result.qubit_to_supernode.get(u)
        sv = result.qubit_to_supernode.get(v)
        if su is None or sv is None:
            continue
        if assignment[su] != assignment[sv]:
            required_cut_penalty += weight
    if required_cut_penalty > 0:
        # By construction (supernode contraction) required edges should never be cut.
        # The required_cut_penalty computation only remains as sanity check and can be removed
        # for efficiency once the invariant is trusted.
        raise RuntimeError(f"Required edge cut detected in assignment: {required_cut_penalty}")

    lookahead_cut_penalty = 0.0
    for (su, sv), weight in result.lookahead_edges.items():
        if assignment[su] != assignment[sv]:
            lookahead_cut_penalty += weight

    required_qubits = set(result.required_unary)
    required_qubit_load = [0.0] * num_pzs
    lookahead_qubit_load = [0.0] * num_pzs
    for q in required_qubits:
        su = result.qubit_to_supernode.get(q)
        if su is None:
            continue
        cluster = assignment[su]
        if 0 <= cluster < num_pzs:
            required_qubit_load[cluster] += 1
    for q, w in result.lookahead_unary.items():
        su = result.qubit_to_supernode.get(q)
        if su is None:
            continue
        cluster = assignment[su]
        if 0 <= cluster < num_pzs:
            lookahead_qubit_load[cluster] += w

    def normalized_variance(arr: list[float]) -> float:
        if not arr:
            return 0.0
        mean = sum(arr) / len(arr)
        return sum((x/mean - 1) ** 2 for x in arr) / len(arr)

    var_required = normalized_variance(required_qubit_load) if any(required_qubit_load) else 0.0
    var_lookahead = normalized_variance(lookahead_qubit_load) if any(lookahead_qubit_load) else 0.0

    total_lookahead_weight = sum(result.lookahead_edges.values())
    lookahead_cut_penalty_norm = lookahead_cut_penalty / max(total_lookahead_weight, 1e-9)

    distance_penalty = 0.0
    if distance_weight_factor and previous_qubit_assignment and pz_positions:
        total_distance = 0.0
        counted = 0
        for qubit, supernode in result.qubit_to_supernode.items():
            prev_cluster = previous_qubit_assignment[qubit] if 0 <= qubit < len(previous_qubit_assignment) else -1
            curr_cluster = assignment[supernode]
            if not (0 <= prev_cluster < num_pzs and 0 <= curr_cluster < num_pzs):
                continue
            prev_pz = pz_positions[prev_cluster] if prev_cluster < len(pz_positions) else None
            curr_pz = pz_positions[curr_cluster] if curr_cluster < len(pz_positions) else None
            if prev_pz is None or curr_pz is None:
                continue
            total_distance += get_pz_distance(prev_pz, curr_pz, pz_distance_map)
            counted += 1
        if counted:
            distance_penalty = distance_weight_factor * (total_distance / counted)

    return (
        lookahead_cut_penalty_norm * lookahead_weight_factor
        + balance_penalty * (var_required + var_lookahead * lookahead_weight_factor)
        + distance_penalty
    )



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
    lookahead_edges: dict[tuple[int, int], float],
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

    for (u, v), weight in lookahead_edges.items():
        if weight <= 0:
            continue
        G.add_edge(u, v, weight=weight, edge_type="lookahead")
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
    lookahead_edges_list = [(u, v) for (u, v, d) in G.edges(data=True) if d.get("edge_type") == "lookahead"]

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=lookahead_edges_list,
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
    print(f"Aggregated lookahead edges: {len(result.lookahead_edges)}")
    for (u, v), weight in sorted(result.lookahead_edges.items()):
        print(f"  S{u} -- S{v}: weight={weight:.2f}")
    if result.lookahead_unary:
        print("Unary weights:")
        for qubit, weight in sorted(result.lookahead_unary.items()):
            print(f"  q{qubit}: {weight:.2f}")
    if result.assignment is not None:
        print("Greedy assignment:", result.assignment)
        if result.cluster_loads:
            print("Cluster loads:", result.cluster_loads)

__all__ = ["fgp_tabu"]


def main() -> None:    
    from pathlib import Path
    import argparse
    parser = argparse.ArgumentParser(description="Partition preview for slices with lookahead.")
    parser.add_argument("qasm", type=Path, help="Path to the QASM file.")
    parser.add_argument(
        "--lookahead-slices",
        type=int,
        default=math.inf,
        help="Number of future slices used for the decayed lookahead weights, default all.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Exponential decay constant used for lookahead weighting.",
    )
    parser.add_argument(
        "--sigma-single",
        type=float,
        default=None,
        help="Optional decay constant for single-qubit gates (defaults to sigma).",
    )
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
        "--balance-penalty",
        type=float,
        default=1.0,
        help="Penalty to discourage imbalance when placing supernodes.",
    )
    parser.add_argument(
        "--lookahead-weight-factor",
        type=float,
        default=1.0,
        help="Relative importance of future slices in the tabu cost.",
    )
    parser.add_argument(
        "--distance-weight-factor",
        type=float,
        default=0.0,
        help="Weight for penalizing movement of qubits between processing zones.",
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
        sigma_edges=args.sigma,
        sigma_single=args.sigma_single,
        balance_penalty=args.balance_penalty,
        lookahead_slices=args.lookahead_slices,
        lookahead_weight_factor=args.lookahead_weight_factor,
        capacity=args.capacity,
        pz_names=None,
        enable_plots=not args.no_plot,
        output_dir=args.output_dir,
        distance_weight_factor=args.distance_weight_factor,
        pz_positions=None,
        pz_distance_map=None,
    )

    for result in partition_output["partition_results"]:
        _print_summary(result)


if __name__ == "__main__":
    main()
