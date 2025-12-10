from __future__ import annotations

import itertools
import math
import os
import shutil
from pathlib import Path
from typing import Iterable, Sequence

import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection

from .fgp_roee import FGPResult, _build_time_slices
from .graph import Graph
from .processing_zone import ProcessingZone
from .types import GateInfo, SlicePlan
from .fgp_tabu import (
    ContractionResult,
    Supernode,
    peel_slice,
    plot_partition_outputs,
    _build_edge_weights,
    _build_decayed_edge_weights,
    _build_unary_weights,
    _contract_supernodes,
    _aggregate_lookahead_edges,
    _seed_assignment_from_previous,
    _greedy_initial_partition,
    _compute_cost,
    _build_qubit_assignment,
    _build_slice_plans_from_results,
    _infer_num_qubits,
    _build_pz_distance_map,
)

DEBUG_FLAG = bool(int(os.getenv("IONSHUTTLER_DEBUG_FGP_KL", "0")))


def fgp_kl(
    graph: Graph,
    *,
    num_pzs: int | None = None,
    capacity: int | None = None,
    sigma: float = 1.0,
    sigma_single: float | None = None,
    balance_penalty: float = 1.0,
    max_iterations: int = 50,  # unused, kept for API symmetry
    tabu_list_length: int = 20,  # unused, kept for API symmetry
    lookahead_weight_factor: float = 1.0,
    lookahead_slices: int | float = math.inf,
    distance_weight_factor: float = 1.0,
    graph_based_distance: bool = False,
) -> FGPResult:
    """FGP variant that refines partitions using Kernighan–Lin instead of Tabu search."""

    if DEBUG_FLAG:
        print("=== FGP Kernighan–Lin Parameters ===")
        print(f"num_ions: {_infer_num_qubits(graph.gate_info)}")
        print(f"num_pzs: {num_pzs}")
        print(f"capacity: {capacity}")
        print(f"sigma: {sigma}")
        print(f"sigma_single: {sigma_single}")
        print(f"balance_penalty: {balance_penalty}")
        print(f"lookahead_weight_factor: {lookahead_weight_factor}")
        print(f"lookahead_slices: {lookahead_slices}")
        print(f"distance_weight_factor: {distance_weight_factor}")
        print(f"graph_based_distance: {graph_based_distance}")
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

    partition_output = _run_fgp_kl(
        graph.sequence,
        gate_info,
        num_qubits=num_qubits,
        num_pzs=num_pzs,
        sigma_edges=sigma,
        sigma_single=sigma_single,
        capacity=capacity,
        pz_names=pz_names,
        balance_penalty=balance_penalty,
        lookahead_weight_factor=lookahead_weight_factor,
        lookahead_slices=lookahead_slices,
        distance_weight_factor=distance_weight_factor,
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
                    
                #for key, value in subslice.items():
                #    printu
                #    pz = value.get("processing_zone", set())
                #    mz = value.get("memory_zone", set())
                #    tbd = value.get("tbd", set())
                #    all_qubits = pz + mz + tbd
                #    all_qubits_partitions.append(all_qubits)
                #print(f"Subslices: {all_qubits_partitions}")
        

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

    moves = _compute_moves(assignments)

    return FGPResult(
        assignments,
        gate_partition_by_pz,
        gate_assignment,
        moves,
        slice_plan,
        time_slices=time_slices,
    )


def partition_slice_kl(
    gate_info: dict[int, GateInfo],
    slice_gate_ids: Sequence[int],
    future_slices: Sequence[Sequence[int]],
    num_qubits: int | None = None,
    *,
    sigma_edges: float = 1.0,
    sigma_single: float | None = None,
    num_pzs: int | None = None,
    balance_penalty: float = 1.0,
    lookahead_weight_factor: float = 1.0,
    previous_qubit_assignment: list[int] | None = None,
    pz_positions: Sequence[ProcessingZone | None] | None = None,
    distance_weight_factor: float = 0.0,
    pz_distance_map: dict[tuple[str, str], float] | None = None,
) -> ContractionResult:
    """Contract the slice graph and refine with Kernighan–Lin."""

    if not slice_gate_ids:
        raise ValueError("Slice must contain at least one gate to contract.")

    sigma_single = sigma_single if sigma_single is not None else sigma_edges

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

    supernodes, qubit_to_supernode = _contract_supernodes(qubits, required_edges)
    aggregated_lookahead_edges = _aggregate_lookahead_edges(lookahead_edges, qubit_to_supernode)

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

    if assignment is not None:
        result.assignment, result.cluster_loads = _kernighan_lin_optimize_partition(
            result,
            num_pzs,
            lookahead_weight_factor=lookahead_weight_factor,
            balance_penalty=balance_penalty,
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


def _kl_balanced_split(
    nodes: Sequence[int],
    node_loads: dict[int, int],
) -> tuple[set[int], set[int]]:
    """Seed KL with a roughly balanced bipartition."""
    sorted_nodes = sorted(nodes, key=lambda n: node_loads.get(n, 1), reverse=True)
    part_a: set[int] = set()
    part_b: set[int] = set()
    load_a = 0
    load_b = 0

    for node in sorted_nodes:
        if len(part_a) < len(part_b):
            target = "a"
        elif len(part_b) < len(part_a):
            target = "b"
        else:
            target = "a" if load_a <= load_b else "b"

        if target == "a":
            part_a.add(node)
            load_a += node_loads.get(node, 1)
        else:
            part_b.add(node)
            load_b += node_loads.get(node, 1)

    if not part_a or not part_b:
        midpoint = max(len(sorted_nodes) // 2, 1)
        part_a = set(sorted_nodes[:midpoint])
        part_b = set(sorted_nodes[midpoint:])

    return part_a, part_b


def _kernighan_lin_optimize_partition(
    result: ContractionResult,
    num_pzs: int,
    *,
    lookahead_weight_factor: float = 1.0,
    balance_penalty: float = 0.1,
    pz_positions: Sequence[ProcessingZone | None] | None = None,
    previous_qubit_assignment: list[int] | None = None,
    distance_weight_factor: float = 0.0,
    pz_distance_map: dict[tuple[str, str], float] | None = None,
) -> tuple[list[int], list[int]]:
    if result.assignment is None:
        raise ValueError("Refinement requires an initial assignment.")

    node_loads = {sn.id: sn.load for sn in result.supernodes}
    graph = nx.Graph()
    for sn in result.supernodes:
        graph.add_node(sn.id, load=sn.load)
    for (u, v), weight in result.lookahead_edges.items():
        graph.add_edge(u, v, weight=weight)

    partitions: list[set[int]] = [set(graph.nodes)]
    while len(partitions) < num_pzs:
        idx_largest = max(range(len(partitions)), key=lambda i: sum(node_loads[n] for n in partitions[i]))
        nodes_to_split = partitions.pop(idx_largest)
        if len(nodes_to_split) <= 1:
            partitions.append(nodes_to_split)
            break

        subgraph = graph.subgraph(nodes_to_split)
        init_a, init_b = _kl_balanced_split(list(nodes_to_split), node_loads)
        partition_arg = (init_a, init_b) if len(nodes_to_split) % 2 == 0 else None
        try:
            part_a, part_b = kernighan_lin_bisection(subgraph, partition=partition_arg, weight="weight", seed=0)
        except Exception:
            part_a, part_b = init_a, init_b

        if not part_a or not part_b:
            nodes_list = list(nodes_to_split)
            pivot = max(len(nodes_list) // 2, 1)
            part_a = set(nodes_list[:pivot])
            part_b = set(nodes_list[pivot:])

        partitions.append(set(part_a))
        partitions.append(set(part_b))

    best_assignment = result.assignment.copy()
    best_cost = _compute_cost(
        result,
        best_assignment,
        num_pzs,
        lookahead_weight_factor=lookahead_weight_factor,
        balance_penalty=balance_penalty,
        pz_positions=pz_positions,
        previous_qubit_assignment=previous_qubit_assignment,
        distance_weight_factor=distance_weight_factor,
        pz_distance_map=pz_distance_map,
    )

    for cluster_perm in itertools.permutations(range(num_pzs), len(partitions)):
        assignment = [-1] * len(result.supernodes)
        for part_idx, cluster in enumerate(cluster_perm):
            for sn in partitions[part_idx]:
                assignment[sn] = cluster

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
        if cost < best_cost:
            best_cost = cost
            best_assignment = assignment

    cluster_loads = [0] * num_pzs
    for sn_id, cluster in enumerate(best_assignment):
        if 0 <= cluster < num_pzs:
            cluster_loads[cluster] += result.supernodes[sn_id].load

    return best_assignment, cluster_loads


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


def _run_fgp_kl(
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
    output_dir: Path | None = Path("outputs/fgp_kl"),
    balance_penalty: float,
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

        result = partition_slice_kl(
            gate_info,
            current_slice,
            future_slice_window,
            num_qubits=num_qubits,
            sigma_edges=sigma_edges,
            sigma_single=sigma_single,
            num_pzs=num_pzs,
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

    peeled_subslices: list[list[dict[str, object]] | None] = [None] * len(time_slices)

    if enable_plots and output_dir is not None:
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
        resolved_pz_names = list(pz_names) if pz_names else [f"pz{i}" for i in range(num_pzs)]
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


__all__ = ["fgp_kl"]
