from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

import math
import os
from pathlib import Path

if TYPE_CHECKING:
    from .graph import Graph
    from .types import GateInfo

DEBUG_ENABLED=1

if DEBUG_ENABLED:
    import matplotlib.pyplot as plt
    import networkx as nx
from collections import defaultdict
import numpy as np
from scipy.spatial import ConvexHull

DEBUG_PLOT_AVAILABLE = DEBUG_ENABLED and "plt" in globals() and "nx" in globals()
DEBUG_PLOT_DIR = Path("runs/fgp_debug") if DEBUG_ENABLED else None


@dataclass(slots=True)
class FGPResult:
    """Container with detailed output of the FGP-rOEE-inspired partitioner."""

    assignments: list[list[int]]
    """Qubit-to-cluster assignments per time slice."""
    gate_partition_by_pz: dict[str, list[int]]
    """Ordered mapping of processing-zone names to gate ids."""
    gate_assignment: dict[int, str]
    """Direct gate id to processing-zone name mapping."""
    moves_between_slices: list[list[tuple[int, int, int]]]
    """Non-local moves required between successive slices (qubit, src, dst)."""


def _debug(*args: object) -> None:
    if DEBUG_ENABLED:
        print("[FGP]", *args)


def compute_gate_partition(
    graph: "Graph",
    *,
    num_clusters: int | None = None,
    capacity: int | None = None,
    sigma: float = 1.0,
    large_weight: float = 1e6,
    max_iterations_multiplier: int = 10,
    aggregate_slices: int = 1,
) -> FGPResult:
    """Derive a gate partition using a simplified FGP-rOEE heuristic.

    Parameters
    ----------
    graph:
        Current graph instance containing the gate sequence and gate metadata.
    num_clusters:
        Number of processing zones to partition into. Defaults to ``len(graph.pzs)``.
    capacity:
        Maximum number of qubits that may reside in a single processing zone.
        Defaults to ``ceil(num_qubits / num_clusters)``.
    sigma:
        Decay constant used in the lookahead weighting (``2 ** (-d / sigma)``).
    large_weight:
        Weight assigned to current-slice edges to force locality.
    max_iterations_multiplier:
        Bounds on exchange attempts during the relaxed OEE phase.

    Returns
    -------
    FGPResult
        Result object containing per-slice assignments and the resulting
        gate-to-processing-zone mapping.
    """

    if not graph.sequence:
        gate_partition_by_pz = {pz.name: [] for pz in graph.pzs}
        return FGPResult([], gate_partition_by_pz, {}, [])

    _debug("Starting FGP partitioning")

    # Setup: determine clusters and capacity constraints
    num_clusters = num_clusters or len(graph.pzs)
    if num_clusters <= 0:
        msg = "Number of processing zones must be positive."
        raise ValueError(msg)

    gate_info = graph.gate_info
    all_gate_ids = list(graph.sequence)
    two_qubit_gate_ids = [gate_id for gate_id in all_gate_ids if len(gate_info[gate_id].qubits) >= 2]

    if not two_qubit_gate_ids:
        gate_partition_by_pz = {pz.name: [] for pz in graph.pzs}
        return FGPResult([], gate_partition_by_pz, {}, [])

    num_qubits = _infer_num_qubits(gate_info)
    capacity = max(capacity or math.ceil(num_qubits / num_clusters), 1)
    _debug(f"num_qubits={num_qubits}, num_clusters={num_clusters}, capacity={capacity}")

    # Phase 1: Group gates into time slices respecting original order
    time_slices = _build_time_slices(all_gate_ids, gate_info, len(graph.state))
    time_slices = _aggregate_time_slices(time_slices, aggregate_slices)
    _debug(f"Built {len(time_slices)} time slices:")
    for i, slice_gates in enumerate(time_slices):
        gate_details = [f"gate={gate}: {gate_info[gate].qubits}" for gate in slice_gates]
        _debug(f"  Slice {i}: {gate_details}")
    if not time_slices:
        gate_partition_by_pz = {pz.name: [] for pz in graph.pzs}
        return FGPResult([], gate_partition_by_pz, {}, [])

    # Phase 2: Create initial partition based on total qubit interaction frequency
    total_weights = _build_total_interaction_graph(two_qubit_gate_ids, gate_info)
    initial_assignment = _initial_partition(num_qubits, num_clusters, capacity, total_weights)
    _debug("Initial assignment:", initial_assignment)

    # Phase 3: Precompute weighted interactions for each slice (current + future lookahead)
    lookahead_cache = _precompute_future_interactions(time_slices, gate_info, sigma, large_weight)

    # Phase 4: Iteratively refine partition for each time slice using relaxed OEE
    assignments: list[list[int]] = []
    previous_assignment = initial_assignment

    for slice_index, slice_gate_ids in enumerate(time_slices):
        weights, required_edges = lookahead_cache[slice_index]
        _debug(f"Slice {slice_index}: gates={slice_gate_ids}, required_edges={required_edges}")
        
        # Build adjacency representation for this slice's weighted graph
        neighbor_map = _build_neighbor_map(weights, num_qubits)
        
        

        # Apply relaxed OEE to satisfy current slice constraints while minimizing cut weight
        assignment = _roee_partition(
            previous_assignment,
            neighbor_map,
            required_edges,
            num_clusters,
            capacity,
            max_iterations_multiplier,
        )
        if DEBUG_PLOT_AVAILABLE:
            _plot_slice_graph(slice_index, weights, required_edges, num_qubits, assignment)
        assignments.append(assignment.copy())
        _debug(f"Assignment for slice {slice_index}: {assignment}")
        previous_assignment = assignment

    # Phase 5: Compute required qubit movements between consecutive slices
    moves = _compute_moves(assignments)
    _debug("Computed moves between slices:", moves)
    
    # Phase 6: Map final assignments to processing zone names
    pz_names = [pz.name for pz in graph.pzs]
    if num_clusters > len(pz_names):
        msg = "The number of clusters exceeds the number of processing zones defined on the graph."
        raise ValueError(msg)

    gate_partition_by_pz = {pz_name: [] for pz_name in pz_names}
    gate_assignment: dict[int, str] = {}

    for slice_assignment, slice_gate_ids in zip(assignments, time_slices):
        for gate_id in slice_gate_ids:
            qubits = gate_info[gate_id].qubits
            cluster = slice_assignment[qubits[0]]  # All qubits in same gate must be in same cluster
            pz_name = pz_names[cluster]
            gate_partition_by_pz[pz_name].append(gate_id)
            gate_assignment[gate_id] = pz_name

    return FGPResult(assignments, gate_partition_by_pz, gate_assignment, moves)


# ---------------------------------------------------------------------------
# Helper functions


def _infer_num_qubits(gate_info: dict[int, "GateInfo"]) -> int:
    """Find the total number of qubits by looking at the highest qubit index."""
    max_index = -1
    for info in gate_info.values():
        if info.qubits:
            max_index = max(max_index, max(info.qubits))
    if max_index < 0:
        msg = "Unable to infer qubit count from gate metadata."
        raise ValueError(msg)
    return max_index + 1


def _build_time_slices(gate_ids: list[int], gate_info: dict[int, "GateInfo"], num_qubits: int) -> list[list[int]]:
    """Group gates into ordered time slices while respecting original gate order."""
    gate_ids = sorted(gate_ids) # Ensure time-consistent ordering for gate_ids shuffled by DAG
    processed: set[int] = set()
    slices: list[list[int]] = []
    total_gates = len(gate_ids)

    while len(processed) < total_gates:
        blocked_qubits: set[int] = set()
        current_slice: list[int] = []
        progress = False

        for gate_id in gate_ids:
            if len(blocked_qubits) == num_qubits:
                break
            if gate_id in processed:
                continue
            qubits = gate_info[gate_id].qubits
            if not qubits:
                processed.add(gate_id)
                progress = True
                continue

            if any(qubit in blocked_qubits for qubit in qubits):
                #print("  skipping gate", gate_id, "on qubits", qubits)
                blocked_qubits.update(qubits)
                continue

            blocked_qubits.update(qubits)
            processed.add(gate_id)
            progress = True

            if len(qubits) >= 1:
                current_slice.append(gate_id)

            #print("current slice:", [gate_info[gate_id].qubits for gate_id in current_slice])
            #print("blocked qubits:", blocked_qubits)


        if current_slice:
            slices.append(current_slice)

        if not progress:
            # Deadlock fallback: place the next unprocessed 2-qubit gate in its own slice
            for gate_id in gate_ids:
                if gate_id in processed:
                    continue
                qubits = gate_info[gate_id].qubits
                processed.add(gate_id)
                if len(qubits) >= 2:
                    slices.append([gate_id])
                break
            else:
                break

    return slices

def _aggregate_time_slices(
    time_slices: list[list[int]],
    aggregate_slices: int
) -> list[list[int]]:
    """Aggregate time slices by combining adjacent slices."""
    if aggregate_slices <= 1:
        return time_slices

    aggregated: list[list[int]] = []
    current_slice: list[int] = []

    for i, slice in enumerate(time_slices):
        current_slice.extend(slice)
        if (i + 1) % aggregate_slices == 0:
            aggregated.append(current_slice)
            current_slice = []

    if current_slice:
        aggregated.append(current_slice)

    return aggregated

def _build_total_interaction_graph(
    gate_ids: list[int],
    gate_info: dict[int, "GateInfo"],
) -> dict[tuple[int, int], float]:
    """Count total interactions between each pair of qubits across all gates."""
    weights: dict[tuple[int, int], float] = defaultdict(float)
    for gate_id in gate_ids:
        qubits = gate_info[gate_id].qubits
        if len(qubits) != 2:
            continue
        pair = tuple(sorted(qubits))
        weights[pair] += 1.0
    return weights


def _initial_partition(
    num_qubits: int,
    num_clusters: int,
    capacity: int,
    weights: dict[tuple[int, int], float],
) -> list[int]:
    """Create initial partition by greedily placing highly-connected qubits with their neighbors."""
    # Calculate total interaction weight for each qubit
    neighbor_weights = defaultdict(float)
    for (u, v), weight in weights.items():
        neighbor_weights[u] += weight
        neighbor_weights[v] += weight

    # Process qubits in order of decreasing connectivity
    sorted_qubits = sorted(range(num_qubits), key=lambda q: neighbor_weights[q], reverse=True)
    clusters: list[set[int]] = [set() for _ in range(num_clusters)]
    assignment = [-1] * num_qubits

    for qubit in sorted_qubits:
        # Find cluster that maximizes internal edge weight (greedy placement)
        best_cluster = None
        best_gain = -math.inf
        for cluster_idx in range(num_clusters):
            if len(clusters[cluster_idx]) >= capacity:
                continue
            gain = sum(
                weights.get(tuple(sorted((qubit, other))), 0.0)
                for other in clusters[cluster_idx]
            )
            if gain > best_gain:
                best_gain = gain
                best_cluster = cluster_idx
        
        # Fallback to least loaded cluster if no beneficial placement found
        if best_cluster is None:
            best_cluster = min(range(num_clusters), key=lambda idx: len(clusters[idx]))
        
        assignment[qubit] = best_cluster
        clusters[best_cluster].add(qubit)

    return assignment


def _precompute_future_interactions(
    time_slices: list[list[int]],
    gate_info: dict[int, "GateInfo"],
    sigma: float,
    large_weight: float,
) -> list[tuple[dict[tuple[int, int], float], set[tuple[int, int]]]]:
    """Precompute weighted edge graphs combining current slice (high weight) + future slices (decaying weight)."""
    cache: list[tuple[dict[tuple[int, int], float], set[tuple[int, int]]]] = []
    total_slices = len(time_slices)

    for current_index in range(total_slices):
        weights: dict[tuple[int, int], float] = defaultdict(float)
        required_edges: set[tuple[int, int]] = set()

        # Current slice edges get large weight to enforce locality
        for gate_id in time_slices[current_index]:
            qubits = gate_info[gate_id].qubits
            if len(qubits) != 2:
                continue
            pair = tuple(sorted(qubits))
            weights[pair] += large_weight
            required_edges.add(pair)

        # Future slice edges get exponentially decaying weight for lookahead
        if sigma != 0:
            for future_index in range(current_index + 1, total_slices):
                decay = 2.0 ** (-(future_index - current_index) / sigma)
                for gate_id in time_slices[future_index]:
                    qubits = gate_info[gate_id].qubits
                    if len(qubits) != 2:
                        continue
                    pair = tuple(sorted(qubits))
                    weights[pair] += decay

        cache.append((weights, required_edges))

    return cache


def _build_neighbor_map(
    weights: dict[tuple[int, int], float],
    num_qubits: int,
) -> list[list[tuple[int, float]]]:
    """Convert edge weights to adjacency list representation for efficient neighbor lookup."""
    neighbor_map: list[list[tuple[int, float]]] = [[] for _ in range(num_qubits)]
    for (u, v), weight in weights.items():
        neighbor_map[u].append((v, weight))
        neighbor_map[v].append((u, weight))
    return neighbor_map


def _roee_partition(
    previous_assignment: list[int],
    neighbor_map: list[list[tuple[int, float]]],
    required_edges: set[tuple[int, int]],
    num_clusters: int,
    capacity: int,
    max_iterations_multiplier: int,
) -> list[int]:
    """Apply relaxed OEE: iteratively fix constraint violations via moves/swaps while respecting capacity."""
    _debug("Running rOEE partitioner")
    assignment = previous_assignment.copy()
    
    # Maintain cluster membership for efficient capacity checking
    clusters: list[set[int]] = [set() for _ in range(num_clusters)]
    for qubit, cluster_idx in enumerate(assignment):
        clusters[cluster_idx].add(qubit)

    if not required_edges:
        _debug("No required edges in this slice; returning previous assignment")
        return assignment

    max_iterations = max(max_iterations_multiplier * max(1, len(required_edges)), 10)
    _debug(f"Max iterations: {max_iterations}")
    
    # Iteratively fix violations by moving/swapping qubits
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        
        # Find edges where qubits are in different clusters (constraint violations)
        violating_edges = [
            edge for edge in required_edges if assignment[edge[0]] != assignment[edge[1]]
        ]
        if not violating_edges:
            _debug(f"All constraints satisfied after {iteration-1} iterations")
            break

        _debug(f"Iteration {iteration}: violating edges {violating_edges}")
        change_applied = False
        
        # For each violating edge, try to fix via move or swap
        for u, v in violating_edges:
            best_operation: tuple[str, tuple[int, ...], float] | None = None

            # Option 1: Move u to v's cluster (if capacity allows)
            target_cluster = assignment[v]
            if len(clusters[target_cluster]) < capacity:
                delta = _move_delta(u, target_cluster, assignment, neighbor_map)
                best_operation = ("move", (u, target_cluster), delta)

            # Option 2: Move v to u's cluster (if capacity allows)
            target_cluster_alt = assignment[u]
            if len(clusters[target_cluster_alt]) < capacity:
                delta = _move_delta(v, target_cluster_alt, assignment, neighbor_map)
                if best_operation is None or delta < best_operation[2]:
                    best_operation = ("move", (v, target_cluster_alt), delta)

            # Option 3: Swap u with someone in v's cluster (if moves not possible)
            if best_operation is None:
                swap_candidate = _best_swap_candidate(
                    u,
                    v,
                    assignment,
                    clusters,
                    capacity,
                    neighbor_map,
                )
                if swap_candidate is not None:
                    best_operation = swap_candidate

            if best_operation is None:
                _debug(
                    "Failed to find move/swap for edge",
                    (u, v),
                    "current assignment",
                    assignment,
                    "cluster sizes",
                    [len(c) for c in clusters],
                )
                msg = (
                    "Unable to satisfy current slice constraints within capacity limits. "
                    "Consider relaxing capacity or revising the lookahead parameters."
                )
                raise ValueError(msg)

            # Apply the best operation found
            op_type, data, _ = best_operation
            _debug("Applying operation:", op_type, data)
            if op_type == "move":
                qubit, target = data
                _apply_move(qubit, target, assignment, clusters)
            elif op_type == "swap":
                qubit_a, qubit_b = data
                _apply_swap(qubit_a, qubit_b, assignment, clusters)
            change_applied = True
            break

        if not change_applied:
            break

    # Verify all constraints are satisfied
    unsatisfied = [
        edge for edge in required_edges if assignment[edge[0]] != assignment[edge[1]]
    ]
    if unsatisfied:
        _debug("Final assignment violates edges", unsatisfied, "assignment", assignment)
        msg = (
            "Relaxed partitioner terminated without satisfying all slice constraints. "
            "Try adjusting capacity or decay parameters."
        )
        raise ValueError(msg)

    return assignment


def _move_delta(
    qubit: int,
    target_cluster: int,
    assignment: list[int],
    neighbor_map: list[list[tuple[int, float]]],
) -> float:
    """Calculate change in cut weight if qubit is moved to target cluster."""
    current_cluster = assignment[qubit]
    if current_cluster == target_cluster:
        return 0.0

    delta = 0.0
    for other, weight in neighbor_map[qubit]:
        if assignment[other] == current_cluster:
            delta += weight  # Lose internal edge (bad)
        elif assignment[other] == target_cluster:
            delta -= weight  # Gain internal edge (good)
    return delta


def _best_swap_candidate(
    u: int,
    v: int,
    assignment: list[int],
    clusters: list[set[int]],
    capacity: int,
    neighbor_map: list[list[tuple[int, float]]],
) -> tuple[str, tuple[int, int], float] | None:
    """Find the best qubit in v's cluster to swap with u, or suggest a move if space allows."""
    target_cluster = assignment[v]
    current_cluster = assignment[u]
    if current_cluster == target_cluster:
        return None

    # Try swapping u with each qubit in v's cluster
    best_swap: tuple[int, int, float] | None = None
    for candidate in clusters[target_cluster]:
        if candidate == v:
            continue
        delta = _swap_delta(u, candidate, assignment, neighbor_map)
        if best_swap is None or delta < best_swap[2]:
            best_swap = (u, candidate, delta)

    # Fallback to move if cluster has capacity and no good swap found
    if best_swap is None and len(clusters[target_cluster]) < capacity:
        return ("move", (u, target_cluster), _move_delta(u, target_cluster, assignment, neighbor_map))

    if best_swap is None:
        return None

    return ("swap", (best_swap[0], best_swap[1]), best_swap[2])


def _swap_delta(
    qubit_a: int,
    qubit_b: int,
    assignment: list[int],
    neighbor_map: list[list[tuple[int, float]]],
) -> float:
    """Calculate change in cut weight if two qubits swap clusters."""
    cluster_a = assignment[qubit_a]
    cluster_b = assignment[qubit_b]
    if cluster_a == cluster_b:
        return 0.0

    # Calculate current cut cost for both qubits
    original = _incident_cut_cost(qubit_a, assignment, neighbor_map) + _incident_cut_cost(
        qubit_b, assignment, neighbor_map
    )

    # Temporarily swap and calculate new cut cost
    assignment[qubit_a] = cluster_b
    assignment[qubit_b] = cluster_a
    new_cost = _incident_cut_cost(qubit_a, assignment, neighbor_map) + _incident_cut_cost(
        qubit_b, assignment, neighbor_map
    )

    # Restore original assignment
    assignment[qubit_a] = cluster_a
    assignment[qubit_b] = cluster_b
    return new_cost - original


def _incident_cut_cost(
    qubit: int,
    assignment: list[int],
    neighbor_map: list[list[tuple[int, float]]],
) -> float:
    """Calculate total weight of edges from qubit to qubits in different clusters."""
    cluster = assignment[qubit]
    return sum(weight for other, weight in neighbor_map[qubit] if assignment[other] != cluster)


def _apply_move(
    qubit: int,
    target_cluster: int,
    assignment: list[int],
    clusters: list[set[int]],
) -> None:
    """Move qubit to target cluster, updating both assignment and cluster membership."""
    current_cluster = assignment[qubit]
    if current_cluster == target_cluster:
        return
    clusters[current_cluster].remove(qubit)
    clusters[target_cluster].add(qubit)
    assignment[qubit] = target_cluster


def _apply_swap(
    qubit_a: int,
    qubit_b: int,
    assignment: list[int],
    clusters: list[set[int]],
) -> None:
    """Swap cluster assignments of two qubits, updating both assignment and cluster membership."""
    cluster_a = assignment[qubit_a]
    cluster_b = assignment[qubit_b]
    if cluster_a == cluster_b:
        return
    clusters[cluster_a].remove(qubit_a)
    clusters[cluster_b].add(qubit_a)
    clusters[cluster_b].remove(qubit_b)
    clusters[cluster_a].add(qubit_b)
    assignment[qubit_a] = cluster_b
    assignment[qubit_b] = cluster_a


def _compute_moves(assignments: list[list[int]]) -> list[list[tuple[int, int, int]]]:
    """Identify qubits that change clusters between consecutive time slices."""
    moves: list[list[tuple[int, int, int]]] = []
    for previous, current in _pairwise(assignments):
        slice_moves: list[tuple[int, int, int]] = []
        for qubit, (src, dst) in enumerate(zip(previous, current)):
            if src != dst:
                slice_moves.append((qubit, src, dst))
        moves.append(slice_moves)
    return moves


def _pairwise(assignments: list[list[int]]) -> Iterator[tuple[list[int], list[int]]]:
    """Generate consecutive pairs of assignments for move computation."""
    for idx in range(1, len(assignments)):
        yield assignments[idx - 1], assignments[idx]


def _plot_slice_graph(
    slice_index: int,
    weights: dict[tuple[int, int], float],
    required_edges: set[tuple[int, int]],
    num_qubits: int,
    assignment: list[int]
) -> None:
    if not DEBUG_PLOT_AVAILABLE:
        return
    DEBUG_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    graph = nx.Graph()
    graph.add_nodes_from(range(num_qubits))
    max_weight = max(weights.values(), default=1.0)
    for (u, v), weight in weights.items():
        edge_key = tuple(sorted((u, v)))
        color = "red" if edge_key in required_edges else "gray"
        width = 1.0 + 4.0 * (weight / max_weight if max_weight else 0.0)
        graph.add_edge(u, v, weight=weight, color=color, width=width)
    pos = nx.circular_layout(graph)
    edge_colors = [graph[u][v]["color"] for u, v in graph.edges()]
    edge_widths = [graph[u][v]["width"] for u, v in graph.edges()]
    edge_labels = {edge: f"{graph[edge[0]][edge[1]]['weight']:.2f}" for edge in graph.edges()}
    
    plt.figure(figsize=(8, 6))
    
    # Draw partition shapes first (behind everything)
    import matplotlib.patches as patches
    
    partitions = defaultdict(list)
    for qubit, partition in enumerate(assignment):
        partitions[partition].append(qubit)
    
    # Define distinct colors for partitions
    partition_colors = plt.cm.Set3(np.linspace(0, 1, len(partitions)))
    
    for partition_idx, (partition, qubits) in enumerate(partitions.items()):
        if len(qubits) <= 1:
            continue
        
        # Get positions of qubits in this partition
        partition_pos = np.array([pos[qubit] for qubit in qubits])
        
        # Calculate convex hull to draw a shape around the partition
        if len(qubits) >= 3:
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
        elif len(qubits) == 2:
            # For 2 qubits, draw a line with some thickness
            p1, p2 = partition_pos
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                    color=partition_colors[partition_idx], 
                    linewidth=8, alpha=0.3, linestyle='--')
    
    # Draw the graph on top
    nx.draw(
        graph,
        pos,
        with_labels=True,
        edge_color=edge_colors,
        width=edge_widths,
        node_color="#ccccff",
        node_size=600,
        font_size=10,
    )
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
    
    # Add legend for partitions
    legend_elements = []
    for partition_idx, (partition, qubits) in enumerate(partitions.items()):
        legend_elements.append(patches.Patch(
            color=partition_colors[partition_idx], 
            alpha=0.5,
            label=f'Partition {partition} ({len(qubits)} qubits)'
        ))
    
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.title(f"Slice {slice_index}")
    output_path = DEBUG_PLOT_DIR / f"slice_{slice_index:03d}.png"
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


__all__ = ["FGPResult", "compute_gate_partition"]
