from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
from statistics import variance
from typing import Iterable, Sequence
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.patches as patches

from .compilation import create_initial_sequence
from .fgp_roee import _build_time_slices
from .types import GateInfo

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


@dataclass(slots=True)
class Partition:
    processing_zone: set[int]
    memory_zone: set[int]
    tbd: set[int]


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
    print("REQUIRED EDGES:", result.required_edges)
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
            print(f">{pz_idx}>> initial :", partition)
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
                print("cap:", remaining_cap, "sn_id:", sn_id, "qs:", qs)
                if remaining_cap < len(qs): # Only add a supernode if it fits entirely!
                    break
                take = min(len(qs), remaining_cap)
                selected = set(list(qs)[:take])
                partition.processing_zone.update(selected)
                partition.tbd.difference_update(selected)
                remaining_cap -= take
                progress = progress or bool(selected)

            print(f">{pz_idx}>> required gates:", required_gates)
            print(f">{pz_idx}>> progress?", progress)
            print(f">{pz_idx}>> partition state:", partition)

        performed: set[int] = set()
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
                "gates": sorted(performed),
            }
        )

        required_gates.difference_update(performed)
        for p in partitions:
            p.memory_zone.update(p.processing_zone)
            p.processing_zone.clear()

        if not progress and not performed and required_gates:
            raise RuntimeError("Peeling stalled: no progress but required gates remain.")
        

    if len(subslices) > 1:
        print(f"\n>>> DEBUG: Slice {slice_gate_ids[0]}-{slice_gate_ids[-1]} produced {len(subslices)} subslices")
        print(f">>> num_pzs={num_pzs}, capacities={capacities}")
        print(f">>> Initial assignment: {result.assignment}")
        print(f">>> Cluster loads: {result.cluster_loads}")
        print(f">>> Supernodes:")
        for sn in result.supernodes:
            print(f"    S{sn.id}: qubits={sn.qubits}, cluster={result.assignment[sn.id] if result.assignment else None}")
        print(f">>> Required qubits: {required_qubits}")
        print(f">>> Subslices breakdown:")
        for i, subslice in enumerate(subslices):
            print(f"  Subslice {i}:")
            for pz_idx, p in enumerate(subslice["partitions"]):
                print(f"    PZ{pz_idx}: processing={p['processing_zone']}, memory={p['memory_zone']}, tbd={p['tbd']}")
            print(f"    Gates performed: {subslice['gates']}")

    return subslices


def _compute_cost(
    result: ContractionResult,
    assignment: list[int],
    num_pzs: int,
    *,
    lookahead_weight_factor: float = 1.0, # relative weight of lookahead vs current slice
    balance_penalty: float = 0.5,
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

    return lookahead_cut_penalty_norm * lookahead_weight_factor + balance_penalty * (var_required + var_lookahead *lookahead_weight_factor)


def tabu_optimize_partition(
    result: ContractionResult,
    num_pzs: int,
    *,
    max_iterations: int = 50,
    tabu_list_length: int = 20,
    lookahead_weight_factor: float = 1.0,
    balance_penalty: float = 0.1,
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
    )
    tabu_list: list[tuple[int, int]] = []

    for _ in range(max_iterations):
        current_cost = _compute_cost(
            result,
            assignment,
            num_pzs,
            lookahead_weight_factor=lookahead_weight_factor,
            balance_penalty=balance_penalty,
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
    if num_pzs is not None:
        # TODO: use previous partitioning as seed, to skip greedy partitioning?
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

    # tabu search to optimize partitions
    print("result.assignment before tabu:", result.assignment)

    if num_pzs is not None and assignment is not None:
        result.assignment, result.cluster_loads = tabu_optimize_partition(
            result,
            num_pzs,
            balance_penalty=balance_penalty,
        )

    print("result.assignment after tabu:", result.assignment)

    return result


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

    
def fgp_tabu(sequence, gate_info, num_qubits, args) -> None:

    # Slice circuit into moments (max 1 gate per qubit)
    # -> Gates within a slice commute, slices don't commute amongst each other
    time_slices = _build_time_slices(sequence, gate_info, num_qubits)

    partitioning_results = []
    # Graph-based partitioning of circuit slices using exponentially decaying lookahead 
    for idx, current_slice in enumerate(time_slices):
        if args.lookahead_slices == math.inf:
            future_slice_window = time_slices[idx + 1 :]
        else:   
            future_slice_window = time_slices[idx + 1 : idx + 1 + args.lookahead_slices]

        result = partition_slice(
            gate_info,
            current_slice,
            future_slice_window,
            num_qubits=num_qubits,
            sigma_edges=args.sigma,
            sigma_single=args.sigma_single,
            num_pzs=args.num_pzs,
            balance_penalty=args.balance_penalty,
        )
        partitioning_results.append(result)
    for idx, result in enumerate(partitioning_results):
        print(f"\n=== Slice {idx} ===")
        _print_summary(result)

    # Process each slice for plotting and peeling
    fgp_result = []
    for idx, current_slice in enumerate(time_slices):
        result = partitioning_results[idx]
        if not args.no_plot:
            plot_partition_outputs(num_qubits, result, gate_info, current_slice, future_slice_window, args.output_dir)
        if args.capacity and args.num_pzs:
            capacities = [args.capacity] * args.num_pzs
            subslices = peel_slice(result, gate_info, current_slice, capacities)
            fgp_result.append(subslices)

    return fgp_result    

    


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
        "--no-plot",
        action="store_true",
        help="Disable diagnostic plotting.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/fgp_new"),
        help="Directory that receives the generated plots.",
    )
    args = parser.parse_args()

    sequence, gate_info = _load_gate_metadata(args.qasm)
    num_qubits = _infer_num_qubits(gate_info)
    
    fgp_tabu(sequence, gate_info, num_qubits, args)



if __name__ == "__main__":
    main()
