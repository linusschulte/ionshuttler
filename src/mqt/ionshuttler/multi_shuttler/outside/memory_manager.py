from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .cycles import check_if_edge_is_filled, find_next_edge, find_ordered_edges
from .graph_utils import get_idx_from_idc

if TYPE_CHECKING:
    from .graph import Graph
    from .processing_zone import ProcessingZone
    from .types import Edge, SlicePlan

DEBUG_FLAG = bool(int(os.getenv("IONSHUTTLER_DEBUG_MEMORY_MANAGER", "0")))

# Limit how many opportunistic moves each PZ can request per timestep
_MAX_MOVES_PER_PZ = 10

def apply_memory_zone_manager(
    graph: Graph,
    all_cycles: dict[int, list[Edge]],
    part_prio_queues: dict[str, list[int]],
    plan_active: bool = False,
    current_plan: SlicePlan | None = None,
    timestep: int | None = None,
) -> dict[int, list[Edge]]:
    """Propose low-impact memory-zone moves for idle ions.

    The returned cycles are merged into the normal scheduling cycles and therefore
    still go through the regular conflict resolution.
    """
    if not graph.enable_memory_zone_manager:
        return {}
    
    extra_cycles: dict[int, list[Edge]] = {}
    busy_ions: set[int] = set(all_cycles.keys()) | set(graph.in_process)
    if plan_active and current_plan is not None:
        for ions in current_plan.qubits_by_pz.values():
            busy_ions.update(ions)

    if DEBUG_FLAG:
        ts_str = f"t={timestep} " if timestep is not None else ""
        print(f"------ {ts_str} ------")

    for pz in graph.pzs:
        prio_queue = part_prio_queues.get(pz.name, [])
        free_edges = _get_free_memory_edges(graph, pz, all_cycles)
        if not free_edges:
            continue

        moves_added = 0
        for ion in prio_queue:
            if DEBUG_FLAG:
                pass#print(f"[mzm] pz={pz.name} considering ion {ion}")
            if moves_added >= _MAX_MOVES_PER_PZ:
                break
            if not _ion_is_idle_candidate(graph, ion, pz, busy_ions):
                continue

            current_edge = graph.state[ion]
            current_dist = len(graph.dist_dict[pz.name][current_edge])

            # Assign the most attractive free edge to the highest priority ion first
            target_edge = None
            for candidate_edge in free_edges:
                if candidate_edge == current_edge:
                    target_edge = candidate_edge
                    break
                target_edge = candidate_edge
                break

            if target_edge is None:
                continue

            target_dist = len(graph.dist_dict[pz.name][target_edge])
            # Only move if target is strictly closer to parking; prevents oscillation between equally good spots
            if target_dist >= current_dist:
                free_edges.remove(target_edge)
                continue

            try:
                next_edge = find_next_edge(
                    graph,
                    current_edge,
                    target_edge,
                    exclude_exit=True,
                    exclude_first_entry_connection=True,
                )
            except Exception:
                continue

            next_idx = get_idx_from_idc(graph.idc_dict, next_edge)
            if not _edge_is_memory_edge(graph, pz, next_idx):
                if DEBUG_FLAG:
                    print(f"[mzm] pz={pz.name} considered ion {ion} but next edge {next_edge} is not memory edge")
                continue
            if check_if_edge_is_filled(graph, next_edge):
                if DEBUG_FLAG:
                    print(f"[mzm] pz={pz.name} considered ion {ion} but next edge {next_edge} is filled")
                continue
            if _edge_used_in_cycles(graph, next_idx, all_cycles) or _edge_used_in_cycles(graph, next_idx, extra_cycles):
                if DEBUG_FLAG:
                    print(f"[mzm] pz={pz.name} considered ion {ion} but next edge {next_edge} is used in another cycle")
                continue

            current_edge, next_edge = find_ordered_edges(graph, current_edge, next_edge)
            extra_cycles[ion] = [current_edge, next_edge]
            free_edges = [edge for edge in free_edges if edge != target_edge and get_idx_from_idc(graph.idc_dict, edge) != next_idx]

            moves_added += 1
            if DEBUG_FLAG:
                print(f"[mzm] pz={pz.name} added move {ion}: {current_edge} -> {next_edge}")

    return extra_cycles


def _edge_is_memory_edge(graph: Graph, pz: ProcessingZone, edge_idx: int) -> bool:
    """Return True if an edge index is part of the memory region (not path/parking) for this PZ."""
    if edge_idx in pz.path_to_pz_idxs or edge_idx in pz.path_from_pz_idxs:
        return False
    if edge_idx == get_idx_from_idc(graph.idc_dict, pz.parking_edge):
        return False
    owner = graph.edge_to_pz_map.get(edge_idx)
    if owner is not None and owner is not pz:
        return False
    return True


def _edge_used_in_cycles(graph: Graph, edge_idx: int, all_cycles: dict[int, list[Edge]]) -> bool:
    for path in all_cycles.values():
        for edge in path:
            if get_idx_from_idc(graph.idc_dict, edge) == edge_idx:
                return True
    return False


def _get_free_memory_edges(graph: Graph, pz: ProcessingZone, all_cycles: dict[int, list[Edge]]) -> list[Edge]:
    occupied_edge_idxs = {get_idx_from_idc(graph.idc_dict, edge) for edge in graph.state.values()}
    free_edges: list[Edge] = []
    for edge_idc in graph.edges():
        edge_idx = get_idx_from_idc(graph.idc_dict, edge_idc)
        if not _edge_is_memory_edge(graph, pz, edge_idx):
            continue
        if edge_idx in occupied_edge_idxs:
            continue
        if _edge_used_in_cycles(graph, edge_idx, all_cycles):
            continue
        free_edges.append(edge_idc)
    # Prefer edges closer to the parking edge for better accessibility
    free_edges.sort(key=lambda edge: len(graph.dist_dict[pz.name][edge]))
    return free_edges


def _ion_is_idle_candidate(graph: Graph, ion: int, pz: ProcessingZone, busy_ions: set[int]) -> bool:
    if ion in busy_ions:
        return False
    current_edge = graph.state.get(ion)
    if current_edge is None:
        return False
    edge_idx = get_idx_from_idc(graph.idc_dict, current_edge)
    return _edge_is_memory_edge(graph, pz, edge_idx)
