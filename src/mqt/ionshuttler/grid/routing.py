# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Bounded movement search for the minimal Grid compiler."""

from __future__ import annotations

from collections import deque
from itertools import combinations
from typing import TYPE_CHECKING

from mqt.ionshuttler.core.schedule import ScheduledAction
from mqt.ionshuttler.grid.actions import Cycle, JunctionMove
from mqt.ionshuttler.grid.model import SegmentOccupancy

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mqt.ionshuttler.core.actions import Action
    from mqt.ionshuttler.core.gates import GateAction
    from mqt.ionshuttler.grid.architecture import GridArchitecture
    from mqt.ionshuttler.grid.state import GridMachineState


def route_gate(
    architecture: GridArchitecture,
    state: GridMachineState,
    gate: GateAction,
    target_segment_ids: frozenset[str],
    *,
    max_states: int,
) -> tuple[tuple[Action, ...] | None, int]:
    """Find a shortest legal movement sequence that colocates one gate.

    Returns:
        The movement sequence, or ``None`` if the bound is exhausted, and the
        number of explored occupancy states.
    """
    queue: deque[tuple[GridMachineState, tuple[Action, ...]]] = deque([(state, ())])
    visited = {_occupancy_key(state)}
    explored = 0
    while queue and explored < max_states:
        current, path = queue.popleft()
        explored += 1
        if _gate_is_placed(current, gate, target_segment_ids):
            return path, explored
        for action in movement_candidates(architecture, current):
            duration = architecture.action_duration(action)
            scheduled_action = ScheduledAction(0, action, current.time, duration)
            try:
                following = architecture.apply_layer(current, (scheduled_action,)).at_time(scheduled_action.end_time)
            except ValueError:
                continue
            key = _occupancy_key(following)
            if key in visited:
                continue
            visited.add(key)
            queue.append((following, (*path, action)))
    return None, explored


def movement_candidates(architecture: GridArchitecture, state: GridMachineState) -> tuple[Action, ...]:
    """Return deterministic single-junction moves and simultaneous rotations."""
    moves: list[JunctionMove] = []
    for junction in architecture.junctions:
        for source in junction.endpoints:
            chains = _departure_chains(architecture, state, source.segment_id, source.orientation)
            for destination in junction.endpoints:
                if source.segment_id == destination.segment_id:
                    continue
                moves.extend(JunctionMove(source, destination, chain) for chain in chains)
    return (*moves, *_cycle_candidates(architecture, state, moves))


def _departure_chains(
    architecture: GridArchitecture,
    state: GridMachineState,
    segment_id: str,
    orientation: str,
) -> tuple[tuple[int, ...], ...]:
    """Return all ion chains that can leave a segment through one endpoint."""
    occupants = state.occupants(segment_id)
    if not occupants:
        return ()
    segment = architecture.segment(segment_id)
    if segment.occupancy is SegmentOccupancy.UNORDERED:
        return tuple(chain for length in range(1, len(occupants) + 1) for chain in combinations(occupants, length))
    if orientation == "start":
        return tuple(occupants[:length] for length in range(1, len(occupants) + 1))
    return tuple(occupants[len(occupants) - length :] for length in range(1, len(occupants) + 1))


def _cycle_candidates(
    architecture: GridArchitecture,
    state: GridMachineState,
    moves: Sequence[JunctionMove],
) -> tuple[Cycle, ...]:
    """Return legal simultaneous rotations built from one-ion junction moves."""
    # Restrict discovery to one-ion moves to bound the candidate combinations.
    # Cycle validation itself supports mixed chain sizes.
    one_ion_moves = tuple(move for move in moves if len(move.ions) == 1)
    by_source: dict[str, list[JunctionMove]] = {}
    for move in one_ion_moves:
        by_source.setdefault(move.source.segment_id, []).append(move)
    cycles: dict[tuple[tuple[str, str, str, str], ...], Cycle] = {}
    for start in sorted(by_source):
        _extend_cycles(
            architecture,
            state,
            start,
            start,
            by_source,
            (),
            frozenset(),
            frozenset(),
            cycles,
        )
    return tuple(cycles[key] for key in sorted(cycles))


def _extend_cycles(
    architecture: GridArchitecture,
    state: GridMachineState,
    start: str,
    current: str,
    by_source: dict[str, list[JunctionMove]],
    path: tuple[JunctionMove, ...],
    used_segments: frozenset[str],
    used_junctions: frozenset[str],
    cycles: dict[tuple[tuple[str, str, str, str], ...], Cycle],
) -> None:
    """Explore simple move paths and record cycles that close at the start segment."""
    if len(path) >= len(architecture.segments):
        return
    for move in by_source.get(current, []):
        junction_id = architecture.junction_for(move.source).junction_id
        destination = move.destination.segment_id
        if junction_id in used_junctions or move.ions[0] in {prior_move.ions[0] for prior_move in path}:
            continue
        extended = (*path, move)
        if destination == start:
            if len(extended) < 2:
                continue
            try:
                cycle = Cycle(extended)
                scheduled_cycle = ScheduledAction(0, cycle, state.time, architecture.action_duration(cycle))
                architecture.apply_layer(state, (scheduled_cycle,))
            except ValueError:
                continue
            key = _canonical_cycle_key(extended)
            cycles.setdefault(key, cycle)
            continue
        if destination in used_segments or destination == current:
            continue
        _extend_cycles(
            architecture,
            state,
            start,
            destination,
            by_source,
            extended,
            used_segments | {current},
            used_junctions | {junction_id},
            cycles,
        )


def _canonical_cycle_key(moves: Sequence[JunctionMove]) -> tuple[tuple[str, str, str, str], ...]:
    """Return a rotation-independent identity for a cycle's endpoint sequence."""
    parts = tuple(
        (
            move.source.segment_id,
            move.source.orientation,
            move.destination.segment_id,
            move.destination.orientation,
        )
        for move in moves
    )
    rotations = tuple((*parts[index:], *parts[:index]) for index in range(len(parts)))
    return min(rotations)


def _gate_is_placed(state: GridMachineState, gate: GateAction, target_segment_ids: frozenset[str]) -> bool:
    """Return whether all gate ions share a target processing-zone segment."""
    return (
        bool(gate.ions)
        and all(state.ion_segment(ion) in target_segment_ids for ion in gate.ions)
        and len({state.ion_segment(ion) for ion in gate.ions}) == 1
    )


def _occupancy_key(state: GridMachineState) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Return the position-only identity used to detect repeated search states."""
    return state.occupancy


__all__ = ["movement_candidates", "route_gate"]
