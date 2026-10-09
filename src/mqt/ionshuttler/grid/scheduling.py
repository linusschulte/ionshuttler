# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Scheduling strategies for Grid compilation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mqt.ionshuttler.core.gates import GateAction
from mqt.ionshuttler.core.schedule import Schedule, ScheduledAction
from mqt.ionshuttler.grid.actions import Cycle, JunctionMove
from mqt.ionshuttler.grid.result import GridDiagnostics
from mqt.ionshuttler.grid.routing import (
    junction_moves,
    next_segment_toward,
    route_gate_bfs,
    route_junction_crossing_greedy_cycle,
    routing_distance,
    select_greedy_target,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mqt.ionshuttler.circuit import Circuit
    from mqt.ionshuttler.core.actions import Action
    from mqt.ionshuttler.grid.architecture import GridArchitecture
    from mqt.ionshuttler.grid.config import GridCompilerConfig
    from mqt.ionshuttler.grid.state import GridMachineState


@dataclass(frozen=True)
class _GreedyGatePlan:
    """Store selected ready gates and the short scheduling horizon."""

    selected_ready_gates: tuple[GateAction, ...]
    lookahead_gates: tuple[GateAction, ...]


_GREEDY_LOOKAHEAD_ROUNDS = 5
_GREEDY_PRIORITY_IONS = 10


def schedule_breadth_first(
    architecture: GridArchitecture,
    circuit: Circuit,
    initial_state: GridMachineState,
    config: GridCompilerConfig,
) -> tuple[Schedule[Action, GridMachineState], GridDiagnostics]:
    """Schedule gates with a bounded shortest-route search.

    Returns:
        The resulting schedule prefix and diagnostic counters.
    """
    state = initial_state
    scheduled_actions: list[ScheduledAction[Action]] = []
    completed: set[int] = set()
    iterations = explored_states = movement_count = cycle_count = 0

    while len(completed) < len(circuit.gates) and iterations < config.max_iterations:
        iterations += 1
        ready = _ready_gates(circuit, completed)
        gate_layer = _startable_gate_layer(
            architecture,
            state,
            ready,
            len(scheduled_actions),
            gate_to_segment_lock={},
        )
        if gate_layer:
            state = architecture.apply_layer(state, gate_layer)
            scheduled_actions.extend(gate_layer)
            completed.update(_gate_ids(gate_layer))
            state = state.at_time(max(item.end_time for item in gate_layer))
            continue
        if not ready:
            break
        gate = ready[0]
        route, explored = route_gate_bfs(
            architecture,
            state,
            gate,
            frozenset(_candidate_target_segments(architecture, gate)),
            max_states=config.max_routing_states,
            allowed_crossings=config.allowed_junction_crossings,
        )
        explored_states += explored
        if not route:
            break
        for action in route:
            timed_action = ScheduledAction(
                len(scheduled_actions),
                action,
                state.time,
                architecture.action_duration(action),
            )
            state = architecture.apply_layer(state, (timed_action,)).at_time(timed_action.end_time)
            scheduled_actions.append(timed_action)
            movement_count, cycle_count = _count_transport(action, movement_count, cycle_count)

    return (
        Schedule(tuple(scheduled_actions), state.time, initial_state),
        GridDiagnostics(
            iterations=iterations,
            explored_routing_states=explored_states,
            junction_moves=movement_count,
            cycles=cycle_count,
            gates=len(completed),
        ),
    )


def schedule_greedy(
    architecture: GridArchitecture,
    circuit: Circuit,
    initial_state: GridMachineState,
    config: GridCompilerConfig,
) -> tuple[Schedule[Action, GridMachineState], GridDiagnostics]:
    """Schedule gates with distance-ranked lookahead and prioritized transport.

    Returns:
        The resulting schedule prefix and diagnostic counters.
    """
    state = initial_state
    scheduled_actions: list[ScheduledAction[Action]] = []
    completed: set[int] = set()
    started: dict[int, int] = {}
    gate_to_segment_lock: dict[int, str] = {}
    iterations = explored_states = movement_count = cycle_count = 0

    while len(completed) < len(circuit.gates) and iterations < config.max_iterations:
        iterations += 1
        _complete_started_gates(state.time, started, completed, gate_to_segment_lock)
        if len(completed) == len(circuit.gates):
            break
        plan = _greedy_gate_plan(
            architecture,
            circuit,
            state,
            completed,
            started,
            gate_to_segment_lock,
            config.allowed_junction_crossings,
        )
        gate_layer = _startable_gate_layer(
            architecture,
            state,
            plan.selected_ready_gates,
            len(scheduled_actions),
            gate_to_segment_lock=gate_to_segment_lock,
        )
        for item in gate_layer:
            gate = item.action
            assert isinstance(gate, GateAction)
            assert gate.gate_id is not None
            started[gate.gate_id] = item.end_time
        priority = _greedy_ion_priority(
            architecture,
            circuit,
            state,
            plan.lookahead_gates,
            completed,
            started,
            gate_to_segment_lock,
            config.allowed_junction_crossings,
        )
        retained_ions = _retained_pz_ions(
            architecture,
            state,
            plan.selected_ready_gates,
            gate_to_segment_lock,
            gate_layer,
        )
        movement_layer, explored = _priority_movement_layer(
            architecture,
            state,
            len(scheduled_actions) + len(gate_layer),
            priority,
            gate_layer,
            retained_ions,
            max_states=config.max_routing_states,
            allowed_crossings=config.allowed_junction_crossings,
        )
        explored_states += explored
        action_layer = (*gate_layer, *movement_layer)
        if not action_layer:
            next_time = _next_resource_release(state)
            if next_time is None:
                break
            state = state.at_time(next_time)
            continue
        state = architecture.apply_layer(state, action_layer)
        scheduled_actions.extend(action_layer)
        for movement_item in movement_layer:
            movement_count, cycle_count = _count_transport(
                movement_item.action,
                movement_count,
                cycle_count,
            )
        if any(item.end_time > state.time for item in action_layer):
            state = state.at_time(state.time + 1)

    if scheduled_actions:
        state = state.at_time(max(state.time, *(action.end_time for action in scheduled_actions)))
        _complete_started_gates(state.time, started, completed, gate_to_segment_lock)

    return (
        Schedule(tuple(scheduled_actions), state.time, initial_state),
        GridDiagnostics(
            iterations=iterations,
            explored_routing_states=explored_states,
            junction_moves=movement_count,
            cycles=cycle_count,
            gates=len(completed),
        ),
    )


def _ready_gates(circuit: Circuit, completed: set[int]) -> tuple[GateAction, ...]:
    """Return gates whose dependencies are complete."""
    return tuple(
        gate
        for gate, predecessors in zip(circuit.gates, circuit.predecessors, strict=True)
        if gate.gate_id not in completed and predecessors <= completed
    )


def _gate_ids(layer: Sequence[ScheduledAction[Action]]) -> set[int]:
    """Return the gate identifiers in one action layer."""
    return {
        item.action.gate_id for item in layer if isinstance(item.action, GateAction) and item.action.gate_id is not None
    }


def _count_transport(action: Action, moves: int, cycles: int) -> tuple[int, int]:
    """Add one transport action to the diagnostic counters.

    Returns:
        The updated junction-move and cycle counters.
    """
    if isinstance(action, Cycle):
        return moves + len(action.moves), cycles + 1
    if isinstance(action, JunctionMove):
        return moves + 1, cycles
    return moves, cycles


def _complete_started_gates(
    time: int,
    started: dict[int, int],
    completed: set[int],
    gate_to_segment_lock: dict[int, str],
) -> None:
    """Mark elapsed gates as complete and release their segment locks."""
    for gate_id in sorted(gate_id for gate_id, end_time in started.items() if end_time <= time):
        completed.add(gate_id)
        started.pop(gate_id)
        gate_to_segment_lock.pop(gate_id, None)


def _greedy_gate_plan(
    architecture: GridArchitecture,
    circuit: Circuit,
    state: GridMachineState,
    completed: set[int],
    started: Mapping[int, int],
    gate_to_segment_lock: dict[int, str],
    allowed_crossings: frozenset[tuple[str, str]] | None,
) -> _GreedyGatePlan:
    """Choose at most one nearby gate per processing-zone segment over a short dependency horizon.

    Returns:
        The selected ready gates and the distance-ranked lookahead order.
    """
    started_ids = set(started)
    virtually_completed = set(completed) | started_ids
    lookahead_gates: list[GateAction] = []
    selected_ready_gates: list[GateAction] = []
    target_order = {
        architecture.processing_zone_segment(zone.zone_id).segment_id: index
        for index, zone in enumerate(architecture.processing_zones)
    }
    for round_index in range(_GREEDY_LOOKAHEAD_ROUNDS):
        ready = tuple(gate for gate in _ready_gates(circuit, virtually_completed) if gate.gate_id not in started_ids)
        if not ready:
            break
        virtual = tuple(gate for gate in ready if architecture.is_virtual_gate(gate))
        gates_by_segment: dict[str, list[GateAction]] = defaultdict(list)
        for gate in ready:
            if architecture.is_virtual_gate(gate):
                continue
            target_segment_id = _select_gate_segment(
                architecture,
                state,
                gate,
                gate_to_segment_lock,
                allowed_crossings,
            )
            gates_by_segment[target_segment_id].append(gate)
        chosen = list(virtual)
        ordered_segment_ids = sorted(
            gates_by_segment,
            key=lambda item: (target_order.get(item, len(target_order)), item),
        )
        chosen.extend(
            min(
                gates_by_segment[target_segment_id],
                key=lambda gate: (
                    *_gate_distance_rank(
                        architecture,
                        state,
                        gate,
                        target_segment_id,
                        allowed_crossings,
                    ),
                    gate.gate_id,
                ),
            )
            for target_segment_id in ordered_segment_ids
        )
        chosen.sort(key=lambda gate: gate.gate_id if gate.gate_id is not None else -1)
        if round_index == 0:
            selected_ready_gates.extend(chosen)
        lookahead_gates.extend(chosen)
        virtually_completed.update(gate.gate_id for gate in chosen if gate.gate_id is not None)
    return _GreedyGatePlan(tuple(selected_ready_gates), tuple(lookahead_gates))


def _select_gate_segment(
    architecture: GridArchitecture,
    state: GridMachineState,
    gate: GateAction,
    gate_to_segment_lock: dict[int, str],
    allowed_crossings: frozenset[tuple[str, str]] | None,
) -> str:
    """Select a processing-zone segment and lock it for each pending two-ion gate.

    Returns:
        The selected processing-zone segment ID.
    """
    gate_id = gate.gate_id
    assert gate_id is not None
    if len(gate.ions) == 1 or gate_id not in gate_to_segment_lock:
        gate_to_segment_lock[gate_id] = select_greedy_target(
            architecture,
            state,
            gate,
            _candidate_target_segments(architecture, gate),
            allowed_crossings,
        )
    return gate_to_segment_lock[gate_id]


def _gate_distance_rank(
    architecture: GridArchitecture,
    state: GridMachineState,
    gate: GateAction,
    target_segment_id: str,
    allowed_crossings: frozenset[tuple[str, str]] | None,
) -> tuple[int, int]:
    """Return maximum and total operand distance for ready-gate selection."""
    distances = [
        routing_distance(architecture, state.ion_segment(ion), target_segment_id, allowed_crossings)
        for ion in gate.ions
    ]
    if any(distance is None for distance in distances):
        return (10**9, 10**9)
    reachable_distances = tuple(distance for distance in distances if distance is not None)
    return max(reachable_distances), sum(reachable_distances)


def _candidate_target_segments(
    architecture: GridArchitecture,
    gate: GateAction,
) -> tuple[str, ...]:
    """Return processing-zone segments that support the gate."""
    zones = tuple(zone for zone in architecture.processing_zones if zone.supports(gate))
    return tuple(dict.fromkeys(architecture.processing_zone_segment(zone.zone_id).segment_id for zone in zones))


def _greedy_ion_priority(
    architecture: GridArchitecture,
    circuit: Circuit,
    state: GridMachineState,
    ordered_gates: Sequence[GateAction],
    completed: set[int],
    started: Mapping[int, int],
    gate_to_segment_lock: dict[int, str],
    allowed_crossings: frozenset[tuple[str, str]] | None,
) -> tuple[tuple[int, str], ...]:
    """Build a capped first-use ion priority and apply the path-length filter.

    Returns:
        Prioritized ion and target-segment pairs selected for movement.
    """
    ordered_ids = {gate.gate_id for gate in ordered_gates}
    planned_ions: set[tuple[int, str]] = set()
    remaining = tuple(
        gate
        for gate in circuit.gates
        if gate.gate_id not in completed and gate.gate_id not in started and gate.gate_id not in ordered_ids
    )
    priority: dict[int, str] = {}
    for gate in (*ordered_gates, *remaining):
        if architecture.is_virtual_gate(gate):
            continue
        target_segment_id = _select_gate_segment(
            architecture,
            state,
            gate,
            gate_to_segment_lock,
            allowed_crossings,
        )
        for ion in gate.ions:
            priority.setdefault(ion, target_segment_id)
            if gate.gate_id in ordered_ids:
                planned_ions.add((ion, target_segment_id))
        if len(priority) >= _GREEDY_PRIORITY_IONS:
            break

    by_target: dict[str, list[int]] = defaultdict(list)
    for ion, target_segment_id in priority.items():
        by_target[target_segment_id].append(ion)
    selected: set[int] = set()
    for target_segment_id, ions in by_target.items():
        selected_distances: list[int] = []
        for index, ion in enumerate(ions):
            distance = routing_distance(
                architecture,
                state.ion_segment(ion),
                target_segment_id,
                allowed_crossings,
            )
            if distance is None:
                continue
            if (
                (distance == 0 and (ion, target_segment_id) in planned_ions)
                or index == 0
                or not any(selected_distances)
                or all(distance >= value for value in selected_distances)
            ):
                selected.add(ion)
                selected_distances.append(distance)
    return tuple((ion, target) for ion, target in priority.items() if ion in selected)


def _retained_pz_ions(
    architecture: GridArchitecture,
    state: GridMachineState,
    selected_gates: Sequence[GateAction],
    gate_to_segment_lock: Mapping[int, str],
    gate_layer: Sequence[ScheduledAction[Action]],
) -> frozenset[int]:
    """Return PZ ions needed by a selected, running, or starting gate."""
    retained: set[int] = set()
    for gate in selected_gates:
        if architecture.is_virtual_gate(gate) or gate.gate_id is None:
            continue
        target_segment_id = gate_to_segment_lock[gate.gate_id]
        retained.update(ion for ion in gate.ions if state.ion_segment(ion) == target_segment_id)
    busy_ions = {ion for ion, busy_until in state.ions_busy_until if busy_until > state.time}
    for zone_id, busy_until in state.pzs_busy_until:
        if busy_until > state.time:
            retained.update(
                busy_ions.intersection(state.occupants(architecture.processing_zone_segment(zone_id).segment_id))
            )
    for item in gate_layer:
        if isinstance(item.action, GateAction):
            retained.update(item.action.ions)
    return frozenset(retained)


def _priority_movement_layer(
    architecture: GridArchitecture,
    state: GridMachineState,
    first_action_id: int,
    priority: Sequence[tuple[int, str]],
    fixed_actions: Sequence[ScheduledAction[Action]],
    retained_ions: frozenset[int],
    *,
    max_states: int,
    allowed_crossings: frozenset[tuple[str, str]] | None,
) -> tuple[tuple[ScheduledAction[Action], ...], int]:
    """Pack next-step transport candidates in global ion-priority order.

    Returns:
        The selected transport layer and total candidate count.
    """
    selected_layer: list[ScheduledAction[Action]] = []
    included_actions: set[Action] = set()
    claimed_by_gates = {
        ion for item in fixed_actions if isinstance(item.action, GateAction) for ion in item.action.ions
    }
    available_moves = tuple(
        move
        for move in junction_moves(architecture, state, allowed_crossings=allowed_crossings)
        if not _moves_retained_ion(move, retained_ions)
    )
    ion_priorities = {ion: rank for rank, (ion, _target) in enumerate(priority)}
    ion_to_target_segment = dict(priority)
    explored = 0
    for ion, target_segment_id in priority:
        if ion in claimed_by_gates:
            continue
        source_segment_id = state.ion_segment(ion)
        if source_segment_id == target_segment_id:
            continue
        destination_segment_id = next_segment_toward(
            architecture,
            source_segment_id,
            target_segment_id,
            allowed_crossings,
        )
        if destination_segment_id is None:
            continue
        candidate_actions, candidates_explored = route_junction_crossing_greedy_cycle(
            architecture,
            state,
            ion,
            destination_segment_id,
            max_states=max_states,
            moves=available_moves,
            allowed_crossings=allowed_crossings,
            ion_priorities=ion_priorities,
        )
        explored += candidates_explored
        if not candidate_actions or any(action in included_actions for action in candidate_actions):
            continue
        candidate_layer: tuple[ScheduledAction[Action], ...] = tuple(
            ScheduledAction(
                first_action_id + len(selected_layer) + offset,
                action,
                state.time,
                architecture.action_duration(action),
            )
            for offset, action in enumerate(candidate_actions)
        )
        try:
            state_before_candidate = architecture.apply_layer(state, (*fixed_actions, *selected_layer))
            state_after_candidate = architecture.apply_layer(
                state,
                (*fixed_actions, *selected_layer, *candidate_layer),
            )
        except ValueError:
            continue
        if _harms_higher_priority_ion(
            architecture,
            state_before_candidate,
            state_after_candidate,
            ion,
            ion_priorities,
            ion_to_target_segment,
            allowed_crossings,
        ):
            continue
        included_actions.update(candidate_actions)
        selected_layer.extend(candidate_layer)
    return tuple(selected_layer), explored


def _moves_retained_ion(action: Action, retained_ions: frozenset[int]) -> bool:
    """Return whether a transport action moves a retained gate operand."""
    return isinstance(action, JunctionMove | Cycle) and not retained_ions.isdisjoint(action.ions)


def _harms_higher_priority_ion(
    architecture: GridArchitecture,
    before: GridMachineState,
    after: GridMachineState,
    requesting_ion: int,
    priorities: Mapping[int, int],
    ion_to_target_segment: Mapping[int, str],
    allowed_crossings: frozenset[tuple[str, str]] | None,
) -> bool:
    """Return whether candidate actions move a higher-priority ion away from its target."""
    requesting_priority = priorities[requesting_ion]
    for ion, priority in priorities.items():
        if priority >= requesting_priority or before.ion_segment(ion) == after.ion_segment(ion):
            continue
        target_segment_id = ion_to_target_segment[ion]
        distance_before = routing_distance(architecture, before.ion_segment(ion), target_segment_id, allowed_crossings)
        distance_after = routing_distance(architecture, after.ion_segment(ion), target_segment_id, allowed_crossings)
        if distance_before is not None and (distance_after is None or distance_after > distance_before):
            return True
    return False


def _next_resource_release(state: GridMachineState) -> int | None:
    """Return the next time at which a busy Grid resource becomes free."""
    releases = [
        release
        for _resource, release in (
            *state.ions_busy_until,
            *state.junctions_busy_until,
            *state.pzs_busy_until,
        )
        if release > state.time
    ]
    return min(releases, default=None)


def _startable_gate_layer(
    architecture: GridArchitecture,
    state: GridMachineState,
    candidate_gates: Sequence[GateAction],
    first_action_id: int,
    *,
    gate_to_segment_lock: Mapping[int, str],
) -> tuple[ScheduledAction[Action], ...]:
    """Return ready gates that can start together on the current hardware state."""
    selected: list[ScheduledAction[Action]] = []
    for gate in candidate_gates:
        zone_ids: tuple[str | None, ...]
        if architecture.is_virtual_gate(gate):
            zone_ids = (None,)
        else:
            ion_segments = {state.ion_segment(ion) for ion in gate.ions}
            locked_segment_id = gate_to_segment_lock.get(gate.gate_id) if gate.gate_id is not None else None
            zone_ids = tuple(
                zone.zone_id
                for zone in architecture.processing_zones
                if zone.supports(gate)
                and architecture.processing_zone_segment(zone.zone_id).segment_id in ion_segments
                and (
                    locked_segment_id is None
                    or architecture.processing_zone_segment(zone.zone_id).segment_id == locked_segment_id
                )
                and len(ion_segments) == 1
            )
        for zone_id in zone_ids:
            candidate: ScheduledAction[Action] = ScheduledAction(
                first_action_id + len(selected),
                gate,
                state.time,
                architecture.action_duration(gate),
                zone_id,
            )
            try:
                architecture.apply_layer(state, (*selected, candidate))
            except ValueError:
                continue
            selected.append(candidate)
            break
    return tuple(selected)


__all__ = ["schedule_breadth_first", "schedule_greedy"]
