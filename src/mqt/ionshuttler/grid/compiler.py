# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""User-facing entry point for minimal Grid compilation."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING

from mqt.ionshuttler.circuit import parse_circuit
from mqt.ionshuttler.core.gates import GateAction
from mqt.ionshuttler.core.result import CompilationResult, CompilationStatus
from mqt.ionshuttler.core.schedule import Schedule, ScheduledAction
from mqt.ionshuttler.grid.actions import Cycle, JunctionMove
from mqt.ionshuttler.grid.config import GridCompilerConfig
from mqt.ionshuttler.grid.result import GridCompilationResult, GridDiagnostics
from mqt.ionshuttler.grid.routing import route_gate

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mqt.ionshuttler.circuit import Circuit, CircuitInput
    from mqt.ionshuttler.core.actions import Action
    from mqt.ionshuttler.grid.architecture import GridArchitecture
    from mqt.ionshuttler.grid.state import GridMachineState


@dataclass(frozen=True)
class GridCompiler:
    """Compile supported circuits to a fixed Grid hardware model."""

    architecture: GridArchitecture
    config: GridCompilerConfig = field(default_factory=GridCompilerConfig)

    def compile(
        self,
        circuit: CircuitInput,
        *,
        initial_placement: Mapping[str, Sequence[int]] | None = None,
    ) -> GridCompilationResult:
        """Compile a circuit with bounded deterministic routing.

        Args:
            circuit: Circuit to compile.
            initial_placement: Optional segment-to-ion placement.

        Returns:
            A validated complete or partial compilation result.
        """
        started = perf_counter()
        gate_types = tuple(
            action_type
            for action_type in self.architecture.supported_action_types
            if issubclass(action_type, GateAction)
        )
        parsed = parse_circuit(circuit, gate_types=gate_types)
        placement = (
            _default_placement(parsed.num_ions, self.architecture) if initial_placement is None else initial_placement
        )
        state = self.architecture.initial_state(placement)
        _validate_circuit_ions(parsed, state)
        _validate_gate_capabilities(parsed, self.architecture)

        scheduled: list[ScheduledAction[Action]] = []
        completed: set[int] = set()
        iterations = 0
        explored_states = 0
        movement_count = 0
        cycle_count = 0
        gate_count = 0

        while len(completed) < len(parsed.gates) and iterations < self.config.max_iterations:
            iterations += 1
            ready = tuple(
                gate
                for gate, predecessors in zip(parsed.gates, parsed.predecessors, strict=True)
                if gate.gate_id not in completed and predecessors <= completed
            )
            gate_layer = _executable_gate_layer(self.architecture, state, ready, len(scheduled))
            if gate_layer:
                state = self.architecture.apply_layer(state, gate_layer)
                scheduled.extend(gate_layer)
                for scheduled_gate in gate_layer:
                    if isinstance(scheduled_gate.action, GateAction):
                        gate_id = scheduled_gate.action.gate_id
                        assert gate_id is not None
                        completed.add(gate_id)
                gate_count += len(gate_layer)
                state = state.at_time(max(scheduled_gate.end_time for scheduled_gate in gate_layer))
                continue
            if not ready:
                break
            gate = ready[0]
            target_segments = frozenset(
                self.architecture.processing_zone_segment(zone.zone_id).segment_id
                for zone in self.architecture.processing_zones
                if zone.supports(gate)
            )
            route, explored = route_gate(
                self.architecture,
                state,
                gate,
                target_segments,
                max_states=self.config.max_routing_states,
            )
            explored_states += explored
            if not route:
                break
            for action in route:
                duration = self.architecture.action_duration(action)
                scheduled_action = ScheduledAction(len(scheduled), action, state.time, duration)
                state = self.architecture.apply_layer(state, (scheduled_action,)).at_time(scheduled_action.end_time)
                scheduled.append(scheduled_action)
                if isinstance(action, Cycle):
                    cycle_count += 1
                    movement_count += len(action.moves)
                elif isinstance(action, JunctionMove):
                    movement_count += 1

        status = CompilationStatus.SUCCESS if len(completed) == len(parsed.gates) else CompilationStatus.FAILED
        schedule = Schedule(tuple(scheduled), state.time, self.architecture.initial_state(placement))
        final_state = self.architecture.replay_schedule(schedule)
        result = CompilationResult(
            status=status,
            schedule=schedule,
            architecture=self.architecture,
            final_state=final_state,
            wall_clock_s=perf_counter() - started,
            diagnostics=GridDiagnostics(
                iterations=iterations,
                explored_routing_states=explored_states,
                junction_moves=movement_count,
                cycles=cycle_count,
                gates=gate_count,
            ),
        )
        result.validate()
        return result


def _default_placement(num_ions: int, architecture: GridArchitecture) -> dict[str, tuple[int, ...]]:
    if num_ions > sum(segment.capacity for segment in architecture.segments):
        msg = "circuit has more ions than the Grid architecture capacity"
        raise ValueError(msg)
    remaining = iter(range(num_ions))
    placement: dict[str, tuple[int, ...]] = {}
    next_ion = next(remaining, None)
    for segment in architecture.segments:
        ions: list[int] = []
        while next_ion is not None and len(ions) < segment.capacity:
            ions.append(next_ion)
            next_ion = next(remaining, None)
        placement[segment.segment_id] = tuple(ions)
    return placement


def _validate_circuit_ions(circuit: Circuit, state: GridMachineState) -> None:
    if state.ions != tuple(range(circuit.num_ions)):
        msg = "initial placement must contain every circuit ion exactly once"
        raise ValueError(msg)


def _validate_gate_capabilities(circuit: Circuit, architecture: GridArchitecture) -> None:
    for gate in circuit.gates:
        if architecture.is_virtual_gate(gate):
            continue
        if not any(zone.supports(gate) for zone in architecture.processing_zones):
            msg = f"Grid architecture has no processing zone for {type(gate).__name__}"
            raise ValueError(msg)


def _executable_gate_layer(
    architecture: GridArchitecture,
    state: GridMachineState,
    ready: Sequence[GateAction],
    first_action_id: int,
) -> tuple[ScheduledAction[Action], ...]:
    """Return a greedy selection of ready gates that can start together now."""
    selected: list[ScheduledAction[Action]] = []
    for gate in ready:
        zone_ids: tuple[str | None, ...]
        if architecture.is_virtual_gate(gate):
            zone_ids = (None,)
        else:
            ion_segments = {state.ion_segment(ion) for ion in gate.ions}
            zone_ids = tuple(
                zone.zone_id
                for zone in architecture.processing_zones
                if zone.supports(gate)
                and architecture.processing_zone_segment(zone.zone_id).segment_id in ion_segments
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


__all__ = ["GridCompiler"]
