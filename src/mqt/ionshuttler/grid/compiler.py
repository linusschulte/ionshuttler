# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""User-facing entry point for Grid compilation."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING

from mqt.ionshuttler.circuit import parse_circuit
from mqt.ionshuttler.core.gates import GateAction
from mqt.ionshuttler.core.result import CompilationResult, CompilationStatus
from mqt.ionshuttler.grid.config import GridCompilerConfig, GridCompilerStrategy
from mqt.ionshuttler.grid.scheduling import schedule_breadth_first, schedule_greedy

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mqt.ionshuttler.circuit import Circuit, CircuitInput
    from mqt.ionshuttler.core.actions import Action
    from mqt.ionshuttler.core.schedule import Schedule
    from mqt.ionshuttler.grid.architecture import GridArchitecture
    from mqt.ionshuttler.grid.result import GridCompilationResult, GridDiagnostics
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
        """Compile a circuit with the configured scheduling strategy.

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
        initial_state = self.architecture.initial_state(placement)
        _validate_circuit_ions(parsed, initial_state)
        _validate_gate_capabilities(parsed, self.architecture)

        schedule, diagnostics = self._create_schedule(parsed, initial_state)
        status = CompilationStatus.SUCCESS if diagnostics.gates == len(parsed.gates) else CompilationStatus.FAILED
        result = CompilationResult(
            status=status,
            schedule=schedule,
            architecture=self.architecture,
            final_state=self.architecture.replay_schedule(schedule),
            wall_clock_s=perf_counter() - started,
            diagnostics=diagnostics,
        )
        result.validate()
        return result

    def _create_schedule(
        self,
        circuit: Circuit,
        initial_state: GridMachineState,
    ) -> tuple[Schedule[Action, GridMachineState], GridDiagnostics]:
        """Create a schedule with the selected strategy.

        Returns:
            The schedule and its diagnostic counters.
        """
        if self.config.strategy is GridCompilerStrategy.GREEDY:
            return schedule_greedy(self.architecture, circuit, initial_state, self.config)
        return schedule_breadth_first(self.architecture, circuit, initial_state, self.config)


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


__all__ = ["GridCompiler"]
