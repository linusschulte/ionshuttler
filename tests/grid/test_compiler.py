# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Contract tests for minimal Grid compilation."""

from __future__ import annotations

import pytest

from mqt.ionshuttler.core.gates import GateAction
from mqt.ionshuttler.grid import (
    CompilationStatus,
    Cycle,
    GridArchitecture,
    GridCompiler,
    GridCompilerConfig,
    GridDiagnostics,
    Junction,
    JunctionMove,
    ProcessingZone,
    Segment,
    result_from_json,
)


def _qasm(num_ions: int, *instructions: str) -> str:
    return "\n".join(("OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{num_ions}];", *instructions))


def test_compiler_routes_a_two_ion_gate_and_round_trips_result() -> None:
    """Move a chain to a processing zone and retain the shared result contract."""
    memory = Segment("memory", capacity=2)
    processor = Segment("processor", capacity=2, processing_zones=(ProcessingZone("pz"),))
    architecture = GridArchitecture(
        (memory, processor),
        (Junction("j0", (memory.end, processor.start)),),
    )

    result = GridCompiler(architecture).compile(
        _qasm(2, "rxx(0.5) q[0],q[1];"),
        initial_placement={"memory": (0, 1)},
    )

    assert result.status is CompilationStatus.SUCCESS
    assert isinstance(result.path[0], JunctionMove)
    assert isinstance(result.path[-1], GateAction)
    assert result.final_state.occupants("processor") == (0, 1)
    assert result.diagnostics is not None
    assert result.diagnostics.junction_moves == 1
    assert result.diagnostics.cycles == 0
    assert result.diagnostics.gates == 1
    result.validate()
    assert result_from_json(result.to_json()) == result


def test_compiler_uses_a_simultaneous_cycle_when_every_segment_is_full() -> None:
    """Rotate a full ring when no capacity-safe single move exists."""
    a = Segment("a")
    b = Segment("b", processing_zones=(ProcessingZone("pz"),))
    c = Segment("c")
    architecture = GridArchitecture(
        (a, b, c),
        (
            Junction("ab", (a.end, b.start)),
            Junction("bc", (b.end, c.start)),
            Junction("ca", (c.end, a.start)),
        ),
    )

    result = GridCompiler(architecture).compile(
        _qasm(3, "rx(0.5) q[0];"),
        initial_placement={"a": (0,), "b": (1,), "c": (2,)},
    )

    assert result.status is CompilationStatus.SUCCESS
    assert isinstance(result.path[0], Cycle)
    assert result.final_state.ion_segment(0) == "b"
    assert result.diagnostics is not None
    assert result.diagnostics.cycles == 1
    assert result.diagnostics.junction_moves == 3


def test_compiler_schedules_independent_ready_gates_together() -> None:
    """Use separate processing zones for independent gates in one layer."""
    left = Segment("left", processing_zones=(ProcessingZone("left-pz"),))
    right = Segment("right", processing_zones=(ProcessingZone("right-pz"),))
    architecture = GridArchitecture((left, right), ())

    result = GridCompiler(architecture).compile(
        _qasm(2, "rx(0.5) q[0];", "ry(0.25) q[1];"),
        initial_placement={"left": (0,), "right": (1,)},
    )

    assert result.status is CompilationStatus.SUCCESS
    assert [item.start_time for item in result.schedule.scheduled_actions] == [0, 0]
    assert result.end_time == 1
    assert result.diagnostics is not None
    assert result.diagnostics.gates == 2


def test_compiler_preserves_gate_dependencies() -> None:
    """Start two gates on one ion in circuit order and on separate ticks."""
    processor = Segment("processor", processing_zones=(ProcessingZone("pz"),))
    architecture = GridArchitecture((processor,), ())

    result = GridCompiler(architecture).compile(
        _qasm(1, "rx(0.5) q[0];", "ry(0.25) q[0];"),
        initial_placement={"processor": (0,)},
    )

    assert result.status is CompilationStatus.SUCCESS
    assert all(isinstance(action, GateAction) for action in result.path)
    assert [action.gate_id for action in result.path if isinstance(action, GateAction)] == [0, 1]
    assert [item.start_time for item in result.schedule.scheduled_actions] == [0, 1]


def test_compiler_executes_virtual_gates_without_a_processing_zone() -> None:
    """Keep virtual single-ion gates independent of physical zone placement."""
    architecture = GridArchitecture((Segment("memory"),), ())

    result = GridCompiler(architecture).compile(
        _qasm(1, "rz(0.5) q[0];"),
        initial_placement={"memory": (0,)},
    )

    assert result.status is CompilationStatus.SUCCESS
    assert result.end_time == 0
    assert result.schedule.scheduled_actions[0].processing_zone_id is None


def test_compiler_returns_a_valid_failed_prefix_when_the_zone_is_unreachable() -> None:
    """Stop within the routing bound and return the replayable empty prefix."""
    memory = Segment("memory")
    processor = Segment("processor", processing_zones=(ProcessingZone("pz"),))
    architecture = GridArchitecture((memory, processor), ())

    result = GridCompiler(architecture, GridCompilerConfig(max_routing_states=2)).compile(
        _qasm(1, "rx(0.5) q[0];"),
        initial_placement={"memory": (0,)},
    )

    assert result.status is CompilationStatus.FAILED
    assert result.path == []
    assert result.final_state == result.initial_state
    assert result.diagnostics == GridDiagnostics(1, 1, 0, 0, 0)
    result.validate()


def test_compiler_uses_deterministic_default_placement() -> None:
    """Fill segments in architecture order when no placement is supplied."""
    first = Segment("first")
    second = Segment("second", processing_zones=(ProcessingZone("pz"),))
    architecture = GridArchitecture(
        (first, second),
        (Junction("j0", (first.end, second.start)),),
    )

    result = GridCompiler(architecture).compile(_qasm(1, "rx(0.5) q[0];"))

    assert result.status is CompilationStatus.SUCCESS
    assert result.initial_state.occupants("first") == (0,)


@pytest.mark.parametrize("field", ["max_iterations", "max_routing_states"])
def test_compiler_config_requires_positive_integer_bounds(field: str) -> None:
    """Reject routing limits that cannot permit compiler progress."""
    values = {"max_iterations": 1, "max_routing_states": 1, field: 0}

    with pytest.raises(ValueError, match=rf"{field} must be an integer >= 1"):
        GridCompilerConfig(**values)


def test_compiler_rejects_an_incomplete_initial_placement() -> None:
    """Require one placement entry for every circuit ion."""
    architecture = GridArchitecture((Segment("memory", capacity=2),), ())

    with pytest.raises(ValueError, match="initial placement must contain every circuit ion exactly once"):
        GridCompiler(architecture).compile(_qasm(2, "rz(0.5) q[0];"), initial_placement={"memory": (0,)})


def test_compiler_rejects_a_gate_without_a_capable_processing_zone() -> None:
    """Report missing hardware capability before routing starts."""
    architecture = GridArchitecture((Segment("memory"),), ())

    with pytest.raises(ValueError, match="Grid architecture has no processing zone for Rx"):
        GridCompiler(architecture).compile(_qasm(1, "rx(0.5) q[0];"))
