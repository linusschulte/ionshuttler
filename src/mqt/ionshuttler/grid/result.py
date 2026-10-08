# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Grid diagnostics, compilation results, and result persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from mqt.ionshuttler.core.actions import Action
from mqt.ionshuttler.core.result import CompilationResult as _CompilationResult
from mqt.ionshuttler.grid.actions import decode_grid_action
from mqt.ionshuttler.grid.architecture import GridArchitecture
from mqt.ionshuttler.grid.state import GridMachineState

from .._json_utils import require_int, require_mapping


@dataclass(frozen=True)
class GridDiagnostics:
    """Store statistics from Grid compilation."""

    iterations: int
    explored_routing_states: int
    junction_moves: int
    cycles: int
    gates: int

    def __post_init__(self) -> None:
        """Validate diagnostic counters.

        Raises:
            TypeError: If a counter is not an integer.
            ValueError: If a counter is negative.
        """
        for name in ("iterations", "explored_routing_states", "junction_moves", "cycles", "gates"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                msg = f"{name} must be an integer"
                raise TypeError(msg)
            if value < 0:
                msg = f"{name} must be non-negative"
                raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible diagnostics."""
        return {
            "iterations": self.iterations,
            "explored_routing_states": self.explored_routing_states,
            "junction_moves": self.junction_moves,
            "cycles": self.cycles,
            "gates": self.gates,
        }

    @classmethod
    def from_dict(cls, data: object) -> GridDiagnostics:
        """Restore Grid diagnostics.

        Returns:
            The restored diagnostics.
        """
        mapping = require_mapping(data, "Grid diagnostics")
        return cls(
            iterations=require_int(mapping, "iterations"),
            explored_routing_states=require_int(mapping, "explored_routing_states"),
            junction_moves=require_int(mapping, "junction_moves"),
            cycles=require_int(mapping, "cycles"),
            gates=require_int(mapping, "gates"),
        )


GridCompilationResult: TypeAlias = _CompilationResult[GridArchitecture, Action, GridMachineState, GridDiagnostics]


def result_from_dict(data: object) -> GridCompilationResult:
    """Restore a Grid compilation result.

    Returns:
        The restored result.
    """
    return _CompilationResult.from_dict(
        data,
        decode_architecture=GridArchitecture.from_dict,
        decode_action=decode_grid_action,
        decode_state=GridMachineState.from_dict,
        decode_diagnostics=GridDiagnostics.from_dict,
    )


def result_from_json(raw: str) -> GridCompilationResult:
    """Restore a Grid compilation result from JSON text.

    Returns:
        The restored result.
    """
    return result_from_dict(json.loads(raw))


def load_result(filename: str | Path) -> GridCompilationResult:
    """Load a Grid compilation result from a UTF-8 JSON file.

    Returns:
        The restored result.
    """
    return result_from_json(Path(filename).read_text(encoding="utf-8"))


__all__ = [
    "GridCompilationResult",
    "GridDiagnostics",
    "load_result",
    "result_from_dict",
    "result_from_json",
]
