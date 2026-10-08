# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Settings for minimal Grid compilation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GridCompilerConfig:
    """Bound the deterministic Grid compiler.

    ``max_iterations`` bounds scheduling progress. ``max_routing_states``
    bounds each search for a route to a processing zone.
    """

    max_iterations: int = 10_000
    max_routing_states: int = 10_000

    def __post_init__(self) -> None:
        """Validate compiler limits."""
        _require_positive_int(self.max_iterations, "max_iterations")
        _require_positive_int(self.max_routing_states, "max_routing_states")


def _require_positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        msg = f"{name} must be an integer >= 1"
        raise ValueError(msg)


__all__ = ["GridCompilerConfig"]
