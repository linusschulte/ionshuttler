# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the supported Grid import surface."""

from __future__ import annotations

import importlib


def test_package_exports_the_hardware_model() -> None:
    """Keep the package-level Grid API small and intentional."""
    package = importlib.import_module("mqt.ionshuttler.grid")

    assert package.__all__ == [
        "Cycle",
        "GridArchitecture",
        "GridMachineState",
        "Junction",
        "JunctionMove",
        "ProcessingZone",
        "Segment",
        "SegmentEndpoint",
        "SegmentOccupancy",
        "TransportTiming",
    ]
