# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Focused checks for paper-level aggregation and covariance settings."""

from __future__ import annotations

import reproduce.paper.analysis as paper_analysis
from mqt.ionshuttler.linear.actions import AdvanceTime
from mqt.ionshuttler.linear.architecture import Architecture
from mqt.ionshuttler.linear.schedule import ActionSchedule
from mqt.ionshuttler.linear.state import create_initial_state


def test_proxy_sweeps_hold_the_other_correlation_quasistatic(monkeypatch) -> None:
    architecture = Architecture(num_sites=1, processing_zones={"pz": [0]})
    schedule = ActionSchedule.from_actions(
        [AdvanceTime()],
        create_initial_state(1, architecture, initial_positions=[0]),
    )
    schedules = {
        "NoDD": (schedule, frozenset()),
        "FullSADD": (schedule, frozenset()),
    }
    correlations: list[tuple[float, float]] = []

    def record_correlations(
        *args: object,
        temporal_corr: float,
        spatial_corr: float,
        **kwargs: object,
    ) -> float:
        del args, kwargs
        correlations.append((temporal_corr, spatial_corr))
        return 0.0

    monkeypatch.setattr(paper_analysis, "_log_chi_ratio", record_correlations)
    paper_analysis.covariance_proxy_rows(
        schedules,
        architecture,
        temporal_scales=(2.0,),
        spatial_scales=(3.0,),
        dt_seconds=0.1,
    )

    assert correlations == [
        (float("inf"), float("inf")),
        (2.0, float("inf")),
        (float("inf"), 3.0),
    ]
