# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""End-to-end smoke test for the public-boundary paper quick run."""

from __future__ import annotations

import csv

from reproduce.paper.run import main


def test_quick_run_writes_outputs_extends_and_resumes(tmp_path, monkeypatch) -> None:
    output = tmp_path / "paper"
    assert main(("quick", "--output", str(output))) == 0
    expected = (
        output / "csv" / "raw_trajectories.csv",
        output / "csv" / "aggregate_metrics.csv",
        output / "csv" / "table_ii_source.csv",
        output / "checkpoints" / "compiled.json",
        output / "checkpoints" / "manifest.json",
        output / "checkpoints" / "sample_000000.csv",
        output / "figures" / "figure_2_operating_regime.pdf",
        output / "figures" / "figure_3_rerouting_benefit.pdf",
        output / "figures" / "figure_4_profile_awareness.pdf",
        output / "figures" / "figure_5_proxy_applicability.pdf",
        output / "figures" / "figure_6_compilation_cost.pdf",
        output / "figures" / "figure_7_objective_fidelity.pdf",
    )
    assert all(path.stat().st_size > 0 for path in expected)
    with (output / "csv" / "raw_trajectories.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    profile_baselines = [
        row["state_infidelity"] for row in rows if row["scenario"] == "profile_awareness" and row["method"] == "NoDD"
    ]
    assert len(set(profile_baselines)) == 1

    assert main(("quick", "--samples", "2", "--output", str(output))) == 0
    with (output / "csv" / "raw_trajectories.csv").open(newline="", encoding="utf-8") as handle:
        resumed_rows = list(csv.DictReader(handle))
    assert {row["sample"] for row in resumed_rows} == {"0", "1"}
    assert (output / "checkpoints" / "sample_000001.csv").is_file()

    def fail_if_simulated(*args: object, **kwargs: object) -> float:
        del args, kwargs
        msg = "completed samples must not be simulated again"
        raise AssertionError(msg)

    monkeypatch.setattr("reproduce.paper.run.simulate_infidelity", fail_if_simulated)
    assert main(("quick", "--samples", "2", "--output", str(output))) == 0
