# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Small CSV aggregations and covariance proxies used by the paper plots."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from mqt.ionshuttler.linear.dd import compute_critical_segments

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

    from mqt.ionshuttler.linear import ActionSchedule, Architecture

RawValue = str | int | float
RawRow = dict[str, RawValue]


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write a nonempty sequence of consistently shaped mappings as CSV.

    Raises:
        ValueError: If ``rows`` is empty.
    """
    if not rows:
        msg = f"refusing to write empty CSV: {path}"
        raise ValueError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read one UTF-8 CSV table.

    Returns:
        The table's rows.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_infidelities(rows: Sequence[RawRow]) -> list[RawRow]:
    """Aggregate trajectories per scenario/method point with paired NoDD ratios.

    Returns:
        Per-case metric rows.
    """
    key_fields = ("scenario", "case", "method", "detuning", "control", "heating", "profile")
    grouped: dict[tuple[RawValue, ...], list[float]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in key_fields)].append(float(row["state_infidelity"]))
    means = {key: float(np.mean(values)) for key, values in grouped.items()}
    output: list[RawRow] = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        scenario, case, method, detuning, control, heating, profile = key
        reference_key = (scenario, case, "NoDD", detuning, control, heating, profile)
        reference = means.get(reference_key)
        ratio = ""
        if reference is not None:
            ratio = math.log10(max(means[key], 1e-15) / max(reference, 1e-15))
        output.append({
            "scenario": scenario,
            "case": case,
            "method": method,
            "detuning": detuning,
            "control": control,
            "heating": heating,
            "profile": profile,
            "mean_infidelity": means[key],
            "std_infidelity": float(np.std(values)),
            "mean_log10_ratio": ratio,
            "samples": len(values),
        })
    return output


def summarize_across_cases(
    rows: Sequence[RawRow],
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[RawRow]:
    """Average paired per-case log ratios with a circuit-bootstrap interval.

    Returns:
        Aggregate plot rows.
    """
    fields = ("scenario", "method", "detuning", "control", "heating", "profile")
    grouped: dict[tuple[RawValue, ...], list[float]] = defaultdict(list)
    for row in rows:
        value = row["mean_log10_ratio"]
        if isinstance(value, int | float):
            grouped[tuple(row[field] for field in fields)].append(float(value))
    output: list[RawRow] = []
    rng = np.random.default_rng(seed)
    for key, values in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        center = float(np.mean(values))
        if len(values) == 1:
            low = high = center
        else:
            samples = rng.choice(values, size=(bootstrap_samples, len(values)), replace=True)
            low, high = (float(value) for value in np.quantile(np.mean(samples, axis=1), (0.025, 0.975)))
        output.append({
            **dict(zip(fields, key, strict=True)),
            "mean_log10_ratio": center,
            "ci95_low": low,
            "ci95_high": high,
            "cases": len(values),
        })
    return output


def covariance_proxy_rows(
    schedules: Mapping[str, tuple[ActionSchedule, frozenset[int]]],
    architecture: Architecture,
    *,
    temporal_scales: Iterable[float],
    spatial_scales: Iterable[float],
    dt_seconds: float,
) -> tuple[list[RawRow], list[RawRow]]:
    """Calculate FullSADD/NoDD susceptibility ratios for temporal/spatial sweeps.

    Returns:
        Temporal and spatial proxy rows.
    """
    base_schedule, base_ids = schedules["NoDD"]
    full_schedule, full_ids = schedules["FullSADD"]
    temporal: list[RawRow] = []
    spatial: list[RawRow] = []
    rank_one_ratio = _log_chi_ratio(
        full_schedule,
        full_ids,
        base_schedule,
        base_ids,
        architecture,
        temporal_corr=1e12,
        spatial_corr=1e12,
        dt_seconds=dt_seconds,
    )
    for tau in temporal_scales:
        ratio = _log_chi_ratio(
            full_schedule,
            full_ids,
            base_schedule,
            base_ids,
            architecture,
            temporal_corr=float(tau),
            spatial_corr=2.0,
            dt_seconds=dt_seconds,
        )
        temporal.append({
            "tau_steps": tau,
            "tau_seconds": float(tau) * dt_seconds,
            "mean_R_chi": ratio,
            "ci95_low": ratio,
            "ci95_high": ratio,
            "quasistatic_R_J": rank_one_ratio,
            "window_min_seconds": dt_seconds,
            "window_max_seconds": base_schedule.num_timesteps * dt_seconds,
            "window_median_seconds": 0.5 * base_schedule.num_timesteps * dt_seconds,
            "window_p90_seconds": 0.9 * base_schedule.num_timesteps * dt_seconds,
        })
    for length in spatial_scales:
        ratio = _log_chi_ratio(
            full_schedule,
            full_ids,
            base_schedule,
            base_ids,
            architecture,
            temporal_corr=10000.0,
            spatial_corr=float(length),
            dt_seconds=dt_seconds,
        )
        spatial.append({
            "ell_sites": length,
            "mean_R_chi": ratio,
            "ci95_low": ratio,
            "ci95_high": ratio,
            "rank_one_R_chi": rank_one_ratio,
            "architecture_span_sites": architecture.num_sites - 1,
        })
    return temporal, spatial


def schedule_chi(
    schedule: ActionSchedule,
    local_ids: frozenset[int],
    architecture: Architecture,
    *,
    temporal_corr: float,
    spatial_corr: float,
    dt_seconds: float,
) -> float:
    """Return the summed dephasing susceptibility for one schedule.

    Returns:
        The nonnegative covariance-weighted susceptibility.
    """
    return _schedule_chi(schedule, local_ids, architecture, temporal_corr, spatial_corr, dt_seconds)


def _log_chi_ratio(
    candidate: ActionSchedule,
    candidate_ids: frozenset[int],
    reference: ActionSchedule,
    reference_ids: frozenset[int],
    architecture: Architecture,
    *,
    temporal_corr: float,
    spatial_corr: float,
    dt_seconds: float,
) -> float:
    candidate_chi = _schedule_chi(candidate, candidate_ids, architecture, temporal_corr, spatial_corr, dt_seconds)
    reference_chi = _schedule_chi(reference, reference_ids, architecture, temporal_corr, spatial_corr, dt_seconds)
    return math.log10(max(candidate_chi, 1e-30) / max(reference_chi, 1e-30))


def _schedule_chi(
    schedule: ActionSchedule,
    local_ids: frozenset[int],
    architecture: Architecture,
    temporal_corr: float,
    spatial_corr: float,
    dt_seconds: float,
) -> float:
    trace = compute_critical_segments(
        schedule,
        architecture,
        dt=dt_seconds,
        local_pulse_action_ids=local_ids,
    )
    total = 0.0
    for segment in trace.segments:
        positions = np.asarray(segment.positions, dtype=float)
        weights = np.asarray(segment.toggling_signs, dtype=float) * np.asarray(segment.sensitivities)
        spatial = np.exp(-np.abs(positions[:, None] - positions[None, :]) / spatial_corr)
        times = np.arange(len(positions), dtype=float)
        temporal = np.exp(-np.abs(times[:, None] - times[None, :]) / temporal_corr)
        total += float(0.5 * dt_seconds**2 * weights @ (spatial * temporal) @ weights)
    return total
