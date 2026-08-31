# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Compile, simulate, aggregate, and plot the DD/SADD paper reproduction."""

from __future__ import annotations

import argparse
import hashlib
import math
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from qiskit import QuantumCircuit

from mqt.ionshuttler.linear import (
    Architecture,
    CompilationStatus,
    GateTiming,
    HardwareTiming,
    LinearCompiler,
    LinearCompilerConfig,
    SearchConfig,
    TransportTiming,
)
from mqt.ionshuttler.linear.actions import PhysicalSwap, Shuttle
from mqt.ionshuttler.linear.dd import (
    SADDConfig,
    SADDMethod,
    apply_idealized_hahn,
    compute_critical_segments,
    run_nearest_hahn,
    run_sadd,
)
from mqt.ionshuttler.linear.field_profile import FieldProfile
from reproduce.paper.analysis import (
    RawRow,
    covariance_proxy_rows,
    schedule_chi,
    summarize_across_cases,
    summarize_infidelities,
    write_csv,
)
from reproduce.paper.plots import render_all
from reproduce.paper.simulation import (
    PaperNoise,
    build_noisy_circuit,
    build_reference_circuit,
    local_pulse_action_ids,
    simulate_infidelity,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mqt.ionshuttler.linear import ActionSchedule

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("paper.toml")
METHODS = ("NoDD", "IdealizedHahn", "NearestHahn", "PulseOnlySADD", "FullSADD")
ROUTING_METHODS = ("NoDD", "NearestHahn", "PulseOnlySADD", "FullSADD")


@dataclass(frozen=True)
class Case:
    """One deterministically generated paper circuit."""

    family: str
    qubits: int
    seed: int

    @property
    def name(self) -> str:
        """Stable case label used in CSV rows."""
        return f"{self.family}_{self.qubits}"


@dataclass(frozen=True)
class RunSettings:
    """Resolved mode-specific settings used by the command."""

    mode: str
    cases: tuple[Case, ...]
    samples: int
    bootstrap_samples: int
    detunings: tuple[float, ...]
    controls: tuple[float, ...]
    heating: tuple[float, ...]
    profile_heating: tuple[float, ...]
    temporal_scales: tuple[float, ...]
    spatial_scales: tuple[float, ...]


@dataclass(frozen=True)
class MethodAudit:
    """Timing and diagnostics retained for result figures and Table II."""

    runtime_seconds: float
    report: object | None = None


def main(argv: Sequence[str] | None = None) -> int:
    """Run the reproduction command.

    Returns:
        A successful process status.

    Raises:
        RuntimeError: If compilation or a DD pass cannot produce a schedule.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("quick", "paper"))
    parser.add_argument("--output", type=Path, required=True, help="fresh or replaceable output directory")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    config = _load_config(args.config)
    settings = _settings(config, args.mode)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    architecture, profile_architecture = _architectures(_table(config, "architecture"))
    compiler = _compiler(architecture, _table(config, "compiler"))
    dd_config = _dd_config(_table(config, "dd"))
    simulation_config = _table(config, "simulation")
    sweep_config = _table(config, "sweeps")

    raw_rows: list[RawRow] = []
    proxy_temporal: list[RawRow] = []
    proxy_spatial: list[RawRow] = []
    schedule_rows: list[RawRow] = []
    opportunity_rows: list[RawRow] = []
    for case_index, case in enumerate(settings.cases, start=1):
        print(f"[{case_index}/{len(settings.cases)}] compiling {case.name}", flush=True)
        result = compiler.compile(_make_circuit(case))
        if result.status is not CompilationStatus.SUCCESS:
            msg = f"compilation of {case.name} ended with {result.status.value}"
            raise RuntimeError(msg)
        schedules, audits, case_opportunities = _schedule_variants(result.schedule, architecture, dd_config)
        aware_full = _full_sadd(result.schedule, profile_architecture, dd_config)
        schedule_rows.extend(
            _schedule_audit_rows(
                case,
                schedules,
                audits,
                architecture,
                dt_seconds=_number(simulation_config, "dt_seconds"),
                temporal_corr=_number(simulation_config, "dephasing_correlation_time_steps"),
            )
        )
        opportunity_rows.extend({"case": case.name, "family": case.family, **row} for row in case_opportunities)
        print(f"[{case_index}/{len(settings.cases)}] simulating {case.name}", flush=True)
        raw_rows.extend(
            _simulate_case(
                case,
                schedules,
                aware_full,
                architecture,
                profile_architecture,
                simulation_config,
                sweep_config,
                settings,
            )
        )
        temporal, spatial = covariance_proxy_rows(
            schedules,
            architecture,
            temporal_scales=settings.temporal_scales,
            spatial_scales=settings.spatial_scales,
            dt_seconds=_number(simulation_config, "dt_seconds"),
        )
        proxy_temporal.extend({"case": case.name, **row} for row in temporal)
        proxy_spatial.extend({"case": case.name, **row} for row in spatial)

    per_case = summarize_infidelities(raw_rows)
    aggregate = summarize_across_cases(
        per_case,
        bootstrap_samples=settings.bootstrap_samples,
        seed=_integer(simulation_config, "seed"),
    )
    temporal = _average_proxy(proxy_temporal, "tau_steps")
    spatial = _average_proxy(proxy_spatial, "ell_sites")
    table_ii = _table_ii_rows(per_case, schedule_rows, sweep_config, settings)
    objective = _objective_fidelity_rows(
        per_case,
        schedule_rows,
        sweep_config,
        settings,
        seed=_integer(simulation_config, "seed"),
    )
    csv_dir = output / "csv"
    write_csv(csv_dir / "raw_trajectories.csv", raw_rows)
    write_csv(csv_dir / "per_case_metrics.csv", per_case)
    write_csv(csv_dir / "aggregate_metrics.csv", aggregate)
    write_csv(csv_dir / "schedule_summary.csv", schedule_rows)
    write_csv(csv_dir / "runtime_opportunities.csv", opportunity_rows)
    write_csv(csv_dir / "table_ii_source.csv", table_ii)
    write_csv(csv_dir / "objective_fidelity.csv", objective)
    write_csv(csv_dir / "temporal_applicability.csv", temporal)
    write_csv(csv_dir / "spatial_applicability.csv", spatial)
    print("rendering six manuscript result figures", flush=True)
    figures = render_all(
        aggregate,
        per_case,
        temporal,
        spatial,
        schedule_rows,
        opportunity_rows,
        objective,
        output / "figures",
    )
    print("generated:", flush=True)
    for path in sorted((*csv_dir.glob("*.csv"), *figures)):
        print(f"  {path}", flush=True)
    return 0


def _load_config(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        return cast("dict[str, object]", tomllib.load(handle))


def _table(config: Mapping[str, object], name: str) -> dict[str, object]:
    value = config.get(name)
    if not isinstance(value, dict):
        msg = f"configuration requires a [{name}] table"
        raise TypeError(msg)
    return cast("dict[str, object]", value)


def _number(config: Mapping[str, object], name: str) -> float:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"configuration value {name!r} must be numeric"
        raise TypeError(msg)
    return float(value)


def _integer(config: Mapping[str, object], name: str) -> int:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"configuration value {name!r} must be an integer"
        raise TypeError(msg)
    return value


def _float_tuple(config: Mapping[str, object], name: str) -> tuple[float, ...]:
    value = config.get(name)
    if not isinstance(value, list) or not value:
        msg = f"configuration value {name!r} must be a nonempty array"
        raise TypeError(msg)
    if any(isinstance(item, bool) or not isinstance(item, int | float) for item in value):
        msg = f"configuration value {name!r} must contain only numbers"
        raise TypeError(msg)
    return tuple(float(cast("int | float", item)) for item in value)


def _settings(config: Mapping[str, object], mode: str) -> RunSettings:
    raw_cases = config.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        msg = "configuration requires at least one [[cases]] table"
        raise ValueError(msg)
    parsed: list[Case] = []
    for item in raw_cases:
        if not isinstance(item, dict):
            msg = "each cases entry must be a table"
            raise TypeError(msg)
        entry = cast("dict[str, object]", item)
        family = entry.get("family")
        if not isinstance(family, str):
            msg = "case family must be a string"
            raise TypeError(msg)
        parsed.append(Case(family, _integer(entry, "qubits"), _integer(entry, "seed")))
    cases = tuple(parsed)
    sim = _table(config, "simulation")
    sweeps = _table(config, "sweeps")
    selected = sweeps
    samples = _integer(sim, "samples")
    bootstrap_samples = _integer(sim, "bootstrap_samples")
    if mode == "quick":
        quick = _table(config, "quick")
        cases = cases[: _integer(quick, "case_count")]
        samples = _integer(quick, "samples")
        bootstrap_samples = _integer(quick, "bootstrap_samples")
        selected = quick
    return RunSettings(
        mode=mode,
        cases=cases,
        samples=samples,
        bootstrap_samples=bootstrap_samples,
        detunings=_float_tuple(selected, "detuning"),
        controls=_float_tuple(selected, "control"),
        heating=_float_tuple(selected, "heating"),
        profile_heating=_float_tuple(selected, "profile_heating"),
        temporal_scales=_float_tuple(selected, "temporal_correlation_steps"),
        spatial_scales=_float_tuple(selected, "spatial_correlation_sites"),
    )


def _architectures(config: Mapping[str, object]) -> tuple[Architecture, Architecture]:
    num_sites = _integer(config, "num_sites")
    raw_zones = config.get("processing_zones")
    if not isinstance(raw_zones, dict):
        msg = "architecture.processing_zones must be a table"
        raise TypeError(msg)
    zones: dict[str, tuple[int, ...]] = {}
    for name, sites in raw_zones.items():
        if not isinstance(sites, list) or any(isinstance(site, bool) or not isinstance(site, int) for site in sites):
            msg = f"processing zone {name!r} must contain integer sites"
            raise TypeError(msg)
        zones[str(name)] = tuple(cast("list[int]", sites))
    flat = Architecture(num_sites=num_sites, processing_zones=zones)
    profile_values = _float_tuple(config, "profile_site_field")
    if len(profile_values) != num_sites:
        msg = "architecture.profile_site_field must contain one value per site"
        raise ValueError(msg)
    profile = FieldProfile(num_sites, tuple(enumerate(profile_values)))
    return flat, Architecture(num_sites=num_sites, processing_zones=zones, field_profile=profile)


def _compiler(architecture: Architecture, config: Mapping[str, object]) -> LinearCompiler:
    timing = HardwareTiming(
        transport=TransportTiming(shuttle=_integer(config, "shuttle_duration"), swap=_integer(config, "swap_duration")),
        gates=GateTiming(
            rx=_integer(config, "rx_duration"),
            ry=_integer(config, "ry_duration"),
            rz=_integer(config, "rz_duration"),
            rzz=_integer(config, "rzz_duration"),
        ),
    )
    search = SearchConfig(
        horizon=_integer(config, "horizon"),
        committed_gates=_integer(config, "committed_gates"),
        max_frontier_size=_integer(config, "max_frontier_size"),
        max_compile_time=_number(config, "max_compile_time"),
        iterative_diving_search=bool(config["iterative_diving_search"]),
        informed_action_prioritization=bool(config["informed_action_prioritization"]),
        use_dependencies=bool(config["use_dependencies"]),
    )
    return LinearCompiler(architecture, LinearCompilerConfig(hardware_timing=timing, search=search))


def _dd_config(config: Mapping[str, object]) -> SADDConfig:
    return SADDConfig(
        min_window_length=_integer(config, "min_window_length"),
        max_window_length=_integer(config, "max_window_length"),
        max_participating_ions=_integer(config, "max_participating_ions"),
        timeout_s=_number(config, "timeout_seconds"),
        num_search_workers=_integer(config, "num_search_workers"),
    )


def _make_circuit(case: Case) -> QuantumCircuit:
    if case.family == "ising":
        circuit = QuantumCircuit(case.qubits, name=case.name)
        for _round in range(6):
            for qubit in range(case.qubits - 1):
                circuit.rz(0.2, qubit)
            circuit.rz(0.2, case.qubits - 1)
            for qubit in range(case.qubits):
                circuit.ry(math.pi / 2, qubit)
            for qubit in range(case.qubits - 1):
                circuit.rzz(0.1, qubit, qubit + 1)
            circuit.rzz(0.1, 0, case.qubits - 1)
            for qubit in range(case.qubits):
                circuit.ry(-math.pi / 2, qubit)
        return circuit

    benchmark = {"qft": "qft", "random": "randomcircuit", "qpe": "qpeexact", "ghz": "ghz"}.get(case.family)
    if benchmark is None:
        msg = f"unsupported paper circuit family: {case.family!r}"
        raise ValueError(msg)
    try:
        # The repository lint environment intentionally excludes the paper-only group.
        from mqt.bench.targets import get_target_for_gateset  # ty: ignore[unresolved-import]

        from mqt import bench  # ty: ignore[unresolved-import]
    except ImportError as error:
        msg = "paper circuit generation requires the `paper` dependency group (run `uv sync --group paper`)"
        raise RuntimeError(msg) from error
    return bench.get_benchmark(
        benchmark=benchmark,
        level=bench.BenchmarkLevel.NATIVEGATES,
        circuit_size=case.qubits,
        target=get_target_for_gateset("quantinuum", case.qubits),
        opt_level=2,
        random_parameters=True,
    )


def _schedule_variants(
    base: ActionSchedule, architecture: Architecture, config: SADDConfig
) -> tuple[
    dict[str, tuple[ActionSchedule, frozenset[int]]],
    dict[str, MethodAudit],
    list[RawRow],
]:
    start = time.perf_counter()
    idealized = apply_idealized_hahn(base, architecture)
    idealized_runtime = time.perf_counter() - start
    start = time.perf_counter()
    nearest = run_nearest_hahn(base, architecture)
    nearest_runtime = time.perf_counter() - start
    start = time.perf_counter()
    pulse = run_sadd(base, architecture, SADDMethod.PULSE_ONLY, config)
    pulse_runtime = time.perf_counter() - start
    start = time.perf_counter()
    full = run_sadd(base, architecture, SADDMethod.FULL, config)
    full_runtime = time.perf_counter() - start
    for name, result in (("PulseOnlySADD", pulse), ("FullSADD", full)):
        if result.unavailable_reason is not None:
            msg = f"{name} unavailable: {result.unavailable_reason}"
            raise RuntimeError(msg)
    schedules = {
        "NoDD": (base, frozenset()),
        "IdealizedHahn": (idealized.schedule, local_pulse_action_ids(idealized.report)),
        "NearestHahn": (nearest.schedule, local_pulse_action_ids(nearest.report)),
        "PulseOnlySADD": (pulse.schedule, local_pulse_action_ids(pulse.report)),
        "FullSADD": (full.schedule, local_pulse_action_ids(full.report)),
    }
    audits = {
        "NoDD": MethodAudit(0.0),
        "IdealizedHahn": MethodAudit(idealized_runtime, idealized.report),
        "NearestHahn": MethodAudit(nearest_runtime, nearest.report),
        "PulseOnlySADD": MethodAudit(pulse_runtime, pulse.report),
        "FullSADD": MethodAudit(full_runtime, full.report),
    }
    opportunities: list[RawRow] = [
        {
            "opportunity": index,
            "window_start": record.window[0],
            "window_end": record.window[1],
            "participating_ions": len(record.participating_ions),
            "accepted": record.accepted,
            "runtime_seconds": record.runtime_s,
            "model_variables": record.model_num_variables,
            "model_constraints": record.model_num_constraints,
        }
        for index, record in enumerate(full.report.opportunities)
    ]
    return schedules, audits, opportunities


def _full_sadd(
    base: ActionSchedule, architecture: Architecture, config: SADDConfig
) -> tuple[ActionSchedule, frozenset[int]]:
    result = run_sadd(base, architecture, SADDMethod.FULL, config)
    if result.unavailable_reason is not None:
        msg = f"profile-aware FullSADD unavailable: {result.unavailable_reason}"
        raise RuntimeError(msg)
    return result.schedule, local_pulse_action_ids(result.report)


def _schedule_audit_rows(
    case: Case,
    schedules: Mapping[str, tuple[ActionSchedule, frozenset[int]]],
    audits: Mapping[str, MethodAudit],
    architecture: Architecture,
    *,
    dt_seconds: float,
    temporal_corr: float,
) -> list[RawRow]:
    rows: list[RawRow] = []
    for method, (schedule, local_ids) in schedules.items():
        report = audits[method].report
        opportunities = tuple(getattr(report, "opportunities", ()))
        accepted = tuple(record for record in opportunities if getattr(record, "accepted", False))
        rerouting_pulses = sum(
            int(getattr(record, "pulse_count", 0))
            for record in accepted
            if any(int(delta) != 0 for delta in getattr(record, "transport_delta", {}).values())
        )
        trace = compute_critical_segments(
            schedule,
            architecture,
            dt=dt_seconds,
            local_pulse_action_ids=local_ids,
        )
        rows.append({
            "case": case.name,
            "family": case.family,
            "num_ions": case.qubits,
            "method": method,
            "timesteps": schedule.num_timesteps,
            "actions": len(schedule.scheduled_actions),
            "transport_actions": sum(isinstance(action, Shuttle | PhysicalSwap) for action in schedule.path),
            "local_pulses": len(local_ids),
            "rerouting_pulses": rerouting_pulses,
            "opportunities_evaluated": len(opportunities),
            "opportunities_accepted": len(accepted),
            "local_spacetime_volume": sum(
                len(getattr(record, "participating_ions", ()))
                * (getattr(record, "window", (0, 0))[1] - getattr(record, "window", (0, 0))[0])
                for record in opportunities
            ),
            "model_variables": sum(int(getattr(record, "model_num_variables", 0)) for record in opportunities),
            "model_constraints": sum(int(getattr(record, "model_num_constraints", 0)) for record in opportunities),
            "schedule_ion_timesteps": case.qubits * schedule.num_timesteps,
            "phase_cost": trace.phase_cost,
            "chi": schedule_chi(
                schedule,
                local_ids,
                architecture,
                temporal_corr=temporal_corr,
                spatial_corr=1e12,
                dt_seconds=dt_seconds,
            ),
            "runtime_seconds": audits[method].runtime_seconds,
        })
    return rows


def _table_ii_rows(
    metrics: Sequence[RawRow],
    schedules: Sequence[RawRow],
    sweeps: Mapping[str, object],
    settings: RunSettings,
) -> list[RawRow]:
    detuning = _number(sweeps, "profile_detuning")
    if detuning not in settings.detunings:
        detuning = max(settings.detunings)
    heating = _number(sweeps, "selected_heating")
    if heating not in settings.heating:
        heating = settings.heating[0]
    control = _number(sweeps, "fixed_control")
    metric_index = {
        (
            str(row["case"]),
            str(row["method"]),
            float(row["detuning"]),
            float(row["control"]),
            float(row["heating"]),
        ): row
        for row in metrics
        if row["scenario"] == "control_heating" and row["profile"] == "flat"
    }
    schedule_index = {(str(row["case"]), str(row["method"])): row for row in schedules}
    output: list[RawRow] = []
    for case in sorted({str(row["case"]) for row in schedules}):
        full_metric = metric_index[case, "FullSADD", detuning, control, heating]
        pulse_metric = metric_index[case, "PulseOnlySADD", detuning, control, heating]
        full = schedule_index[case, "FullSADD"]
        pulse = schedule_index[case, "PulseOnlySADD"]
        base = schedule_index[case, "NoDD"]
        output.append({
            "case": case,
            "family": full["family"],
            "num_ions": full["num_ions"],
            "full_vs_nodd_infidelity": 10 ** float(full_metric["mean_log10_ratio"]),
            "full_vs_nodd_chi": float(full["chi"]) / float(base["chi"]),
            "full_vs_pulse_infidelity": float(full_metric["mean_infidelity"])
            / max(float(pulse_metric["mean_infidelity"]), 1e-15),
            "full_vs_pulse_chi": float(full["chi"]) / max(float(pulse["chi"]), 1e-30),
            "nodd_transport": base["transport_actions"],
            "full_transport": full["transport_actions"],
            "pulses": full["local_pulses"],
            "rerouting_pulses": full["rerouting_pulses"],
            "opportunities_evaluated": full["opportunities_evaluated"],
            "opportunities_accepted": full["opportunities_accepted"],
            "sadd_runtime_seconds": full["runtime_seconds"],
        })
    return output


def _objective_fidelity_rows(
    metrics: Sequence[RawRow],
    schedules: Sequence[RawRow],
    sweeps: Mapping[str, object],
    settings: RunSettings,
    *,
    seed: int,
) -> list[RawRow]:
    detuning = _number(sweeps, "fixed_detuning")
    if detuning not in settings.detunings:
        detuning = max(settings.detunings)
    schedule_index = {(str(row["case"]), str(row["method"])): row for row in schedules}
    controls = sorted({float(row["control"]) for row in metrics if row["scenario"] == "operating_envelope"})
    output: list[RawRow] = []
    points_by_control: dict[float, list[tuple[float, float]]] = {}
    for row in metrics:
        if (
            row["scenario"] != "operating_envelope"
            or row["method"] != "FullSADD"
            or not math.isclose(float(row["detuning"]), detuning)
        ):
            continue
        case = str(row["case"])
        full = schedule_index[case, "FullSADD"]
        base = schedule_index[case, "NoDD"]
        rj = math.log10(max(float(full["phase_cost"]), 1e-30) / max(float(base["phase_cost"]), 1e-30))
        ri = float(row["mean_log10_ratio"])
        control = float(row["control"])
        points_by_control.setdefault(control, []).append((rj, ri))
        if math.isclose(control, _number(sweeps, "fixed_control")):
            output.append({
                "kind": "case",
                "case": case,
                "family": full["family"],
                "num_ions": full["num_ions"],
                "rj": rj,
                "ri": ri,
                "control": control,
                "pearson_r": "",
                "ci95_low": "",
                "ci95_high": "",
            })
    for control in controls:
        points = points_by_control.get(control, [])
        x = np.asarray([point[0] for point in points])
        y = np.asarray([point[1] for point in points])
        correlation = 0.0
        low = 0.0
        high = 0.0
        if len(points) >= 2 and float(np.std(x)) > 1e-12 and float(np.std(y)) > 1e-12:
            correlation = float(np.corrcoef(x, y)[0, 1])
            rng = np.random.default_rng(_seed(seed, "correlation", control))
            bootstrap: list[float] = []
            for _sample in range(settings.bootstrap_samples):
                indices = rng.integers(0, len(points), len(points))
                bx = x[indices]
                by = y[indices]
                if float(np.std(bx)) > 1e-12 and float(np.std(by)) > 1e-12:
                    bootstrap.append(float(np.corrcoef(bx, by)[0, 1]))
            if bootstrap:
                low, high = (float(value) for value in np.quantile(bootstrap, (0.025, 0.975)))
        output.append({
            "kind": "correlation",
            "case": "",
            "family": "",
            "num_ions": "",
            "rj": "",
            "ri": "",
            "control": control,
            "pearson_r": correlation,
            "ci95_low": low,
            "ci95_high": high,
        })
    return output


def _simulate_case(
    case: Case,
    schedules: Mapping[str, tuple[ActionSchedule, frozenset[int]]],
    aware_full: tuple[ActionSchedule, frozenset[int]],
    architecture: Architecture,
    profile_architecture: Architecture,
    sim: Mapping[str, object],
    sweeps: Mapping[str, object],
    settings: RunSettings,
) -> list[RawRow]:
    rows: list[RawRow] = []
    for detuning in settings.detunings:
        for control in settings.controls:
            for method in METHODS:
                rows.extend(
                    _trajectory_rows(
                        case,
                        "operating_envelope",
                        method,
                        schedules[method],
                        architecture,
                        sim,
                        settings.samples,
                        detuning,
                        control,
                        0.0,
                        "flat",
                    )
                )
    routing_detunings = _panel_detunings(settings.detunings)
    fixed_control = _number(sweeps, "fixed_control")
    for detuning in routing_detunings:
        for heating in settings.heating:
            for method in ROUTING_METHODS:
                rows.extend(
                    _trajectory_rows(
                        case,
                        "control_heating",
                        method,
                        schedules[method],
                        architecture,
                        sim,
                        settings.samples,
                        detuning,
                        fixed_control,
                        heating,
                        "flat",
                    )
                )
    profile = tuple(float(profile_architecture.field_at(site)) for site in range(architecture.num_sites))
    for detuning in routing_detunings:
        for heating in settings.profile_heating:
            for profile_name, full_schedule in (("aware", aware_full), ("agnostic", schedules["FullSADD"])):
                for method, schedule in (("NoDD", schedules["NoDD"]), ("FullSADD", full_schedule)):
                    rows.extend(
                        _trajectory_rows(
                            case,
                            "profile_awareness",
                            method,
                            schedule,
                            architecture,
                            sim,
                            settings.samples,
                            detuning,
                            fixed_control,
                            heating,
                            profile_name,
                            field_profile=profile,
                        )
                    )
    return rows


def _trajectory_rows(
    case: Case,
    scenario: str,
    method: str,
    schedule_info: tuple[ActionSchedule, frozenset[int]],
    architecture: Architecture,
    sim: Mapping[str, object],
    samples: int,
    detuning: float,
    control: float,
    heating: float,
    profile: str,
    *,
    field_profile: tuple[float, ...] | None = None,
) -> list[RawRow]:
    schedule, local_ids = schedule_info
    reference = build_reference_circuit(schedule, architecture, local_action_ids=local_ids)
    base_seed = _seed(_integer(sim, "seed"), case.name, scenario, detuning, control, heating)
    noise = PaperNoise(
        dt_seconds=_number(sim, "dt_seconds"),
        dephasing_strength=detuning,
        correlation_time_steps=_number(sim, "dephasing_correlation_time_steps"),
        correlation_length_sites=_number(sim, "dephasing_correlation_length_sites"),
        pulse_area_std=control * _number(sim, "pulse_area_std"),
        axis_tilt_std=control * math.radians(_number(sim, "axis_tilt_std_degrees")),
        heating_scale=heating,
        heating_noise_scale=_number(sim, "heating_noise_scale"),
        field_profile=field_profile,
    )
    output: list[RawRow] = []
    for sample in range(samples):
        seed = _seed(base_seed, sample)
        circuit = build_noisy_circuit(schedule, architecture, noise, seed=seed, local_action_ids=local_ids)
        infidelity = simulate_infidelity(
            circuit,
            reference,
            max_bond_dimension=_integer(sim, "max_bond_dimension"),
            svd_threshold=_number(sim, "svd_threshold"),
            seed=seed,
        )
        output.append({
            "scenario": scenario,
            "case": case.name,
            "method": method,
            "sample": sample,
            "seed": seed,
            "detuning": detuning,
            "control": control,
            "heating": heating,
            "profile": profile,
            "state_infidelity": infidelity,
        })
    return output


def _panel_detunings(detunings: tuple[float, ...]) -> tuple[float, ...]:
    desired = tuple(value for value in (0.1, 1.0, 10.0) if value in detunings)
    return desired or detunings


def _seed(base: int, *parts: object) -> int:
    payload = "|".join(map(str, (base, *parts))).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "little")


def _average_proxy(rows: Sequence[RawRow], x_field: str) -> list[RawRow]:
    grouped: dict[float, list[RawRow]] = {}
    for row in rows:
        grouped.setdefault(float(row[x_field]), []).append(row)
    output: list[RawRow] = []
    for x, group in sorted(grouped.items()):
        center = float(np.mean([float(row["mean_R_chi"]) for row in group]))
        first = group[0]
        result = {key: value for key, value in first.items() if key != "case"}
        result["mean_R_chi"] = center
        result["ci95_low"] = center
        result["ci95_high"] = center
        result[x_field] = x
        output.append(result)
    return output


if __name__ == "__main__":
    raise SystemExit(main())
