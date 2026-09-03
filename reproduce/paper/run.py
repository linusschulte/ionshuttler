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
import json
import math
import sys
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from qiskit import QuantumCircuit

from mqt.ionshuttler.linear import (
    ActionSchedule,
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
    read_csv,
    schedule_chi,
    summarize_across_cases,
    summarize_infidelities,
    write_csv,
)
from reproduce.paper.checkpoint import CheckpointStore
from reproduce.paper.plots import render_all
from reproduce.paper.simulation import (
    PaperNoise,
    build_noisy_circuit,
    build_reference_circuit,
    local_pulse_action_ids,
    simulate_state,
    state_infidelity,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mqt.yaqs import State as YaqsState  # ty: ignore[unresolved-import]

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("paper.toml")
INPUT_DIR = Path(__file__).with_name("circuits")
FROZEN_SCHEDULES = INPUT_DIR / "compiled_schedules.json"
METHODS = ("NoDD", "IdealizedHahn", "NearestHahn", "PulseOnlySADD", "FullSADD")
ROUTING_METHODS = ("NoDD", "NearestHahn", "PulseOnlySADD", "FullSADD")
TABLE_METHODS = ("NoDD", "PulseOnlySADD", "FullSADD")


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


@dataclass(frozen=True)
class CompiledCase:
    """Schedules and static analysis retained across sample-major execution."""

    case: Case
    schedules: Mapping[str, tuple[ActionSchedule, frozenset[int]]]
    aware_full: tuple[ActionSchedule, frozenset[int]]
    reference: QuantumCircuit


@dataclass(frozen=True)
class _WorkerContext:
    """Read-only state inherited by paper simulation worker processes."""

    compiled_cases: Sequence[CompiledCase]
    architecture: Architecture
    profile_architecture: Architecture
    simulation: Mapping[str, object]
    sweeps: Mapping[str, object]
    settings: RunSettings


_WORKER_CONTEXT: _WorkerContext | None = None


def main(argv: Sequence[str] | None = None) -> int:
    """Run the reproduction command.

    Returns:
        A successful process status.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("quick", "paper"))
    parser.add_argument("--output", type=Path, required=True, help="fresh or replaceable output directory")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--samples", type=int, help="target deterministic sample prefix (may extend a checkpoint)")
    parser.add_argument("--workers", type=int, help="circuit simulation worker processes")
    parser.add_argument("--plots-only", action="store_true", help="redraw PDFs from existing output CSV files")
    parser.add_argument(
        "--regenerate-schedules",
        action="store_true",
        help="manually recompile and overwrite the frozen paper schedule bundle",
    )
    args = parser.parse_args(argv)
    config = _load_config(args.config)
    settings = _settings(config, args.mode)
    if args.samples is not None:
        if args.samples <= 0:
            parser.error("--samples must be positive")
        settings = replace(settings, samples=args.samples)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.plots_only and args.regenerate_schedules:
        parser.error("--plots-only and --regenerate-schedules are mutually exclusive")
    if args.plots_only:
        _rerender_existing(output)
        return 0

    architecture, profile_architecture = _architectures(_table(config, "architecture"))
    simulation_config = _table(config, "simulation")
    sweep_config = _table(config, "sweeps")
    workers = _integer(simulation_config, "workers") if args.workers is None else args.workers
    if workers <= 0:
        parser.error("--workers must be positive")
    if args.regenerate_schedules and args.mode != "paper":
        parser.error("--regenerate-schedules requires paper mode")

    checkpoint = CheckpointStore(
        output,
        _run_identity(config, settings),
        _expected_rows_per_sample(settings),
    )
    checkpoint.validate_identity()
    if args.regenerate_schedules and checkpoint.exists:
        parser.error("--regenerate-schedules requires an output directory without checkpoints")
    if checkpoint.exists:
        print("restoring exact compiled schedules from checkpoint", flush=True)
        compiled_cases, schedule_rows, opportunity_rows, proxy_temporal, proxy_spatial = _restore_compiled_cases(
            checkpoint.load_compiled(), architecture
        )
    else:
        if args.regenerate_schedules:
            compiled = _compile_cases(
                settings,
                architecture,
                profile_architecture,
                _compiler(architecture, _table(config, "compiler")),
                _dd_config(_table(config, "dd")),
                simulation_config,
            )
            compiled_cases, schedule_rows, opportunity_rows, proxy_temporal, proxy_spatial, fingerprints = compiled
            _write_frozen_schedules(
                _compiled_payload(compiled_cases, schedule_rows, opportunity_rows, proxy_temporal, proxy_spatial),
                config,
            )
            print(f"overwrote frozen schedules at {FROZEN_SCHEDULES}", flush=True)
        else:
            compiled_cases, schedule_rows, opportunity_rows, proxy_temporal, proxy_spatial = _load_frozen_schedules(
                config, settings, architecture
            )
            fingerprints = {
                compiled.case.name: _schedule_fingerprint(compiled.schedules, compiled.aware_full)
                for compiled in compiled_cases
            }
            print(f"loaded frozen schedules from {FROZEN_SCHEDULES}", flush=True)
        checkpoint.prepare(
            fingerprints,
            _compiled_payload(compiled_cases, schedule_rows, opportunity_rows, proxy_temporal, proxy_spatial),
        )
    temporal = _average_proxy(proxy_temporal, "tau_steps")
    spatial = _average_proxy(proxy_spatial, "ell_sites")
    completed = checkpoint.completed_samples()
    table_rows_per_sample = len(settings.cases) * len(TABLE_METHODS)
    table_completed = checkpoint.completed_table_samples(table_rows_per_sample)
    global _WORKER_CONTEXT  # ruff: ignore[global-statement] - Forked one-off research workers inherit immutable run state.
    _WORKER_CONTEXT = _WorkerContext(
        compiled_cases,
        architecture,
        profile_architecture,
        simulation_config,
        sweep_config,
        settings,
    )
    executor = (
        None
        if workers == 1
        else ProcessPoolExecutor(max_workers=min(workers, len(compiled_cases)), mp_context=get_context("fork"))
    )
    preview_samples = 0
    try:
        for sample in range(settings.samples):
            run_main = sample not in completed
            run_table = sample not in table_completed
            if not run_main and not run_table:
                print(f"[sample {sample + 1}/{settings.samples}] checkpoint complete; skipping", flush=True)
                continue
            sample_rows, table_rows = _run_case_tasks(executor, sample, run_main=run_main, run_table=run_table)
            if run_main:
                shard = checkpoint.commit(sample, sample_rows)
                completed = completed.union((sample,))
                print(f"committed {shard}", flush=True)
            if run_table:
                table_shard = checkpoint.commit_table(sample, table_rows, table_rows_per_sample)
                table_completed = table_completed.union((sample,))
                print(f"committed {table_shard}", flush=True)
            _write_outputs(
                [*checkpoint.load_prefix(sample + 1), *checkpoint.load_table_prefix(sample + 1)],
                schedule_rows,
                opportunity_rows,
                temporal,
                spatial,
                sweep_config,
                simulation_config,
                replace(settings, samples=sample + 1),
                output,
            )
            preview_samples = sample + 1
        if preview_samples != settings.samples:
            _write_outputs(
                [
                    *checkpoint.load_prefix(settings.samples),
                    *checkpoint.load_table_prefix(settings.samples),
                ],
                schedule_rows,
                opportunity_rows,
                temporal,
                spatial,
                sweep_config,
                simulation_config,
                settings,
                output,
            )
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)
        _WORKER_CONTEXT = None
    return 0


def _rerender_existing(output: Path) -> None:
    csv_dir = output / "csv"

    def rows(name: str) -> list[RawRow]:
        return cast("list[RawRow]", read_csv(csv_dir / name))

    figures = render_all(
        rows("aggregate_metrics.csv"),
        rows("per_case_metrics.csv"),
        rows("temporal_applicability.csv"),
        rows("spatial_applicability.csv"),
        rows("schedule_summary.csv"),
        rows("runtime_opportunities.csv"),
        rows("objective_fidelity.csv"),
        output / "figures",
    )
    print("regenerated figures:", flush=True)
    for path in figures:
        print(f"  {path}", flush=True)


def _run_case_tasks(
    executor: ProcessPoolExecutor | None,
    sample: int,
    *,
    run_main: bool,
    run_table: bool,
) -> tuple[list[RawRow], list[RawRow]]:
    context = _require_worker_context()
    results: dict[int, tuple[list[RawRow], list[RawRow]]] = {}
    if executor is None:
        completed = (
            _simulate_case_worker(index, sample, run_main=run_main, run_table=run_table)
            for index in range(len(context.compiled_cases))
        )
    else:
        futures = {
            executor.submit(_simulate_case_worker, index, sample, run_main=run_main, run_table=run_table): index
            for index in range(len(context.compiled_cases))
        }
        completed = (future.result() for future in as_completed(futures))
    for completed_count, (case_index, main_rows, table_rows) in enumerate(completed, start=1):
        results[case_index] = (main_rows, table_rows)
        case = context.compiled_cases[case_index].case
        print(
            f"[sample {sample + 1}/{context.settings.samples}]"
            f"[{completed_count}/{len(context.compiled_cases)}] completed {case.name}",
            flush=True,
        )
    ordered_main: list[RawRow] = []
    ordered_table: list[RawRow] = []
    for case_index in range(len(context.compiled_cases)):
        main_rows, table_rows = results[case_index]
        ordered_main.extend(main_rows)
        ordered_table.extend(table_rows)
    return ordered_main, ordered_table


def _simulate_case_worker(
    case_index: int,
    sample: int,
    *,
    run_main: bool,
    run_table: bool,
) -> tuple[int, list[RawRow], list[RawRow]]:
    context = _require_worker_context()
    compiled = context.compiled_cases[case_index]
    ideal = simulate_state(
        compiled.reference,
        max_bond_dimension=_integer(context.simulation, "max_bond_dimension"),
        svd_threshold=_number(context.simulation, "svd_threshold"),
        seed=_seed(_integer(context.simulation, "seed"), compiled.case.name, "ideal"),
    )
    main_rows = (
        _simulate_case_sample(
            compiled,
            sample,
            ideal,
            context.architecture,
            context.profile_architecture,
            context.simulation,
            context.sweeps,
            context.settings,
        )
        if run_main
        else []
    )
    table_rows = (
        _simulate_table_ii_case_sample(
            compiled,
            sample,
            ideal,
            context.architecture,
            context.simulation,
            context.sweeps,
        )
        if run_table
        else []
    )
    return case_index, main_rows, table_rows


def _require_worker_context() -> _WorkerContext:
    if _WORKER_CONTEXT is None:
        msg = "paper simulation worker context is not initialized"
        raise RuntimeError(msg)
    return _WORKER_CONTEXT


def _compile_cases(
    settings: RunSettings,
    architecture: Architecture,
    profile_architecture: Architecture,
    compiler: LinearCompiler,
    dd_config: SADDConfig,
    simulation: Mapping[str, object],
) -> tuple[list[CompiledCase], list[RawRow], list[RawRow], list[RawRow], list[RawRow], dict[str, str]]:
    compiled_cases: list[CompiledCase] = []
    schedule_rows: list[RawRow] = []
    opportunity_rows: list[RawRow] = []
    proxy_temporal: list[RawRow] = []
    proxy_spatial: list[RawRow] = []
    fingerprints: dict[str, str] = {}
    for case_index, case in enumerate(settings.cases, start=1):
        print(f"[{case_index}/{len(settings.cases)}] compiling {case.name}", flush=True)
        result = compiler.compile(_make_circuit(case))
        if result.status is not CompilationStatus.SUCCESS:
            msg = f"compilation of {case.name} ended with {result.status.value}"
            raise RuntimeError(msg)
        schedules, audits, case_opportunities = _schedule_variants(result.schedule, architecture, dd_config)
        aware_full = _full_sadd(result.schedule, profile_architecture, dd_config)
        compiled_cases.append(_compiled_case(case, schedules, aware_full, architecture))
        fingerprints[case.name] = _schedule_fingerprint(schedules, aware_full)
        schedule_rows.extend(
            _schedule_audit_rows(
                case,
                schedules,
                audits,
                architecture,
                dt_seconds=_number(simulation, "dt_seconds"),
                temporal_corr=_number(simulation, "dephasing_correlation_time_steps"),
            )
        )
        opportunity_rows.extend({"case": case.name, "family": case.family, **row} for row in case_opportunities)
        temporal, spatial = covariance_proxy_rows(
            schedules,
            architecture,
            temporal_scales=settings.temporal_scales,
            spatial_scales=settings.spatial_scales,
            dt_seconds=_number(simulation, "dt_seconds"),
        )
        proxy_temporal.extend({"case": case.name, **row} for row in temporal)
        proxy_spatial.extend({"case": case.name, **row} for row in spatial)
    return compiled_cases, schedule_rows, opportunity_rows, proxy_temporal, proxy_spatial, fingerprints


def _compiled_case(
    case: Case,
    schedules: Mapping[str, tuple[ActionSchedule, frozenset[int]]],
    aware_full: tuple[ActionSchedule, frozenset[int]],
    architecture: Architecture,
) -> CompiledCase:
    reference = build_reference_circuit(
        schedules["NoDD"][0],
        architecture,
        local_action_ids=schedules["NoDD"][1],
    )
    return CompiledCase(case, schedules, aware_full, reference)


def _compiled_payload(
    compiled_cases: Sequence[CompiledCase],
    schedule_rows: Sequence[RawRow],
    opportunity_rows: Sequence[RawRow],
    proxy_temporal: Sequence[RawRow],
    proxy_spatial: Sequence[RawRow],
) -> dict[str, object]:
    cases: list[dict[str, object]] = []
    for compiled in compiled_cases:
        schedules = {
            method: {"schedule": schedule.to_dict(), "local_action_ids": sorted(local_ids)}
            for method, (schedule, local_ids) in compiled.schedules.items()
        }
        cases.append({
            "case": {
                "family": compiled.case.family,
                "qubits": compiled.case.qubits,
                "seed": compiled.case.seed,
            },
            "schedules": schedules,
            "aware_full": {
                "schedule": compiled.aware_full[0].to_dict(),
                "local_action_ids": sorted(compiled.aware_full[1]),
            },
        })
    return {
        "cases": cases,
        "schedule_rows": list(schedule_rows),
        "opportunity_rows": list(opportunity_rows),
        "proxy_temporal": list(proxy_temporal),
        "proxy_spatial": list(proxy_spatial),
    }


def _write_frozen_schedules(payload: Mapping[str, object], config: Mapping[str, object]) -> None:
    envelope = {
        "schema_version": 1,
        "schedule_inputs_sha256": _schedule_inputs_sha256(config),
        "payload": payload,
    }
    temporary = FROZEN_SCHEDULES.with_name(f".{FROZEN_SCHEDULES.name}.tmp")
    temporary.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(FROZEN_SCHEDULES)


def _load_frozen_schedules(
    config: Mapping[str, object], settings: RunSettings, architecture: Architecture
) -> tuple[list[CompiledCase], list[RawRow], list[RawRow], list[RawRow], list[RawRow]]:
    if not FROZEN_SCHEDULES.is_file():
        msg = f"missing frozen schedules: {FROZEN_SCHEDULES}; run paper mode with --regenerate-schedules"
        raise FileNotFoundError(msg)
    value = json.loads(FROZEN_SCHEDULES.read_text(encoding="utf-8"))
    envelope = _checkpoint_mapping(value, "frozen schedule bundle")
    if envelope.get("schema_version") != 1 or envelope.get("schedule_inputs_sha256") != _schedule_inputs_sha256(config):
        msg = "frozen schedules do not match the configured compiler/DD/circuit inputs; regenerate them explicitly"
        raise RuntimeError(msg)
    payload = _checkpoint_mapping(envelope.get("payload"), "frozen schedule payload")
    compiled, schedules, opportunities, _temporal, _spatial = _restore_compiled_cases(payload, architecture)
    selected = {case.name for case in settings.cases}
    selected_compiled = [case for case in compiled if case.case.name in selected]
    simulation = _table(config, "simulation")
    temporal: list[RawRow] = []
    spatial: list[RawRow] = []
    for case in selected_compiled:
        case_temporal, case_spatial = covariance_proxy_rows(
            case.schedules,
            architecture,
            temporal_scales=settings.temporal_scales,
            spatial_scales=settings.spatial_scales,
            dt_seconds=_number(simulation, "dt_seconds"),
        )
        temporal.extend({"case": case.case.name, **row} for row in case_temporal)
        spatial.extend({"case": case.case.name, **row} for row in case_spatial)
    return (
        selected_compiled,
        [row for row in schedules if str(row["case"]) in selected],
        [row for row in opportunities if str(row["case"]) in selected],
        temporal,
        spatial,
    )


def _schedule_inputs_sha256(config: Mapping[str, object]) -> str:
    simulation = _table(config, "simulation")
    payload = {
        "schema_version": 1,
        "cases": _case_tables(config),
        "architecture": _table(config, "architecture"),
        "compiler": _table(config, "compiler"),
        "dd": _table(config, "dd"),
        "proxy_settings": {
            "dt_seconds": simulation["dt_seconds"],
            "dephasing_correlation_time_steps": simulation["dephasing_correlation_time_steps"],
        },
        "circuit_inputs_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(INPUT_DIR.glob("*.qasm"))
        },
    }
    linear = hashlib.sha256()
    for path in sorted((ROOT / "src" / "mqt" / "ionshuttler" / "linear").rglob("*.py")):
        linear.update(str(path.relative_to(ROOT)).encode())
        linear.update(path.read_bytes())
    payload["linear_implementation_sha256"] = linear.hexdigest()
    return _json_sha256(payload)


def _restore_compiled_cases(
    payload: Mapping[str, object], architecture: Architecture
) -> tuple[list[CompiledCase], list[RawRow], list[RawRow], list[RawRow], list[RawRow]]:
    raw_cases = _checkpoint_list(payload, "cases")
    compiled_cases: list[CompiledCase] = []
    for raw_case in raw_cases:
        entry = _checkpoint_mapping(raw_case, "each compiled case")
        case_data = _checkpoint_mapping(entry.get("case"), "compiled case metadata")
        family = case_data.get("family")
        qubits = case_data.get("qubits")
        seed = case_data.get("seed")
        if not isinstance(family, str) or not isinstance(qubits, int) or not isinstance(seed, int):
            msg = "compiled case metadata is malformed"
            raise TypeError(msg)
        case = Case(family, qubits, seed)
        raw_schedules = _checkpoint_mapping(entry.get("schedules"), "compiled schedules")
        schedules = {
            method: _restore_schedule(_checkpoint_mapping(value, f"compiled {method} schedule"))
            for method, value in raw_schedules.items()
        }
        aware_full = _restore_schedule(_checkpoint_mapping(entry.get("aware_full"), "profile-aware schedule"))
        compiled_cases.append(_compiled_case(case, schedules, aware_full, architecture))
    return (
        compiled_cases,
        _checkpoint_rows(payload, "schedule_rows"),
        _checkpoint_rows(payload, "opportunity_rows"),
        _checkpoint_rows(payload, "proxy_temporal"),
        _checkpoint_rows(payload, "proxy_spatial"),
    )


def _restore_schedule(value: Mapping[str, object]) -> tuple[ActionSchedule, frozenset[int]]:
    raw_ids = value.get("local_action_ids")
    if not isinstance(raw_ids, list) or any(not isinstance(action_id, int) for action_id in raw_ids):
        msg = "compiled local_action_ids must be an integer array"
        raise TypeError(msg)
    try:
        schedule = ActionSchedule.from_dict(value.get("schedule"))
    except (TypeError, ValueError) as error:
        msg = "cannot restore compiled action schedule"
        raise RuntimeError(msg) from error
    return schedule, frozenset(raw_ids)


def _checkpoint_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        msg = f"{label} must be an object"
        raise TypeError(msg)
    return cast("dict[str, object]", value)


def _checkpoint_list(payload: Mapping[str, object], name: str) -> list[object]:
    value = payload.get(name)
    if not isinstance(value, list):
        msg = f"compiled checkpoint field {name!r} must be an array"
        raise TypeError(msg)
    return value


def _checkpoint_rows(payload: Mapping[str, object], name: str) -> list[RawRow]:
    rows = _checkpoint_list(payload, name)
    if any(not isinstance(row, dict) for row in rows):
        msg = f"compiled checkpoint field {name!r} must contain objects"
        raise TypeError(msg)
    return cast("list[RawRow]", rows)


def _run_identity(config: Mapping[str, object], settings: RunSettings) -> dict[str, object]:
    simulation = dict(_table(config, "simulation"))
    simulation.pop("samples", None)
    simulation.pop("bootstrap_samples", None)
    simulation.pop("workers", None)
    scientific_inputs = {
        "mode": settings.mode,
        "cases": [{"family": case.family, "qubits": case.qubits, "seed": case.seed} for case in settings.cases],
        "detunings": settings.detunings,
        "controls": settings.controls,
        "heating": settings.heating,
        "profile_heating": settings.profile_heating,
        "temporal_scales": settings.temporal_scales,
        "spatial_scales": settings.spatial_scales,
        "architecture": _table(config, "architecture"),
        "compiler": _table(config, "compiler"),
        "dd": _table(config, "dd"),
        "simulation": simulation,
        "sweeps": _table(config, "sweeps"),
        "circuit_inputs_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(INPUT_DIR.glob("*.qasm"))
        },
    }
    implementation = hashlib.sha256()
    paper_paths = [
        Path(__file__).with_name(name)
        for name in ("analysis.py", "checkpoint.py", "plots.py", "run.py", "simulation.py")
    ]
    linear_paths = sorted((ROOT / "src" / "mqt" / "ionshuttler" / "linear").rglob("*.py"))
    for path in (*paper_paths, ROOT / "uv.lock", *linear_paths):
        implementation.update(str(path.relative_to(ROOT)).encode())
        implementation.update(path.read_bytes())
    return {
        "mode": settings.mode,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "scientific_inputs_sha256": _json_sha256(scientific_inputs),
        "implementation_sha256": implementation.hexdigest(),
    }


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _schedule_fingerprint(
    schedules: Mapping[str, tuple[ActionSchedule, frozenset[int]]],
    aware_full: tuple[ActionSchedule, frozenset[int]],
) -> str:
    payload = {
        method: {"schedule": schedule.to_dict(), "local_action_ids": sorted(local_ids)}
        for method, (schedule, local_ids) in schedules.items()
    }
    payload["ProfileAwareFullSADD"] = {
        "schedule": aware_full[0].to_dict(),
        "local_action_ids": sorted(aware_full[1]),
    }
    return _json_sha256(payload)


def _expected_rows_per_sample(settings: RunSettings) -> int:
    routing_detunings = _panel_detunings(settings.detunings)
    per_case = (
        len(settings.detunings) * len(settings.controls) * len(METHODS)
        + len(routing_detunings) * len(settings.heating) * len(ROUTING_METHODS)
        + len(routing_detunings) * len(settings.profile_heating) * 4
    )
    return len(settings.cases) * per_case


def _write_outputs(
    raw_rows: Sequence[RawRow],
    schedule_rows: Sequence[RawRow],
    opportunity_rows: Sequence[RawRow],
    temporal: Sequence[RawRow],
    spatial: Sequence[RawRow],
    sweeps: Mapping[str, object],
    simulation: Mapping[str, object],
    settings: RunSettings,
    output: Path,
) -> None:
    per_case = summarize_infidelities(raw_rows)
    aggregate = summarize_across_cases(
        per_case,
        bootstrap_samples=settings.bootstrap_samples,
        seed=_integer(simulation, "seed"),
    )
    table_ii = _table_ii_rows(per_case, schedule_rows, sweeps)
    objective = _objective_fidelity_rows(
        per_case,
        schedule_rows,
        sweeps,
        settings,
        seed=_integer(simulation, "seed"),
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
    print(f"rendering preview from {settings.samples} completed sample(s)", flush=True)
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


def _load_config(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        return cast("dict[str, object]", tomllib.load(handle))


def _table(config: Mapping[str, object], name: str) -> dict[str, object]:
    value = config.get(name)
    if not isinstance(value, dict):
        msg = f"configuration requires a [{name}] table"
        raise TypeError(msg)
    return cast("dict[str, object]", value)


def _case_tables(config: Mapping[str, object]) -> list[dict[str, object]]:
    value = config.get("cases")
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        msg = "configuration requires [[cases]] tables"
        raise TypeError(msg)
    return cast("list[dict[str, object]]", value)


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
    stem = {
        "ising": "ising",
        "qft": "qft_nativegates_quantinuum_qiskit_opt2",
        "random": "randomcircuit_nativegates_quantinuum_qiskit_opt2_seed0",
        "qpe": "qpeexact_nativegates_quantinuum_qiskit_opt2",
        "ghz": "ghz_nativegates_quantinuum_qiskit_opt2",
    }.get(case.family)
    if stem is None:
        msg = f"unsupported paper circuit family: {case.family!r}"
        raise ValueError(msg)
    path = INPUT_DIR / f"{stem}_{case.qubits}.qasm"
    if not path.is_file():
        msg = f"missing frozen paper circuit: {path}"
        raise FileNotFoundError(msg)
    circuit = QuantumCircuit.from_qasm_file(path)
    circuit.name = case.name
    return circuit


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
) -> list[RawRow]:
    detuning = _number(sweeps, "profile_detuning")
    heating = _number(sweeps, "selected_heating")
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
        if row["scenario"] == "table_ii" and row["profile"] == "flat"
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


def _simulate_case_sample(
    compiled: CompiledCase,
    sample: int,
    ideal_state: YaqsState,
    architecture: Architecture,
    profile_architecture: Architecture,
    sim: Mapping[str, object],
    sweeps: Mapping[str, object],
    settings: RunSettings,
) -> list[RawRow]:
    rows: list[RawRow] = []
    for detuning in settings.detunings:
        for control in settings.controls:
            rows.extend(
                _trajectory_row(
                    compiled.case,
                    "operating_envelope",
                    method,
                    compiled.schedules[method],
                    ideal_state,
                    architecture,
                    sim,
                    sample,
                    detuning,
                    control,
                    0.0,
                    "flat",
                )
                for method in METHODS
            )
    routing_detunings = _panel_detunings(settings.detunings)
    fixed_control = _number(sweeps, "fixed_control")
    for detuning in routing_detunings:
        for heating in settings.heating:
            rows.extend(
                _trajectory_row(
                    compiled.case,
                    "control_heating",
                    method,
                    compiled.schedules[method],
                    ideal_state,
                    architecture,
                    sim,
                    sample,
                    detuning,
                    fixed_control,
                    heating,
                    "flat",
                )
                for method in ROUTING_METHODS
            )
    profile = tuple(float(profile_architecture.field_at(site)) for site in range(architecture.num_sites))
    for detuning in routing_detunings:
        for heating in settings.profile_heating:
            for profile_name, full_schedule in (
                ("aware", compiled.aware_full),
                ("agnostic", compiled.schedules["FullSADD"]),
            ):
                for method, schedule in (
                    ("NoDD", compiled.schedules["NoDD"]),
                    ("FullSADD", full_schedule),
                ):
                    rows.append(
                        _trajectory_row(
                            compiled.case,
                            "profile_awareness",
                            method,
                            schedule,
                            ideal_state,
                            architecture,
                            sim,
                            sample,
                            detuning,
                            fixed_control,
                            heating,
                            profile_name,
                            field_profile=profile,
                        )
                    )
    return rows


def _simulate_table_ii_case_sample(
    compiled: CompiledCase,
    sample: int,
    ideal_state: YaqsState,
    architecture: Architecture,
    sim: Mapping[str, object],
    sweeps: Mapping[str, object],
) -> list[RawRow]:
    detuning = _number(sweeps, "profile_detuning")
    control = _number(sweeps, "fixed_control")
    heating = _number(sweeps, "selected_heating")
    return [
        _trajectory_row(
            compiled.case,
            "table_ii",
            method,
            compiled.schedules[method],
            ideal_state,
            architecture,
            sim,
            sample,
            detuning,
            control,
            heating,
            "flat",
        )
        for method in TABLE_METHODS
    ]


def _trajectory_row(
    case: Case,
    scenario: str,
    method: str,
    schedule_info: tuple[ActionSchedule, frozenset[int]],
    ideal_state: YaqsState,
    architecture: Architecture,
    sim: Mapping[str, object],
    sample: int,
    detuning: float,
    control: float,
    heating: float,
    profile: str,
    *,
    field_profile: tuple[float, ...] | None = None,
) -> RawRow:
    schedule, local_ids = schedule_info
    base_seed = _seed(_integer(sim, "seed"), case.name, scenario, detuning, control, heating)
    noise = PaperNoise(
        dt_seconds=_number(sim, "dt_seconds"),
        dephasing_strength=detuning,
        correlation_time_steps=_number(sim, "dephasing_correlation_time_steps"),
        correlation_length_sites=_number(sim, "dephasing_correlation_length_sites"),
        pulse_area_std=control * _number(sim, "pulse_area_std"),
        axis_tilt_std=control * math.radians(_number(sim, "axis_tilt_std_degrees")),
        heating_scale=heating,
        field_profile=field_profile,
    )
    seed = _seed(base_seed, sample)
    circuit = build_noisy_circuit(schedule, architecture, noise, seed=seed, local_action_ids=local_ids)
    actual_state = simulate_state(
        circuit,
        max_bond_dimension=_integer(sim, "max_bond_dimension"),
        svd_threshold=_number(sim, "svd_threshold"),
        seed=seed,
    )
    infidelity = state_infidelity(actual_state, ideal_state)
    return {
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
    }


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
