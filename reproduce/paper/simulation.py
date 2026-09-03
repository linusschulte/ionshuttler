# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Paper-local lowering of Linear schedules to deterministic YAQS simulations."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from qiskit import QuantumCircuit

from mqt.ionshuttler.linear.actions import (
    Action,
    AdvanceTime,
    GlobalPulse,
    PhysicalSwap,
    Rx,
    Rxx,
    Ry,
    Ryy,
    Rz,
    Rzz,
    Shuttle,
)
from mqt.ionshuttler.linear.dd.frame_replay import (
    FramedActionEvent,
    PauliFrame,
    build_frame_history,
    framed_action_events,
)
from mqt.ionshuttler.linear.dd.timeline import build_timeline

if TYPE_CHECKING:
    # The repository lint environment intentionally excludes the paper-only group.
    from mqt.yaqs import State as YaqsState  # ty: ignore[unresolved-import]

    from mqt.ionshuttler.linear import ActionSchedule, Architecture


@dataclass(frozen=True)
class PaperNoise:
    """One point in the paper's sampled dephasing/control/heating model."""

    dt_seconds: float
    dephasing_strength: float
    correlation_time_steps: float
    correlation_length_sites: float
    pulse_area_std: float = 0.0
    axis_tilt_std: float = 0.0
    heating_scale: float = 0.0
    field_profile: tuple[float, ...] | None = None

    def normalized_profile(self, num_sites: int) -> tuple[float, ...]:
        """Return the nonnegative site-sensitivity profile with unit RMS.

        Args:
            num_sites: Number of sites in the simulated architecture.

        Returns:
            One normalized sensitivity per site.

        Raises:
            ValueError: If the configured profile has the wrong size or cannot be normalized.
        """
        if self.field_profile is None:
            return (1.0,) * num_sites
        if len(self.field_profile) != num_sites:
            msg = "field_profile must contain one value per site"
            raise ValueError(msg)
        values = np.asarray(self.field_profile, dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            msg = "field_profile values must be finite and nonnegative"
            raise ValueError(msg)
        rms = float(np.sqrt(np.mean(values**2)))
        if rms <= 0.0:
            msg = "field_profile must have positive RMS"
            raise ValueError(msg)
        return tuple(float(value / rms) for value in values)


@dataclass
class _ControlState:
    ion_motion: dict[int, float] = field(default_factory=dict)

    def add_transport(self, action: Action) -> None:
        ions: tuple[int, ...]
        if isinstance(action, Shuttle):
            ions = (action.ion,)
        elif isinstance(action, PhysicalSwap):
            ions = (action.ion_a, action.ion_b)
        else:
            return
        for ion in ions:
            self.ion_motion[ion] = self.ion_motion.get(ion, 0.0) + 1.0

    def pair_motion(self, q0: int, q1: int) -> float:
        return 0.5 * (self.ion_motion.get(q0, 0.0) + self.ion_motion.get(q1, 0.0))

    def consume_pair_motion(self, q0: int, q1: int) -> None:
        self.ion_motion[q0] = 0.0
        self.ion_motion[q1] = 0.0


@dataclass
class _DephasingState:
    noise: PaperNoise
    num_sites: int
    rng: np.random.Generator
    field: np.ndarray | None = None

    def __post_init__(self) -> None:
        if math.isinf(self.noise.correlation_length_sites):
            self.cholesky = np.zeros((self.num_sites, self.num_sites), dtype=float)
            self.cholesky[:, 0] = 1.0
        else:
            sites = np.arange(self.num_sites, dtype=float)
            distances = np.abs(sites[:, None] - sites[None, :])
            covariance = np.exp(-distances / self.noise.correlation_length_sites)
            self.cholesky = np.linalg.cholesky(covariance + 1e-12 * np.eye(self.num_sites))
        self.profile = self.noise.normalized_profile(self.num_sites)

    def angles(self, positions: tuple[int, ...]) -> tuple[float, ...]:
        innovation = self.noise.dephasing_strength * (self.cholesky @ self.rng.normal(size=self.num_sites))
        if self.field is None:
            self.field = innovation
        else:
            alpha = math.exp(-1.0 / self.noise.correlation_time_steps)
            self.field = alpha * self.field + math.sqrt(max(0.0, 1.0 - alpha**2)) * innovation
        return tuple(float(self.field[site] * self.profile[site] * self.noise.dt_seconds) for site in positions)


def local_pulse_action_ids(report: object) -> frozenset[int]:
    """Extract report-owned local pulse identities from a DD report.

    Returns:
        All action IDs owned by local DD sequences in the report.
    """
    sequences = getattr(report, "sequences", ())
    return frozenset(action_id for sequence in sequences for action_id in sequence.action_ids)


def build_reference_circuit(
    schedule: ActionSchedule,
    architecture: Architecture,
    *,
    local_action_ids: frozenset[int] = frozenset(),
) -> QuantumCircuit:
    """Build the ideal logical circuit, excluding transport and DD pulses.

    Returns:
        The ideal Qiskit circuit.
    """
    circuit = QuantumCircuit(len(schedule.initial_state.positions))
    for event in framed_action_events(schedule, architecture, local_pulse_action_ids=local_action_ids):
        if event.kind == "algorithmic_gate":
            _append_ideal_action(circuit, event.action)
    return circuit


def build_noisy_circuit(
    schedule: ActionSchedule,
    architecture: Architecture,
    noise: PaperNoise,
    *,
    seed: int,
    local_action_ids: frozenset[int] = frozenset(),
    correct_terminal_frame: bool = True,
) -> QuantumCircuit:
    """Lower one schedule and sampled paper-noise realization to Qiskit for YAQS.

    Returns:
        The sampled physical circuit.

    Raises:
        ValueError: If noise settings or a schedule action are unsupported.
    """
    if noise.dt_seconds <= 0.0:
        msg = "dt_seconds must be positive"
        raise ValueError(msg)
    if noise.correlation_time_steps <= 0.0 or noise.correlation_length_sites <= 0.0:
        msg = "noise correlation scales must be positive"
        raise ValueError(msg)
    num_ions = len(schedule.initial_state.positions)
    circuit = QuantumCircuit(num_ions)
    timeline = build_timeline(schedule, architecture)
    events = framed_action_events(schedule, architecture, timeline, local_action_ids)
    events_by_time: dict[int, list[FramedActionEvent]] = {}
    for event in events:
        events_by_time.setdefault(event.timestep, []).append(event)

    master_rng = np.random.default_rng(seed)
    phase_rng = np.random.default_rng(int(master_rng.integers(0, np.iinfo(np.int64).max)))
    phase = _DephasingState(noise, architecture.num_sites, phase_rng)
    control_seed = int(master_rng.integers(0, np.iinfo(np.int64).max))
    control_state = _ControlState()

    for timestep in range(timeline.makespan + 1):
        current_events = events_by_time.get(timestep, [])
        for event in current_events:
            event_rng = _event_rng(control_seed, event)
            if event.kind == "algorithmic_gate":
                _append_framed_action(circuit, event, noise, event_rng, control_state)
            elif event.kind == "local_dd_pulse":
                _append_physical_action(circuit, event.action, noise, event_rng, control_state)
            elif event.kind == "global_dd_pulse":
                _append_global_pulse(circuit, event.action, num_ions, noise, event_rng)
            elif event.kind not in {"transport", "advance_time", "other"}:
                msg = f"unsupported framed action kind: {event.kind!r}"
                raise ValueError(msg)
        if timestep == timeline.makespan:
            continue
        for event in current_events:
            control_state.add_transport(event.action)
        positions = tuple(timeline.ion_position(ion, timestep) for ion in range(num_ions))
        for ion, angle in enumerate(phase.angles(positions)):
            if angle:
                circuit.rz(angle, ion)
        circuit.barrier()

    if correct_terminal_frame:
        history = build_frame_history(timeline, local_action_ids)
        for ion in range(num_ions):
            _append_virtual_frame(circuit, ion, history.frame_for_ion(ion, timeline.makespan))
    return circuit


def simulate_infidelity(
    circuit: QuantumCircuit,
    ideal_circuit: QuantumCircuit,
    *,
    max_bond_dimension: int,
    svd_threshold: float,
    seed: int,
) -> float:
    """Run YAQS and return pure-state infidelity against an ideal circuit.

    Returns:
        State infidelity in ``[0, 1]``.

    """
    ideal = simulate_state(
        ideal_circuit,
        max_bond_dimension=max_bond_dimension,
        svd_threshold=svd_threshold,
        seed=seed,
    )
    actual = simulate_state(
        circuit,
        max_bond_dimension=max_bond_dimension,
        svd_threshold=svd_threshold,
        seed=seed,
    )
    return state_infidelity(actual, ideal)


def simulate_state(
    circuit: QuantumCircuit,
    *,
    max_bond_dimension: int,
    svd_threshold: float,
    seed: int,
) -> YaqsState:
    """Run one pure-state YAQS circuit for the paper harness.

    Returns:
        The final matrix-product state.

    Raises:
        RuntimeError: If YAQS is missing or does not return a final state.
    """
    try:
        # The repository lint environment intentionally excludes the paper-only group.
        from mqt.yaqs import Simulator, State, StrongSimParams  # ty: ignore[unresolved-import]
    except ImportError as error:
        msg = "paper reproduction requires the `paper` dependency group (run `uv sync --group paper`)"
        raise RuntimeError(msg) from error
    params = StrongSimParams(
        get_state=True,
        num_traj=1,
        max_bond_dim=max_bond_dimension,
        svd_threshold=svd_threshold,
        random_seed=seed,
        gate_mode="mpo",
    )
    result = Simulator(parallel=False, show_progress=False).run(
        State(length=circuit.num_qubits, initial="zeros"), circuit, params
    )
    if result.output_state is None:
        msg = "YAQS did not return the requested final state"
        raise RuntimeError(msg)
    return result.output_state


def state_infidelity(actual: YaqsState, ideal: YaqsState) -> float:
    """Return normalized pure-state infidelity between two YAQS states.

    Returns:
        State infidelity in ``[0, 1]``.
    """
    overlap = ideal.mps.scalar_product(actual.mps)
    ideal_norm = float(np.real(ideal.mps.scalar_product(ideal.mps)))
    actual_norm = float(np.real(actual.mps.scalar_product(actual.mps)))
    fidelity = float(np.abs(overlap) ** 2 / (ideal_norm * actual_norm))
    return float(np.clip(1.0 - fidelity, 0.0, 1.0))


def _event_rng(seed: int, event: FramedActionEvent) -> np.random.Generator:
    identity = f"{seed}|{event.timestep}|{event.kind}|{event.action!r}".encode()
    digest = hashlib.blake2b(identity, digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, byteorder="little"))


def _append_framed_action(
    circuit: QuantumCircuit,
    event: FramedActionEvent,
    noise: PaperNoise,
    rng: np.random.Generator,
    state: _ControlState,
) -> None:
    for ion, frame in event.ion_frames:
        _append_virtual_frame(circuit, ion, frame)
    _append_physical_action(circuit, event.action, noise, rng, state)
    for ion, frame in reversed(event.ion_frames):
        _append_virtual_frame(circuit, ion, frame)


def _append_physical_action(
    circuit: QuantumCircuit,
    action: Action,
    noise: PaperNoise,
    rng: np.random.Generator,
    state: _ControlState,
) -> None:
    if isinstance(action, Rz):
        circuit.rz(action.theta, action.ion)
    elif isinstance(action, Rx):
        _append_noisy_1q(circuit, action.ion, "x", action.theta, noise, rng)
    elif isinstance(action, Ry):
        _append_noisy_1q(circuit, action.ion, "y", action.theta, noise, rng)
    elif isinstance(action, Rxx):
        _append_noisy_2q(circuit, action.ion_a, action.ion_b, "rxx", action.theta, noise, rng, state)
    elif isinstance(action, Ryy):
        _append_noisy_2q(circuit, action.ion_a, action.ion_b, "ryy", action.theta, noise, rng, state)
    elif isinstance(action, Rzz):
        gate = "rzz"
        _append_noisy_2q(circuit, action.ion_a, action.ion_b, gate, action.theta, noise, rng, state)
    else:
        msg = f"unsupported physical gate action: {action!r}"
        raise TypeError(msg)


def _append_noisy_1q(
    circuit: QuantumCircuit,
    qubit: int,
    axis: str,
    angle: float,
    noise: PaperNoise,
    rng: np.random.Generator,
) -> None:
    theta = angle * (1.0 + float(rng.normal(0.0, noise.pulse_area_std)))
    base = {"x": np.array((1.0, 0.0, 0.0)), "y": np.array((0.0, 1.0, 0.0))}[axis]
    tilt = np.array(base, copy=True)
    principal_index = {"x": 0, "y": 1}[axis]
    for index in range(3):
        if index != principal_index:
            tilt[index] += float(rng.normal(0.0, noise.axis_tilt_std))
    tilt /= np.linalg.norm(tilt)
    _append_axis_rotation(circuit, qubit, theta, tilt)


def _append_axis_rotation(circuit: QuantumCircuit, qubit: int, theta: float, axis: np.ndarray) -> None:
    x, y, z = (float(value) for value in axis)
    alpha = math.acos(float(np.clip(z, -1.0, 1.0)))
    beta = math.atan2(y, x)
    circuit.rz(-beta, qubit)
    circuit.ry(-alpha, qubit)
    circuit.rz(theta, qubit)
    circuit.ry(alpha, qubit)
    circuit.rz(beta, qubit)


def _append_noisy_2q(
    circuit: QuantumCircuit,
    q0: int,
    q1: int,
    gate: str,
    angle: float,
    noise: PaperNoise,
    rng: np.random.Generator,
    state: _ControlState,
) -> None:
    heating_sigma = noise.heating_scale * state.pair_motion(q0, q1)
    theta = angle * (1.0 + float(rng.normal(0.0, noise.pulse_area_std)))
    theta += float(rng.normal(0.0, heating_sigma))
    coefficients = _sample_tilted_2q_generator(gate, noise.axis_tilt_std, rng)
    for supported_gate, coefficient in zip(("rxx", "ryy", "rzz"), coefficients, strict=True):
        _append_2q_rotation(circuit, supported_gate, theta * float(coefficient), q0, q1)
    state.consume_pair_motion(q0, q1)


def _sample_tilted_2q_generator(gate: str, sigma: float, rng: np.random.Generator) -> np.ndarray:
    names = ("rxx", "ryy", "rzz")
    if gate not in names:
        msg = f"unsupported two-qubit rotation: {gate!r}"
        raise ValueError(msg)
    vector = np.zeros(3, dtype=float)
    principal = names.index(gate)
    vector[principal] = 1.0
    for index in range(3):
        if index != principal:
            vector[index] += float(rng.normal(0.0, sigma))
    return vector / np.linalg.norm(vector)


def _append_2q_rotation(circuit: QuantumCircuit, gate: str, theta: float, q0: int, q1: int) -> None:
    if not theta:
        return
    if gate == "rxx":
        circuit.rxx(theta, q0, q1)
    elif gate == "ryy":
        circuit.ryy(theta, q0, q1)
    elif gate == "rzz":
        circuit.rzz(theta, q0, q1)
    else:
        msg = f"unsupported two-qubit rotation: {gate!r}"
        raise ValueError(msg)


def _append_global_pulse(
    circuit: QuantumCircuit,
    action: Action,
    num_ions: int,
    noise: PaperNoise,
    rng: np.random.Generator,
) -> None:
    if not isinstance(action, GlobalPulse) or action.gate.theta is None:
        msg = f"invalid global pulse: {action!r}"
        raise ValueError(msg)
    axis = {"Rx": "x", "Ry": "y"}.get(action.gate.gate_name)
    if axis is None:
        msg = f"unsupported global pulse: {action.gate.gate_name}"
        raise ValueError(msg)
    for ion in range(num_ions):
        _append_noisy_1q(circuit, ion, axis, action.gate.theta, noise, rng)


def _append_ideal_action(circuit: QuantumCircuit, action: Action) -> None:
    if isinstance(action, Rx):
        circuit.rx(action.theta, action.ion)
    elif isinstance(action, Ry):
        circuit.ry(action.theta, action.ion)
    elif isinstance(action, Rz):
        circuit.rz(action.theta, action.ion)
    elif isinstance(action, Rxx):
        circuit.rxx(action.theta, action.ion_a, action.ion_b)
    elif isinstance(action, Ryy):
        circuit.ryy(action.theta, action.ion_a, action.ion_b)
    elif isinstance(action, Rzz):
        circuit.rzz(action.theta, action.ion_a, action.ion_b)
    elif not isinstance(action, AdvanceTime):
        msg = f"unsupported algorithmic action: {action!r}"
        raise TypeError(msg)


def _append_virtual_frame(circuit: QuantumCircuit, ion: int, frame: PauliFrame) -> None:
    if frame.label == "X":
        circuit.x(ion)
    elif frame.label == "Y":
        circuit.y(ion)
    elif frame.label == "Z":
        circuit.z(ion)
