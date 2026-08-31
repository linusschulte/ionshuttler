# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Numerical sanity checks for the private paper simulation adapter."""

from __future__ import annotations

from math import pi

from mqt.ionshuttler.linear.actions import AdvanceTime, Rx
from mqt.ionshuttler.linear.architecture import Architecture
from mqt.ionshuttler.linear.dd import apply_idealized_hahn
from mqt.ionshuttler.linear.schedule import ActionSchedule
from mqt.ionshuttler.linear.state import create_initial_state
from reproduce.paper.simulation import (
    PaperNoise,
    build_noisy_circuit,
    build_reference_circuit,
    local_pulse_action_ids,
    simulate_infidelity,
)


def test_terminal_frame_correction_and_sampled_dephasing_are_deterministic() -> None:
    architecture = Architecture(num_sites=1, processing_zones={"pz": [0]})
    base = ActionSchedule.from_actions(
        [Rx(ion=0, theta=pi / 2), AdvanceTime(), AdvanceTime(), AdvanceTime(), AdvanceTime()],
        create_initial_state(1, architecture, initial_positions=[0]),
    )
    result = apply_idealized_hahn(base, architecture)
    local_ids = local_pulse_action_ids(result.report)
    reference = build_reference_circuit(base, architecture)
    augmented_reference = build_reference_circuit(
        result.schedule,
        architecture,
        local_action_ids=local_ids,
    )
    noiseless = PaperNoise(
        dt_seconds=0.1,
        dephasing_strength=0.0,
        correlation_time_steps=10.0,
        correlation_length_sites=1.0,
    )
    corrected = build_noisy_circuit(
        result.schedule,
        architecture,
        noiseless,
        seed=7,
        local_action_ids=local_ids,
    )
    uncorrected = build_noisy_circuit(
        result.schedule,
        architecture,
        noiseless,
        seed=7,
        local_action_ids=local_ids,
        correct_terminal_frame=False,
    )
    corrected_error = simulate_infidelity(corrected, reference, max_bond_dimension=8, svd_threshold=1e-12, seed=7)
    reference_error = simulate_infidelity(
        augmented_reference,
        reference,
        max_bond_dimension=8,
        svd_threshold=1e-12,
        seed=7,
    )
    uncorrected_error = simulate_infidelity(uncorrected, reference, max_bond_dimension=8, svd_threshold=1e-12, seed=7)
    assert corrected_error < 1e-12
    assert reference_error < 1e-12
    assert uncorrected_error > 1.0 - 1e-12

    noise = PaperNoise(
        dt_seconds=0.1,
        dephasing_strength=0.4,
        correlation_time_steps=10.0,
        correlation_length_sites=1.0,
    )
    noisy = build_noisy_circuit(
        result.schedule,
        architecture,
        noise,
        seed=19,
        local_action_ids=local_ids,
    )
    first = simulate_infidelity(noisy, reference, max_bond_dimension=8, svd_threshold=1e-12, seed=19)
    second = simulate_infidelity(noisy, reference, max_bond_dimension=8, svd_threshold=1e-12, seed=19)
    assert first == second
    assert 1e-8 < first < 1.0
