# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Integrity checks for the frozen paper circuit corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from qiskit import QuantumCircuit

INPUT_DIR = Path(__file__).parents[2] / "reproduce" / "paper" / "circuits"

EXPECTED = {
    "ghz_nativegates_quantinuum_qiskit_opt2_4.qasm": (
        "5e91a8c3ae55699b5dd2dec96f5f64a865c36d54c6de0aab5bef46b9bcc9fa65",
        4,
        20,
    ),
    "ghz_nativegates_quantinuum_qiskit_opt2_5.qasm": (
        "650a967f81c2fe70997421f1af2245d8bda8c2fc40c7981dafb9227b980d848b",
        5,
        26,
    ),
    "ghz_nativegates_quantinuum_qiskit_opt2_6.qasm": (
        "316ff564cd7eb18b33fe91c09e1626214d29b11bcff39be39e2a041c024e6f77",
        6,
        32,
    ),
    "ghz_nativegates_quantinuum_qiskit_opt2_8.qasm": (
        "14dde0d889409aff279b1b30551dc1480b72a25cd5eaa75e22e53d8fb49bc3ff",
        8,
        44,
    ),
    "ising_4.qasm": ("5cc0a7f7057f68d6534c49b09f13037e6de27845c17b0cb665d476d3116c2097", 4, 96),
    "ising_5.qasm": ("f53ce433cc5072efbcae0c356c0cb65f2b2e41413c6c8d9b83fae5d61b8f7a0b", 5, 120),
    "ising_6.qasm": ("c58faab31fd39dae37121784c2df40e8a6fc8a6276fd1cce1c56f07852ab6742", 6, 144),
    "ising_8.qasm": ("86c4c6b61e2369ee8c4ab03cd5a9e9a922e8c2dcdf642de5d7f5f10724ac4a02", 8, 192),
    "qft_nativegates_quantinuum_qiskit_opt2_4.qasm": (
        "4eaf4974cc3057e2097ba0e4ce969fde807fd4b103d1f40e1f48f907d9a92bbc",
        4,
        28,
    ),
    "qft_nativegates_quantinuum_qiskit_opt2_5.qasm": (
        "a1e2224bad50ed97809f598fe69180b62c39a5cd7e78cb6568b7588c83fbdff9",
        5,
        102,
    ),
    "qft_nativegates_quantinuum_qiskit_opt2_6.qasm": (
        "265b7318f4e700876cfa9fd5ce06d60ea0e865ee13109f130c8363f9803fd1aa",
        6,
        149,
    ),
    "qft_nativegates_quantinuum_qiskit_opt2_8.qasm": (
        "9d037d50c6bed5f002342f204c4dd3a79fdf593f27d5812825efce10b74e9890",
        8,
        247,
    ),
    "qpeexact_nativegates_quantinuum_qiskit_opt2_4.qasm": (
        "4d4684534c3e72b792300916a9c7e620c745a8370b3bc1d66dd25ec3fad5bfbf",
        4,
        19,
    ),
    "qpeexact_nativegates_quantinuum_qiskit_opt2_5.qasm": (
        "fadf157f7d964a52ad37a8345d77724756251b70298ae0fb8de8c427a7e3d9dc",
        5,
        43,
    ),
    "qpeexact_nativegates_quantinuum_qiskit_opt2_6.qasm": (
        "8efd96ad270823e3eff6dac007dc4f306b374bc26fbd0a85f07360a142f27f30",
        6,
        55,
    ),
    "qpeexact_nativegates_quantinuum_qiskit_opt2_8.qasm": (
        "336a1b55e6e5351c56175ae9f4e55d294b1b57687ad7a39a6865ddca83190449",
        8,
        106,
    ),
    "randomcircuit_nativegates_quantinuum_qiskit_opt2_seed0_4.qasm": (
        "f5cb3ee358f14aa1039209794750ebcda09a799fb8841e84c85b927310ecce36",
        4,
        59,
    ),
    "randomcircuit_nativegates_quantinuum_qiskit_opt2_seed0_5.qasm": (
        "0580ca4d9e0ecef315a37c3b7b853f7c0d6192eb1192e510c72e26dcfcaae7f8",
        5,
        75,
    ),
    "randomcircuit_nativegates_quantinuum_qiskit_opt2_seed0_6.qasm": (
        "c02b3855973a30847ac5f279f7f66bd1868b8a7d5d0a261fabff587d73dcccc6",
        6,
        124,
    ),
    "randomcircuit_nativegates_quantinuum_qiskit_opt2_seed0_8.qasm": (
        "b30cf74f1b7591df2939c91de1726a396c41f216c94b7957433980b534970cc2",
        8,
        286,
    ),
}


@pytest.mark.parametrize(("name", "expected"), EXPECTED.items())
def test_frozen_paper_circuit(name: str, expected: tuple[str, int, int]) -> None:
    path = INPUT_DIR / name
    digest, qubits, operations = expected

    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    circuit = QuantumCircuit.from_qasm_file(path)
    assert (circuit.num_qubits, circuit.size()) == (qubits, operations)


def test_frozen_schedule_bundle_has_the_audited_resource_totals() -> None:
    envelope = json.loads((INPUT_DIR / "compiled_schedules.json").read_text(encoding="utf-8"))
    rows = envelope["payload"]["schedule_rows"]

    totals = {
        method: (
            sum(int(row["local_pulses"]) for row in rows if row["method"] == method),
            sum(int(row["transport_actions"]) for row in rows if row["method"] == method),
        )
        for method in ("NoDD", "IdealizedHahn", "NearestHahn", "PulseOnlySADD", "FullSADD")
    }

    assert envelope["schema_version"] == 1
    assert len(envelope["payload"]["cases"]) == 20
    assert totals == {
        "NoDD": (0, 2003),
        "IdealizedHahn": (1122, 2003),
        "NearestHahn": (755, 2003),
        "PulseOnlySADD": (985, 2003),
        "FullSADD": (1268, 2365),
    }
