# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Focused integrity checks for paper sample checkpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from reproduce.paper.checkpoint import CheckpointStore

if TYPE_CHECKING:
    from reproduce.paper.analysis import RawRow


def test_checkpoint_store_commits_prefix_and_rejects_identity_mismatch(tmp_path) -> None:
    identity = {
        "mode": "quick",
        "python": "3.12.3",
        "scientific_inputs_sha256": "one",
        "implementation_sha256": "old",
    }
    store = CheckpointStore(tmp_path, identity, expected_rows_per_sample=2)
    store.prepare({"case": "schedule"}, {"cases": []})
    rows: list[RawRow] = [
        {"sample": 0, "state_infidelity": 0.1},
        {"sample": 0, "state_infidelity": 0.2},
    ]
    shard = store.commit(0, rows)
    table_shard = store.commit_table(0, rows[:1], expected_rows=1)

    assert store.completed_samples() == frozenset({0})
    assert store.completed_table_samples(1) == frozenset({0})
    assert len(store.load_prefix(1)) == 2
    assert len(store.load_table_prefix(1)) == 1
    assert store.load_compiled() == {"cases": []}
    assert not shard.with_name(f".{shard.name}.tmp").exists()
    assert not table_shard.with_name(f".{table_shard.name}.tmp").exists()
    with pytest.raises(FileExistsError, match="already exists"):
        store.commit(0, rows)

    compatible = CheckpointStore(
        tmp_path,
        {**identity, "implementation_sha256": "new"},
        expected_rows_per_sample=2,
    )
    compatible.validate_identity()

    incompatible = CheckpointStore(
        tmp_path,
        {**identity, "scientific_inputs_sha256": "two"},
        expected_rows_per_sample=2,
    )
    with pytest.raises(RuntimeError, match="does not match"):
        incompatible.validate_identity()

    store.compiled_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match its manifest"):
        store.load_compiled()
