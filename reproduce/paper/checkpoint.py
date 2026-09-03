# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Atomic per-sample checkpoints for the paper reproduction command."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from .analysis import RawRow, read_csv, write_csv

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

_SCHEMA_VERSION = 1
_SAMPLE_NAME = re.compile(r"sample_(\d{6})\.csv")
_TABLE_NAME = re.compile(r"table_ii_(\d{6})\.csv")


@dataclass(frozen=True)
class CheckpointStore:
    """Validate and manage one output directory's sample checkpoints."""

    output_dir: Path
    identity: Mapping[str, object]
    expected_rows_per_sample: int

    @property
    def checkpoint_dir(self) -> Path:
        """Directory containing restart-safe state."""
        return self.output_dir / "checkpoints"

    @property
    def manifest_path(self) -> Path:
        """Run manifest path."""
        return self.checkpoint_dir / "manifest.json"

    @property
    def compiled_path(self) -> Path:
        """Exact compiled schedule snapshot path."""
        return self.checkpoint_dir / "compiled.json"

    @property
    def exists(self) -> bool:
        """Whether this output directory contains an initialized checkpoint."""
        return self.manifest_path.exists()

    def validate_identity(self) -> None:
        """Reject an existing checkpoint created by incompatible inputs.

        Raises:
            RuntimeError: If the manifest is malformed or its identity differs.
        """
        if not self.manifest_path.exists():
            return
        manifest = self._read_manifest()
        stored_identity = manifest.get("identity")
        compatible_keys = ("mode", "python", "scientific_inputs_sha256")
        if not isinstance(stored_identity, dict) or any(
            stored_identity.get(key) != self.identity.get(key) for key in compatible_keys
        ):
            msg = (
                "checkpoint scientific configuration or Python version does not match this run; "
                "choose a new output directory"
            )
            raise RuntimeError(msg)
        if manifest.get("expected_rows_per_sample") != self.expected_rows_per_sample:
            msg = "checkpoint sample shape does not match this run; choose a new output directory"
            raise RuntimeError(msg)

    def prepare(
        self,
        schedule_fingerprints: Mapping[str, str],
        compiled_payload: Mapping[str, object],
    ) -> None:
        """Atomically store exact schedules and initialize the manifest.

        Raises:
            RuntimeError: If checkpoint files already exist inconsistently.
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            self.validate_identity()
            return
        if any(self.checkpoint_dir.glob("sample_*.csv")):
            msg = (
                "sample checkpoints exist without a manifest; "
                "choose a new output directory or remove the orphaned shards"
            )
            raise RuntimeError(msg)
        compiled_sha256 = _write_json_atomic(self.compiled_path, compiled_payload)
        _write_json_atomic(
            self.manifest_path,
            {
                "schema_version": _SCHEMA_VERSION,
                "identity": self.identity,
                "expected_rows_per_sample": self.expected_rows_per_sample,
                "schedule_fingerprints": schedule_fingerprints,
                "compiled_sha256": compiled_sha256,
            },
        )

    def load_compiled(self) -> dict[str, object]:
        """Load and authenticate the exact schedules used by existing shards.

        Returns:
            The stored compiled schedule and static-analysis payload.

        Raises:
            RuntimeError: If the snapshot is absent, malformed, or modified.
            TypeError: If the snapshot's top-level value is not an object.
        """
        manifest = self._read_manifest()
        try:
            raw = self.compiled_path.read_bytes()
            value = json.loads(raw)
        except (OSError, json.JSONDecodeError) as error:
            msg = f"cannot read compiled checkpoint: {self.compiled_path}"
            raise RuntimeError(msg) from error
        if hashlib.sha256(raw).hexdigest() != manifest.get("compiled_sha256"):
            msg = f"compiled checkpoint does not match its manifest: {self.compiled_path}"
            raise RuntimeError(msg)
        if not isinstance(value, dict):
            msg = f"compiled checkpoint must contain an object: {self.compiled_path}"
            raise TypeError(msg)
        return cast("dict[str, object]", value)

    def completed_samples(self) -> frozenset[int]:
        """Validate existing shards and return their sample indices.

        Returns:
            All complete sample indices in the checkpoint directory.

        """
        return self._completed_samples("sample_*.csv", _SAMPLE_NAME, self.expected_rows_per_sample)

    def completed_table_samples(self, expected_rows: int) -> frozenset[int]:
        """Return validated Table II supplement sample indices."""
        return self._completed_samples("table_ii_*.csv", _TABLE_NAME, expected_rows)

    def _completed_samples(self, glob: str, pattern: re.Pattern[str], expected_rows: int) -> frozenset[int]:
        completed: set[int] = set()
        for path in sorted(self.checkpoint_dir.glob(glob)):
            match = pattern.fullmatch(path.name)
            if match is None:
                msg = f"invalid checkpoint shard name: {path.name}"
                raise RuntimeError(msg)
            sample = int(match.group(1))
            rows = read_csv(path)
            if len(rows) != expected_rows:
                msg = f"checkpoint shard {path.name} has {len(rows)} rows; expected {expected_rows}"
                raise RuntimeError(msg)
            if any(row.get("sample") != str(sample) for row in rows):
                msg = f"checkpoint shard {path.name} contains rows for another sample"
                raise RuntimeError(msg)
            completed.add(sample)
        return frozenset(completed)

    def commit(self, sample: int, rows: Sequence[RawRow]) -> Path:
        """Atomically commit one complete sample shard.

        Returns:
            The committed shard path.

        Raises:
            ValueError: If the sample index or row shape is invalid.
        """
        if len(rows) != self.expected_rows_per_sample:
            msg = f"sample {sample} produced {len(rows)} rows; expected {self.expected_rows_per_sample}"
            raise ValueError(msg)
        return self._commit(sample, rows, "sample")

    def commit_table(self, sample: int, rows: Sequence[RawRow], expected_rows: int) -> Path:
        """Atomically commit one Table II supplement shard.

        Returns:
            The committed shard path.

        Raises:
            ValueError: If the row shape is invalid.
        """
        if len(rows) != expected_rows:
            msg = f"Table II sample {sample} produced {len(rows)} rows; expected {expected_rows}"
            raise ValueError(msg)
        return self._commit(sample, rows, "table_ii")

    def _commit(self, sample: int, rows: Sequence[RawRow], prefix: str) -> Path:
        if sample < 0:
            msg = "sample index must be non-negative"
            raise ValueError(msg)
        if any(int(row["sample"]) != sample for row in rows):
            msg = f"sample {sample} contains rows for another sample"
            raise ValueError(msg)
        path = self.checkpoint_dir / f"{prefix}_{sample:06d}.csv"
        if path.exists():
            msg = f"checkpoint already exists: {path}"
            raise FileExistsError(msg)
        _write_csv_atomic(path, rows)
        return path

    def load_prefix(self, samples: int) -> list[RawRow]:
        """Load the requested contiguous sample prefix.

        Returns:
            Rows from samples ``0`` through ``samples - 1``.

        Raises:
            RuntimeError: If a requested sample is missing.
        """
        rows: list[RawRow] = []
        for sample in range(samples):
            path = self.checkpoint_dir / f"sample_{sample:06d}.csv"
            if not path.exists():
                msg = f"missing checkpoint for sample {sample}"
                raise RuntimeError(msg)
            rows.extend(cast("list[RawRow]", read_csv(path)))
        return rows

    def load_table_prefix(self, samples: int) -> list[RawRow]:
        """Load the requested contiguous Table II supplement prefix.

        Returns:
            Rows from the requested samples.

        Raises:
            RuntimeError: If a requested supplement is missing.
        """
        rows: list[RawRow] = []
        for sample in range(samples):
            path = self.checkpoint_dir / f"table_ii_{sample:06d}.csv"
            if not path.exists():
                msg = f"missing Table II checkpoint for sample {sample}"
                raise RuntimeError(msg)
            rows.extend(cast("list[RawRow]", read_csv(path)))
        return rows

    def _read_manifest(self) -> dict[str, object]:
        try:
            value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            msg = f"cannot read checkpoint manifest: {self.manifest_path}"
            raise RuntimeError(msg) from error
        if not isinstance(value, dict) or value.get("schema_version") != _SCHEMA_VERSION:
            msg = f"unsupported checkpoint manifest: {self.manifest_path}"
            raise RuntimeError(msg)
        return cast("dict[str, object]", value)


def _write_csv_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        write_csv(temporary, rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    raw = f"{json.dumps(value, indent=2, sort_keys=True)}\n".encode()
    try:
        temporary.write_bytes(raw)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(raw).hexdigest()


__all__ = ["CheckpointStore"]
