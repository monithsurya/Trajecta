"""Versioned step-transition records and JSONL dataset helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import UUID

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


def _vector(value: Any, name: str) -> tuple[float, ...]:
    """Convert one finite one-dimensional numeric vector to JSON-safe values."""
    if isinstance(value, np.ndarray):
        if (value.ndim != 1 or not np.issubdtype(value.dtype, np.number)
                or np.issubdtype(value.dtype, np.complexfloating)):
            raise ValueError(f"{name} must be a one-dimensional numeric array")
        array = value
    elif isinstance(value, (list, tuple)):
        if any(isinstance(item, (list, tuple, dict, bool)) for item in value):
            raise ValueError(f"{name} must be a one-dimensional numeric vector")
        if any(not isinstance(item, (int, float, np.integer, np.floating)) for item in value):
            raise ValueError(f"{name} must contain only numeric values")
        array = np.asarray(value)
    else:
        raise ValueError(f"{name} must be a one-dimensional numeric vector")
    converted = np.asarray(array, dtype=np.float64)
    if converted.size == 0 or not np.isfinite(converted).all():
        raise ValueError(f"{name} must be a non-empty vector of finite values")
    return tuple(float(item) for item in converted)


class TransitionRecord(BaseModel):
    """One environment transition; configuration is referenced by run IDs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    experiment_id: UUID
    episode_id: UUID
    transition_index: int = Field(ge=0)
    seed: int | None = Field(default=None, ge=0, le=4_294_967_295)
    observation: tuple[FiniteFloat, ...]
    action: tuple[FiniteFloat, ...]
    next_observation: tuple[FiniteFloat, ...]
    reward: FiniteFloat | None = None
    terminated: bool
    truncated: bool
    simulation_time_s: FiniteFloat | None = Field(default=None, ge=0)
    simulation_timestep_s: FiniteFloat | None = Field(default=None, gt=0)
    environment_step: int | None = Field(default=None, ge=0)
    task_phase: str | None = None
    task_success: bool | None = None
    task_failure_reason: str | None = None

    @classmethod
    def from_transition(cls, **values: Any) -> "TransitionRecord":
        for field in ("observation", "action", "next_observation"):
            values[field] = _vector(values[field], field)
        return cls.model_validate(values)


class TransitionRecorder(Protocol):
    """Persistence interface for individual transitions."""

    def write(self, record: TransitionRecord) -> None: ...


class JsonlTransitionStore:
    """Append-only JSONL transition store with strict line-numbered reads."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(self, record: TransitionRecord | dict[str, Any]) -> None:
        validated = record if isinstance(record, TransitionRecord) else TransitionRecord.from_transition(**record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(validated.model_dump_json(exclude_none=False) + "\n")
            stream.flush()

    def read_all(self) -> list[TransitionRecord]:
        if not self.path.exists():
            return []
        records: list[TransitionRecord] = []
        with self.path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    raw = json.loads(line)
                    if not isinstance(raw, dict) or "schema_version" not in raw:
                        raise ValueError("a versioned transition object is required")
                    records.append(TransitionRecord.model_validate(raw))
                except Exception as exc:
                    raise ValueError(f"Invalid transition at {self.path}:{line_number}: {exc}") from exc
        return records


class TransitionDataset:
    """Loaded, validated transition sequence with integrity manifest support."""

    def __init__(
        self,
        records: list[TransitionRecord] | tuple[TransitionRecord, ...],
        *,
        source_path: str = "",
    ) -> None:
        self.records = tuple(TransitionRecord.model_validate(record) for record in records)
        self.source_path = source_path
        next_indices: dict[tuple[UUID, UUID], int] = {}
        for record in self.records:
            key = (record.experiment_id, record.episode_id)
            expected = next_indices.get(key, 0)
            if record.transition_index != expected:
                raise ValueError(
                    f"Transition sequence for episode {record.episode_id} expected index {expected}, "
                    f"got {record.transition_index}"
                )
            next_indices[key] = expected + 1

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "TransitionDataset":
        store = JsonlTransitionStore(path)
        return cls(store.read_all(), source_path=str(Path(path).resolve()))

    def manifest(self) -> dict[str, Any]:
        serialized = "".join(record.model_dump_json(exclude_none=False) + "\n" for record in self.records)
        episodes: dict[str, dict[str, Any]] = {}
        for record in self.records:
            episode = episodes.setdefault(
                str(record.episode_id),
                {"experiment_id": str(record.experiment_id), "seed": record.seed, "record_count": 0},
            )
            episode["record_count"] += 1
        return {
            "manifest_version": 1,
            "schema_versions": sorted({record.schema_version for record in self.records}),
            "record_count": len(self.records),
            "experiment_ids": sorted({str(record.experiment_id) for record in self.records}),
            "episodes": episodes,
            "source_path": self.source_path,
            "content_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        }

    def write_manifest(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.manifest(), indent=2) + "\n", encoding="utf-8")
