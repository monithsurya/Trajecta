"""Typed experience records and local append-only JSON Lines persistence."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator


Position3 = tuple[FiniteFloat, FiniteFloat, FiniteFloat]


class ExperienceRecord(BaseModel):
    """One episode's known configuration and outcome, without simulator data."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    experiment_id: UUID = Field(default_factory=uuid4)
    episode_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    task_id: str = Field(min_length=1)
    environment_id: str = Field(min_length=1)
    experiment_config: dict[str, Any] = Field(default_factory=dict)
    task_config: dict[str, Any] | None = None
    seed: int | None = Field(default=None, ge=0, le=4_294_967_295)
    success: bool
    outcome: Literal["success", "failure"]
    grasp_verified: bool | None = None
    episode_length: int | None = Field(default=None, ge=0)
    duration_seconds: FiniteFloat | None = Field(default=None, ge=0)
    initial_cube_position_m: Position3 | None = None
    final_cube_position_m: Position3 | None = None
    placement_error_m: FiniteFloat | None = Field(default=None, ge=0)
    phase_history: tuple[str, ...] = ()
    failure_reason: str | None = None
    controller_config: dict[str, Any] | None = None
    condition_parameters: dict[str, Any] | None = None
    failure_category: str | None = None
    recovery_attempts: int = Field(default=0, ge=0)
    replanning_count: int = Field(default=0, ge=0)

    @model_validator(mode="before")
    @classmethod
    def derive_outcome_from_explicit_success(cls, value: Any) -> Any:
        if isinstance(value, dict) and "outcome" not in value and isinstance(value.get("success"), bool):
            value = {**value, "outcome": "success" if value["success"] else "failure"}
        return value


class ExperienceRecorder(Protocol):
    """Small persistence boundary consumed by experiment orchestration."""

    def write(self, record: ExperienceRecord) -> None:
        """Validate and persist one episode record."""


def experience_from_result(
    result: Any,
    *,
    task_id: str,
    environment_id: str,
    seed: int | None = None,
    experiment_config: dict[str, Any] | None = None,
    task_config: dict[str, Any] | None = None,
    controller_config: dict[str, Any] | None = None,
    duration_seconds: float | None = None,
    experiment_id: UUID | None = None,
    episode_id: UUID | None = None,
) -> ExperienceRecord:
    """Build a record only from fields exposed by the result and run context."""
    success = getattr(result, "success", None)
    if not isinstance(success, bool) and type(success).__name__ == "bool_":
        success = bool(success)
    if not isinstance(success, bool):
        raise ValueError("task result must expose an explicit boolean 'success'")

    def position(name: str) -> Position3 | None:
        value = getattr(result, name, None)
        if value is None:
            return None
        if len(value) != 3:
            raise ValueError(f"{name} must contain exactly three coordinates")
        return tuple(float(component) for component in value)  # type: ignore[return-value]

    return ExperienceRecord(
        task_id=task_id,
        experiment_id=experiment_id or uuid4(),
        episode_id=episode_id or uuid4(),
        environment_id=environment_id,
        experiment_config=experiment_config or {},
        task_config=task_config,
        seed=seed,
        success=success,
        outcome="success" if success else "failure",
        grasp_verified=getattr(result, "grasp_verified", None),
        episode_length=getattr(result, "episode_length", None),
        duration_seconds=duration_seconds,
        initial_cube_position_m=position("initial_cube_position_m"),
        final_cube_position_m=position("final_cube_position_m"),
        placement_error_m=getattr(result, "final_placement_error_m", None),
        phase_history=tuple(getattr(result, "phase_history", ()) or ()),
        failure_reason=getattr(result, "failure_reason", None),
        controller_config=controller_config,
        condition_parameters=getattr(result, "condition_parameters", None),
        failure_category=getattr(result, "failure_category", None),
        recovery_attempts=getattr(result, "recovery_attempts", 0),
        replanning_count=getattr(result, "replanning_count", 0),
    )


class JsonlExperienceStore:
    """Append-only JSONL store; malformed lines raise with path and line number."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(self, record: ExperienceRecord | dict[str, Any]) -> None:
        """Validate a record and append exactly one JSON object plus newline."""
        validated = ExperienceRecord.model_validate(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(validated.model_dump_json(exclude_none=False) + "\n")

    def write_many(self, records: list[ExperienceRecord] | tuple[ExperienceRecord, ...]) -> None:
        """Validate the complete batch before appending any records."""
        validated = [ExperienceRecord.model_validate(record) for record in records]
        if not validated:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            for record in validated:
                stream.write(record.model_dump_json(exclude_none=False) + "\n")

    def read_all(self) -> list[ExperienceRecord]:
        """Read and validate every line; stop with a diagnostic on corruption."""
        if not self.path.exists():
            return []
        records: list[ExperienceRecord] = []
        with self.path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    raw = json.loads(line)
                    if not isinstance(raw, dict):
                        raise ValueError("each JSONL line must contain an object")
                    if "schema_version" not in raw:
                        raise ValueError("schema_version is required on persisted records")
                    records.append(ExperienceRecord.model_validate(raw))
                except Exception as exc:
                    raise ValueError(
                        f"Invalid experience record at {self.path}:{line_number}: {exc}"
                    ) from exc
        return records
