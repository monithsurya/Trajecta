"""Experiment specification domain models and schema validation for Trajecta.

This module defines the foundational configuration contract for Trajecta experiments,
enforcing strict validation rules for tasks, episode counts, random seeds, and
controller configurations using Pydantic.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TaskType(str, Enum):
    """Supported task identifiers for Trajecta experiments."""

    PICK_AND_PLACE = "pick_and_place"
    PANDA_REACH = "panda_reach"


class ControllerType(str, Enum):
    """Supported robot controller types."""

    JOINT_POSITION = "joint_position"
    JOINT_VELOCITY = "joint_velocity"
    OPERATIONAL_SPACE = "operational_space"
    CARTESIAN_IMPEDANCE = "cartesian_impedance"


def _validate_non_negative_gains(
    value: float | list[float] | None, field_name: str
) -> float | list[float] | None:
    """Validate that control gains are non-negative scalar values or sequences."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError(f"{field_name} must be non-negative; got {value}")
        return float(value)
    if isinstance(value, list):
        for idx, gain in enumerate(value):
            if not isinstance(gain, (int, float)):
                raise ValueError(
                    f"{field_name} elements must be numeric; got element {idx} of type {type(gain).__name__}"
                )
            if gain < 0:
                raise ValueError(
                    f"{field_name} elements must be non-negative; got {gain} at index {idx}"
                )
        return [float(g) for g in value]
    raise ValueError(f"{field_name} must be a float, a list of floats, or None; got {type(value).__name__}")


class ControllerConfig(BaseModel):
    """Structured configuration for a robot controller.

    Attributes:
        controller_type: The type of controller to instantiate.
        frequency_hz: Control loop execution frequency in Hertz (must be > 0).
        kp: Optional proportional gain (scalar or per-joint list), must be >= 0.
        kd: Optional derivative gain (scalar or per-joint list), must be >= 0.
        extra_params: Optional controller-specific tuning parameters.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    controller_type: ControllerType = Field(
        ...,
        description="Controller category or control law identifier.",
    )
    frequency_hz: float = Field(
        default=50.0,
        gt=0.0,
        description="Control frequency in Hertz. Must be strictly positive.",
    )
    kp: float | list[float] | None = Field(
        default=None,
        description="Proportional gain(s). Must be non-negative if specified.",
    )
    kd: float | list[float] | None = Field(
        default=None,
        description="Derivative gain(s). Must be non-negative if specified.",
    )
    extra_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional key-value configuration for specialized controllers.",
    )

    @field_validator("kp", mode="after")
    @classmethod
    def validate_kp(cls, v: float | list[float] | None) -> float | list[float] | None:
        return _validate_non_negative_gains(v, "kp")

    @field_validator("kd", mode="after")
    @classmethod
    def validate_kd(cls, v: float | list[float] | None) -> float | list[float] | None:
        return _validate_non_negative_gains(v, "kd")


# Maximum value for a standard 32-bit unsigned integer random seed (2^32 - 1)
MAX_SEED_VALUE: int = 4_294_967_295


class ExperimentSpec(BaseModel):
    """Immutable, validated experiment specification for Trajecta.

    An ExperimentSpec defines the complete, declarative contract required to
    configure and reproduce a robotics experiment. It remains strictly decoupled
    from execution engines, simulators, persistence layers, and ML runtimes.

    Attributes:
        task: Identifier of the robotics task to execute.
        episodes: Total number of episodes to execute (strictly positive integer).
        seed: Optional deterministic random seed in range [0, 2^32 - 1].
        controller: Structured controller configuration.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: TaskType = Field(
        ...,
        description="The task identifier for the experiment.",
    )
    episodes: int = Field(
        ...,
        gt=0,
        description="Number of episodes to execute. Must be a strictly positive integer.",
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        le=MAX_SEED_VALUE,
        description="Deterministic random seed in range [0, 2^32 - 1]. None indicates unseeded execution.",
    )
    controller: ControllerConfig = Field(
        ...,
        description="Structured controller configuration.",
    )

    @field_validator("episodes", mode="before")
    @classmethod
    def validate_episodes_not_bool(cls, v: Any) -> Any:
        if isinstance(v, bool):
            raise ValueError("Episode count must be an integer, not a boolean")
        return v

    @field_validator("seed", mode="before")
    @classmethod
    def validate_seed_not_bool(cls, v: Any) -> Any:
        if isinstance(v, bool):
            raise ValueError("Seed must be an integer, not a boolean")
        return v
