"""Repeatable Panda joint-position gain calibration experiments."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

import mujoco
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator, model_validator

from roboforge.environments.mujoco_env import MujocoRobotEnvironment
from roboforge.experiments.transitions import TransitionRecord, TransitionRecorder


DEFAULT_TARGET_OFFSETS_RAD = (0.04, -0.04, 0.04, -0.04, 0.04, -0.04, 0.04)


class TrackingCalibrationSpec(BaseModel):
    """Validated bounded target-tracking study; offsets are relative to reset pose."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    initial_condition: Literal["panda_home_keyframe"] = "panda_home_keyframe"
    episodes: int = Field(default=3, ge=1, le=100)
    seed: int = Field(default=0, ge=0, le=4_294_967_295)
    target_offsets_rad: tuple[FiniteFloat, ...] = DEFAULT_TARGET_OFFSETS_RAD
    candidate_kp: tuple[FiniteFloat, ...] = (0.25, 0.5, 0.75, 1.0)
    max_steps: int = Field(default=250, ge=20, le=10_000)
    settling_window: int = Field(default=10, ge=2, le=500)
    settling_tolerance_rad: FiniteFloat = Field(default=0.015, gt=0)
    minimum_rmse_improvement_rad: FiniteFloat = Field(default=0.0005, gt=0)

    @field_validator("episodes", "seed", "max_steps", "settling_window", mode="before")
    @classmethod
    def reject_bool_counts(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("integer settings must not be booleans")
        return value

    @model_validator(mode="after")
    def validate_design(self) -> "TrackingCalibrationSpec":
        if len(self.target_offsets_rad) != 7:
            raise ValueError("target_offsets_rad must provide one offset for each of seven Panda joints")
        if not any(abs(value) > 0 for value in self.target_offsets_rad):
            raise ValueError("target_offsets_rad must contain at least one non-zero excitation")
        if any(abs(value) > 0.10 for value in self.target_offsets_rad):
            raise ValueError("each target offset is bounded to 0.10 radians")
        if not self.candidate_kp:
            raise ValueError("candidate_kp must contain at least one gain")
        if len(set(self.candidate_kp)) != len(self.candidate_kp):
            raise ValueError("candidate_kp values must be unique")
        if any(not 0.0 < value <= 1.0 for value in self.candidate_kp):
            raise ValueError("candidate gains must be in (0, 1]")
        if self.settling_window > self.max_steps:
            raise ValueError("settling_window cannot exceed max_steps")
        return self


class TrackingTrial(BaseModel):
    """Metrics from one seeded target-tracking rollout."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    episode_id: UUID
    seed: int
    configured_kp: FiniteFloat | None = Field(default=None, gt=0, le=1)
    estimated_kp: FiniteFloat
    tracking_rmse_rad: FiniteFloat = Field(ge=0)
    final_rmse_rad: FiniteFloat = Field(ge=0)
    settling_time_s: FiniteFloat | None = Field(default=None, ge=0)
    settled: bool
    constraint_violation_count: int = Field(ge=0)
    completed_steps: int = Field(ge=0)
    terminated: bool
    truncated: bool


class GainEvaluation(BaseModel):
    """Aggregate metrics for one controller-gain setting."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    configured_kp: FiniteFloat | None = Field(default=None, gt=0, le=1)
    estimated_kp: FiniteFloat
    mean_tracking_rmse_rad: FiniteFloat = Field(ge=0)
    mean_final_rmse_rad: FiniteFloat = Field(ge=0)
    settled_episodes: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    constraint_violation_count: int = Field(ge=0)
    trials: tuple[TrackingTrial, ...]


class CalibrationRunRecord(BaseModel):
    """Persisted outcome of a baseline-versus-candidate calibration run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    calibration_id: UUID = Field(default_factory=uuid4)
    experiment_id: UUID
    environment_id: Literal["mujoco_panda_scene"] = "mujoco_panda_scene"
    simulation_timestep_s: FiniteFloat = Field(gt=0)
    n_substeps: int = Field(ge=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    method: Literal["bounded_kp_grid_search"] = "bounded_kp_grid_search"
    objective: Literal["mean_joint_position_tracking_rmse_rad"] = "mean_joint_position_tracking_rmse_rad"
    spec: TrackingCalibrationSpec
    baseline: GainEvaluation
    candidates: tuple[GainEvaluation, ...]
    best_candidate_kp: FiniteFloat = Field(gt=0, le=1)
    accepted_candidate_kp: FiniteFloat | None = Field(default=None, gt=0, le=1)
    candidate_accepted: bool
    acceptance_reason: str
    transition_count: int = Field(ge=0)
    transition_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_decision(self) -> "CalibrationRunRecord":
        candidate_gains = {item.configured_kp for item in self.candidates}
        if self.best_candidate_kp not in candidate_gains:
            raise ValueError("best_candidate_kp must refer to an evaluated candidate")
        if self.candidate_accepted:
            if self.accepted_candidate_kp != self.best_candidate_kp:
                raise ValueError("accepted_candidate_kp must identify the best candidate")
            if not self.acceptance_reason.startswith("accepted:"):
                raise ValueError("accepted calibration must include an accepted decision reason")
        elif self.accepted_candidate_kp is not None:
            raise ValueError("rejected calibration cannot set accepted_candidate_kp")
        return self


class JsonlCalibrationStore:
    """Append-only validated JSONL store for calibration run results."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(self, record: CalibrationRunRecord | dict[str, Any]) -> None:
        validated = CalibrationRunRecord.model_validate(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(validated.model_dump_json(exclude_none=False) + "\n")
            stream.flush()

    def read_all(self) -> list[CalibrationRunRecord]:
        if not self.path.exists():
            return []
        records: list[CalibrationRunRecord] = []
        with self.path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    raw = json.loads(line)
                    if not isinstance(raw, dict) or "schema_version" not in raw:
                        raise ValueError("a versioned calibration record is required")
                    records.append(CalibrationRunRecord.model_validate(raw))
                except Exception as exc:
                    raise ValueError(f"Invalid calibration record at {self.path}:{line_number}: {exc}") from exc
        return records


def estimate_setpoint_gain(
    current_positions: np.ndarray,
    target_positions: np.ndarray,
    normalized_actions: np.ndarray,
    joint_limits: np.ndarray,
) -> float:
    """Least-squares estimate of the controller's setpoint blend gain.

    The controller law is ``q_cmd = q + kp * (q_target - q)``. Panda actuator
    actions are normalized joint-position setpoints, so they can be mapped back
    through the known joint control ranges. The estimate is the origin-constrained
    least-squares slope of ``q_cmd - q`` against ``q_target - q``. This estimates
    the implemented command blend only; it does not identify robot dynamics.
    """
    current = np.asarray(current_positions, dtype=np.float64)
    target = np.asarray(target_positions, dtype=np.float64)
    actions = np.asarray(normalized_actions, dtype=np.float64)
    limits = np.asarray(joint_limits, dtype=np.float64)
    if current.ndim != 2 or target.shape != current.shape or actions.shape != current.shape:
        raise ValueError("positions and actions must be matching episode-by-joint matrices")
    if limits.shape != (current.shape[1], 2):
        raise ValueError("joint_limits must provide min/max for each matrix column")
    if current.shape[0] == 0 or current.shape[1] == 0:
        raise ValueError("at least one sample and one joint are required")
    if not all(np.isfinite(array).all() for array in (current, target, actions, limits)):
        raise ValueError("gain estimation data must be finite")
    if np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("joint limits must have a positive range")
    if np.any(actions < -1.00001) or np.any(actions > 1.00001):
        raise ValueError("normalized actions must be within [-1, 1]")
    commanded = limits[:, 0] + (actions + 1.0) * 0.5 * (limits[:, 1] - limits[:, 0])
    target_delta = target - current
    command_delta = commanded - current
    informative = np.abs(target_delta) > 1e-5
    if int(np.count_nonzero(informative)) < 10:
        raise ValueError("insufficient informative samples to estimate controller gain")
    denominator = float(np.sum(target_delta[informative] ** 2))
    if denominator <= 1e-10:
        raise ValueError("insufficient target displacement to estimate controller gain")
    estimate = float(np.sum(target_delta[informative] * command_delta[informative]) / denominator)
    if not np.isfinite(estimate) or not -0.05 <= estimate <= 1.05:
        raise ValueError(f"estimated gain {estimate!r} is outside the expected bounded range")
    return estimate


def decide_candidate(
    baseline: GainEvaluation,
    candidates: tuple[GainEvaluation, ...] | list[GainEvaluation],
    *,
    minimum_improvement_rad: float,
) -> tuple[GainEvaluation, bool, str]:
    """Select the lowest-RMSE candidate and apply conservative acceptance rules."""
    if not candidates:
        raise ValueError("at least one candidate evaluation is required")
    if not np.isfinite(minimum_improvement_rad) or minimum_improvement_rad <= 0:
        raise ValueError("minimum_improvement_rad must be positive and finite")
    best = min(candidates, key=lambda item: (item.mean_tracking_rmse_rad, item.configured_kp or 0.0))
    improvement = baseline.mean_tracking_rmse_rad - best.mean_tracking_rmse_rad
    if len(baseline.trials) < 2 or len(best.trials) < 2:
        return best, False, "rejected: acceptance requires at least two paired episodes"
    baseline_seeds = tuple(trial.seed for trial in baseline.trials)
    if baseline_seeds != tuple(trial.seed for trial in best.trials):
        return best, False, "rejected: baseline and candidate episodes are not seed-matched"
    if improvement < minimum_improvement_rad:
        return best, False, (
            f"rejected: RMSE improvement {improvement:.6f} rad is below "
            f"the required {minimum_improvement_rad:.6f} rad"
        )
    if best.settled_episodes < baseline.settled_episodes:
        return best, False, "rejected: candidate settled fewer episodes than the baseline"
    if best.constraint_violation_count:
        return best, False, "rejected: candidate produced constraint violations"
    if best.failure_count:
        return best, False, "rejected: candidate had unsettled or terminated/truncated episodes"
    return best, True, f"accepted: RMSE improved by {improvement:.6f} rad and acceptance checks passed"


def compute_tracking_metrics(
    errors_rad: np.ndarray,
    *,
    settling_window: int,
    tolerance_rad: float,
    timestep_s: float,
) -> tuple[float, float, float | None, bool]:
    """Return whole-run RMSE, final-window RMSE, settling time, and settled flag."""
    errors = np.asarray(errors_rad, dtype=np.float64)
    if errors.ndim != 2 or errors.shape[0] == 0 or errors.shape[1] == 0:
        raise ValueError("tracking errors must be a non-empty step-by-joint matrix")
    if not np.isfinite(errors).all():
        raise ValueError("tracking errors must be finite")
    if settling_window < 1 or settling_window > errors.shape[0]:
        raise ValueError("settling_window must fit within the observed steps")
    if not np.isfinite(tolerance_rad) or tolerance_rad <= 0:
        raise ValueError("tolerance_rad must be positive and finite")
    if not np.isfinite(timestep_s) or timestep_s <= 0:
        raise ValueError("timestep_s must be positive and finite")
    rmse = float(np.sqrt(np.mean(errors**2)))
    final_rmse = float(np.sqrt(np.mean(errors[-settling_window:] ** 2)))
    within = np.all(np.abs(errors) <= tolerance_rad, axis=1)
    settle_index = next(
        (index for index in range(len(within)) if bool(np.all(within[index:]))),
        None,
    )
    settling_time = (settle_index + 1) * timestep_s if settle_index is not None else None
    return rmse, final_rmse, settling_time, settle_index is not None


class PandaControllerCalibrator:
    """Run paired seeded joint-target tracking trials against the MuJoCo Panda."""

    def __init__(self, env: MujocoRobotEnvironment) -> None:
        self.env = env
        home_key = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_key < 0:
            raise ValueError("Panda calibration requires the bundled model's 'home' keyframe")
        for joint_index in range(1, 8):
            env.joint_index(f"joint{joint_index}")

    def run(
        self,
        spec: TrackingCalibrationSpec,
        *,
        transition_recorder: TransitionRecorder,
    ) -> CalibrationRunRecord:
        if spec.max_steps >= self.env.max_episode_steps:
            raise ValueError("max_steps must be lower than the environment horizon")
        experiment_id = uuid4()
        transitions: list[TransitionRecord] = []
        baseline = self._evaluate_gain(spec, None, experiment_id, transition_recorder, transitions)
        candidates = tuple(
            self._evaluate_gain(spec, gain, experiment_id, transition_recorder, transitions)
            for gain in spec.candidate_kp
        )
        best, accepted, reason = decide_candidate(
            baseline, candidates,
            minimum_improvement_rad=spec.minimum_rmse_improvement_rad,
        )
        transition_jsonl = "".join(item.model_dump_json(exclude_none=False) + "\n" for item in transitions)
        return CalibrationRunRecord(
            experiment_id=experiment_id,
            simulation_timestep_s=float(self.env.model.opt.timestep),
            n_substeps=self.env.n_substeps,
            spec=spec,
            baseline=baseline,
            candidates=candidates,
            best_candidate_kp=best.configured_kp,
            accepted_candidate_kp=best.configured_kp if accepted else None,
            candidate_accepted=accepted,
            acceptance_reason=reason,
            transition_count=len(transitions),
            transition_content_sha256=hashlib.sha256(transition_jsonl.encode("utf-8")).hexdigest(),
        )

    def _evaluate_gain(
        self,
        spec: TrackingCalibrationSpec,
        kp: float | None,
        experiment_id: UUID,
        recorder: TransitionRecorder,
        all_transitions: list[TransitionRecord],
    ) -> GainEvaluation:
        from roboforge.controllers.end_effector_position import PandaEndEffectorPositionController

        trials: list[TrackingTrial] = []
        current_all: list[np.ndarray] = []
        target_all: list[np.ndarray] = []
        action_all: list[np.ndarray] = []
        for episode_index in range(spec.episodes):
            seed = (spec.seed + episode_index) % (4_294_967_295 + 1)
            episode_id = uuid4()
            observation, _ = self.env.reset(seed=seed)
            controller = PandaEndEffectorPositionController(self.env, kp=kp)
            indices = controller.observation_indices
            action_indices = controller.action_indices
            target = observation[indices].astype(np.float64) + np.asarray(spec.target_offsets_rad)
            limits = controller.ik.limits
            if np.any(target < limits[:, 0]) or np.any(target > limits[:, 1]):
                raise ValueError("configured target offsets exceed Panda joint limits at reset")
            controller.set_joint_target(target)
            error_history: list[np.ndarray] = []
            constraint_violations = 0
            terminated = truncated = False
            for transition_index in range(spec.max_steps):
                current = observation[indices].astype(np.float64)
                action = controller.compute_action(observation)
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                next_q = next_observation[indices].astype(np.float64)
                error_history.append(next_q - target)
                current_all.append(current)
                target_all.append(target.copy())
                action_all.append(action[action_indices].astype(np.float64))
                outside = np.any(next_q < limits[:, 0] - 1e-5) or np.any(next_q > limits[:, 1] + 1e-5)
                bad_action = np.any(action < -1.00001) or np.any(action > 1.00001)
                constraint_violations += int(outside or bad_action)
                transition = TransitionRecord.from_transition(
                    experiment_id=experiment_id,
                    episode_id=episode_id,
                    transition_index=transition_index,
                    seed=seed,
                    observation=observation,
                    action=action,
                    next_observation=next_observation,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    simulation_time_s=info.get("simulation_time"),
                    simulation_timestep_s=float(self.env.model.opt.timestep * self.env.n_substeps),
                    environment_step=info.get("step_count"),
                    task_phase="JOINT_TARGET_TRACKING",
                )
                recorder.write(transition)
                all_transitions.append(transition)
                observation = next_observation
                if terminated or truncated:
                    break
            errors = np.asarray(error_history, dtype=np.float64)
            if errors.size == 0 or not np.isfinite(errors).all():
                raise ValueError("calibration rollout produced no finite tracking samples")
            timestep = float(self.env.model.opt.timestep * self.env.n_substeps)
            rmse, final_rmse, settling_time, settled = compute_tracking_metrics(
                errors,
                settling_window=min(spec.settling_window, len(errors)),
                tolerance_rad=spec.settling_tolerance_rad,
                timestep_s=timestep,
            )
            trials.append(TrackingTrial(
                episode_id=episode_id,
                seed=seed,
                configured_kp=kp,
                # Filled with the common fitted estimate after all trial samples.
                estimated_kp=kp if kp is not None else 1.0,
                tracking_rmse_rad=rmse,
                final_rmse_rad=final_rmse,
                settling_time_s=settling_time,
                settled=settled,
                constraint_violation_count=constraint_violations,
                completed_steps=len(errors),
                terminated=bool(terminated),
                truncated=bool(truncated),
            ))
        joint_limits = controller.ik.limits
        estimated = estimate_setpoint_gain(
            np.asarray(current_all)[:, :],
            np.asarray(target_all)[:, :],
            np.asarray(action_all)[:, :],
            joint_limits,
        )
        trials = [trial.model_copy(update={"estimated_kp": estimated}) for trial in trials]
        failures = sum(not trial.settled or trial.terminated or trial.truncated for trial in trials)
        return GainEvaluation(
            configured_kp=kp,
            estimated_kp=estimated,
            mean_tracking_rmse_rad=float(np.mean([trial.tracking_rmse_rad for trial in trials])),
            mean_final_rmse_rad=float(np.mean([trial.final_rmse_rad for trial in trials])),
            settled_episodes=sum(trial.settled for trial in trials),
            failure_count=failures,
            constraint_violation_count=sum(trial.constraint_violation_count for trial in trials),
            trials=tuple(trials),
        )
