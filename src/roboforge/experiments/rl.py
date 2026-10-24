"""Optional PPO training and paired-policy evaluation for Panda reaching."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator, model_validator

from roboforge.environments.panda_reach import PandaReachConfig, PandaReachEnv
from roboforge.experiments.experience import ExperienceRecord, JsonlExperienceStore, experience_from_result
from roboforge.experiments.transitions import TransitionRecord, TransitionRecorder


class PPOTrainingConfig(BaseModel):
    """Validated PPO hyperparameters; PPO and PyTorch are an optional extra."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_timesteps: int = Field(default=50_000, ge=64, le=100_000_000)
    seed: int = Field(default=0, ge=0, le=4_294_967_295)
    learning_rate: FiniteFloat = Field(default=3e-4, gt=0, le=0.1)
    n_steps: int = Field(default=512, ge=32, le=100_000)
    batch_size: int = Field(default=64, ge=8, le=100_000)
    n_epochs: int = Field(default=10, ge=1, le=100)
    gamma: FiniteFloat = Field(default=0.99, gt=0, le=1)
    gae_lambda: FiniteFloat = Field(default=0.95, gt=0, le=1)
    clip_range: FiniteFloat = Field(default=0.2, gt=0, le=0.5)
    eval_freq: int = Field(default=5_000, ge=1)
    eval_episodes: int = Field(default=5, ge=1, le=1000)
    hidden_sizes: tuple[int, ...] = (64, 64)
    device: Literal["cpu"] = "cpu"

    @field_validator("total_timesteps", "seed", "n_steps", "batch_size", "n_epochs", "eval_freq", "eval_episodes", mode="before")
    @classmethod
    def reject_boolean_integers(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("integer training settings must not be booleans")
        return value

    @model_validator(mode="after")
    def validate_rollout_sizes(self) -> "PPOTrainingConfig":
        if self.batch_size > self.n_steps or self.n_steps % self.batch_size:
            raise ValueError("batch_size must divide n_steps and not exceed it")
        if self.eval_freq > self.total_timesteps:
            raise ValueError("eval_freq must be no larger than total_timesteps")
        if not self.hidden_sizes or any(size < 8 or size > 1024 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain widths in [8, 1024]")
        if self.total_timesteps < self.n_steps:
            raise ValueError("total_timesteps must be at least one PPO rollout (n_steps)")
        return self


class EpisodeEvaluation(BaseModel):
    """Outcome and metrics for one deterministic-policy evaluation episode."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    episode_id: UUID
    policy_name: str
    seed: int
    success: bool
    episode_return: FiniteFloat
    episode_length: int = Field(ge=0)
    final_target_error_m: FiniteFloat = Field(ge=0)
    terminated: bool
    truncated: bool
    failure_reason: str | None = None
    constraint_violations: int = Field(ge=0)


class PolicyEvaluation(BaseModel):
    """Aggregate reach metrics and the underlying episode outcomes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_name: str
    policy_type: Literal["heuristic_ik", "ppo"]
    attempted_episodes: int = Field(ge=0)
    success_rate: FiniteFloat = Field(ge=0, le=1)
    mean_episode_return: FiniteFloat
    mean_final_target_error_m: FiniteFloat = Field(ge=0)
    mean_episode_length: FiniteFloat = Field(ge=0)
    failure_counts: dict[str, int]
    constraint_violations: int = Field(ge=0)
    episodes: tuple[EpisodeEvaluation, ...]


class ReachEvaluationRecord(BaseModel):
    """Validated paired comparison between the heuristic and trained policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    comparison_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    seed: int
    task_config: PandaReachConfig
    model_path: str
    baseline: PolicyEvaluation
    trained_policy: PolicyEvaluation


class JsonlReachEvaluationStore:
    """Append-only JSONL persistence for baseline and PPO evaluation results."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(self, record: ReachEvaluationRecord | dict[str, Any]) -> None:
        validated = ReachEvaluationRecord.model_validate(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(validated.model_dump_json(exclude_none=False) + "\n")
            stream.flush()

    def read_all(self) -> list[ReachEvaluationRecord]:
        if not self.path.exists():
            return []
        records: list[ReachEvaluationRecord] = []
        with self.path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    raw = json.loads(line)
                    if not isinstance(raw, dict) or "schema_version" not in raw:
                        raise ValueError("a versioned evaluation record is required")
                    records.append(ReachEvaluationRecord.model_validate(raw))
                except Exception as exc:
                    raise ValueError(f"Invalid reach evaluation at {self.path}:{line_number}: {exc}") from exc
        return records


class TrainingRunRecord(BaseModel):
    """Training configuration and saved artifact references."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    run_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    algorithm: Literal["PPO"] = "PPO"
    training_config: PPOTrainingConfig
    task_config: PandaReachConfig
    total_timesteps: int = Field(ge=0)
    final_policy_path: str
    best_policy_path: str
    resumed_from: str | None = None


def _require_sb3() -> tuple[Any, Any, Any]:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise RuntimeError(
            "PPO support is optional; install it with `python -m pip install -e '.[rl]'`."
        ) from exc
    return PPO, EvalCallback, Monitor


def train_ppo(
    training_config: PPOTrainingConfig,
    task_config: PandaReachConfig,
    output_dir: str | Path,
    *,
    resume_from: str | Path | None = None,
) -> TrainingRunRecord:
    """Train PPO with periodic held-out evaluation and safe artifact handling."""
    PPO, EvalCallback, Monitor = _require_sb3()
    root = Path(output_dir)
    if resume_from is not None and not Path(resume_from).is_file():
        raise FileNotFoundError(f"PPO checkpoint not found: {resume_from}")
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError(f"training output directory is not empty: {root}")

    checkpoints = root / "checkpoints"
    evaluations = root / "evaluations"
    checkpoints.mkdir()
    evaluations.mkdir()
    config_path = root / "training_config.json"
    config_payload = {
        "algorithm": "PPO",
        "training": training_config.model_dump(mode="json"),
        "task": task_config.model_dump(mode="json"),
        "resume_from": str(resume_from) if resume_from is not None else None,
    }
    config_path.write_text(json.dumps(config_payload, indent=2) + "\n", encoding="utf-8")

    training_env = Monitor(PandaReachEnv(task_config), filename=str(root / "monitor.csv"), info_keywords=("is_success",))
    eval_env = Monitor(PandaReachEnv(task_config), info_keywords=("is_success",))
    callback = EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoints),
        log_path=str(evaluations),
        eval_freq=training_config.eval_freq,
        n_eval_episodes=training_config.eval_episodes,
        deterministic=True,
        render=False,
        verbose=0,
    )
    try:
        if resume_from is None:
            model = PPO(
                "MlpPolicy",
                training_env,
                learning_rate=training_config.learning_rate,
                n_steps=training_config.n_steps,
                batch_size=training_config.batch_size,
                n_epochs=training_config.n_epochs,
                gamma=training_config.gamma,
                gae_lambda=training_config.gae_lambda,
                clip_range=training_config.clip_range,
                seed=training_config.seed,
                device=training_config.device,
                policy_kwargs={"net_arch": list(training_config.hidden_sizes)},
                verbose=0,
            )
        else:
            model = PPO.load(str(resume_from), env=training_env, device=training_config.device)
            model.set_random_seed(training_config.seed)
        model.learn(
            total_timesteps=training_config.total_timesteps,
            callback=callback,
            reset_num_timesteps=resume_from is None,
            progress_bar=False,
        )
        final_path = root / "policy.zip"
        model.save(str(final_path.with_suffix("")))
        best_path = checkpoints / "best_model.zip"
        if not best_path.exists():
            # EvalCallback should always evaluate before training completes,
            # but keep an explicit usable fallback for callback edge cases.
            model.save(str(best_path.with_suffix("")))
        record = TrainingRunRecord(
            training_config=training_config,
            task_config=task_config,
            total_timesteps=int(model.num_timesteps),
            final_policy_path=str(final_path),
            best_policy_path=str(best_path),
            resumed_from=str(resume_from) if resume_from is not None else None,
        )
        (root / "training_result.json").write_text(
            record.model_dump_json(indent=2, exclude_none=False) + "\n", encoding="utf-8",
        )
        return record
    finally:
        training_env.close()
        eval_env.close()


def _evaluate_one_policy(
    env: PandaReachEnv,
    *,
    policy_name: str,
    policy_type: Literal["heuristic_ik", "ppo"],
    model: Any | None,
    seeds: tuple[int, ...],
    experiment_id: UUID,
    experience_store: JsonlExperienceStore | None,
    transition_recorder: TransitionRecorder | None,
    task_config: PandaReachConfig,
) -> PolicyEvaluation:
    episodes: list[EpisodeEvaluation] = []
    failure_counts: dict[str, int] = {}
    for seed in seeds:
        episode_id = uuid4()
        observation, _ = env.reset(seed=seed)
        episode_return = 0.0
        violations = 0
        terminated = truncated = False
        info: dict[str, Any] = {}
        for transition_index in range(task_config.max_episode_steps):
            if policy_type == "heuristic_ik":
                action = env.heuristic_action()
            else:
                action, _ = model.predict(observation, deterministic=True)
                action = np.asarray(action, dtype=np.float32)
            next_observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            violations += int(bool(info.get("constraint_violation", False)))
            if transition_recorder is not None:
                failure_reason = info.get("failure_reason")
                if truncated and not info.get("is_success"):
                    failure_reason = failure_reason or "time limit reached before reaching target"
                transition_recorder.write(TransitionRecord.from_transition(
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
                    simulation_timestep_s=float(env.sim.model.opt.timestep * env.sim.n_substeps),
                    environment_step=info.get("step_count"),
                    task_phase="REACHING",
                    task_success=info.get("is_success"),
                    task_failure_reason=failure_reason,
                ))
            observation = next_observation
            if terminated or truncated:
                break
        success = bool(info.get("is_success", False))
        failure_reason = info.get("failure_reason")
        if not success and failure_reason is None:
            failure_reason = "time limit reached before reaching target" if truncated else "episode ended without reaching target"
        error = float(info.get("distance_to_goal_m", np.inf))
        if not np.isfinite(error):
            raise FloatingPointError("evaluation produced non-finite final target error")
        outcome = EpisodeEvaluation(
            episode_id=episode_id,
            policy_name=policy_name,
            seed=seed,
            success=success,
            episode_return=episode_return,
            episode_length=transition_index + 1,
            final_target_error_m=error,
            terminated=bool(terminated),
            truncated=bool(truncated),
            failure_reason=failure_reason,
            constraint_violations=violations,
        )
        episodes.append(outcome)
        if not success:
            category = "constraint_violation" if violations else ("time_limit" if truncated else "failure")
            failure_counts[category] = failure_counts.get(category, 0) + 1
        if experience_store is not None:
            experience_store.write(ExperienceRecord(
                experiment_id=experiment_id,
                episode_id=episode_id,
                task_id="panda_reach",
                environment_id="mujoco_panda_reach",
                experiment_config={"policy": policy_name, "policy_type": policy_type},
                task_config=task_config.model_dump(mode="json"),
                controller_config={"algorithm": "PPO" if policy_type == "ppo" else "IK heuristic"},
                seed=seed,
                success=success,
                outcome="success" if success else "failure",
                episode_length=transition_index + 1,
                phase_history=("REACHING", "SUCCESS" if success else "FAILURE"),
                failure_reason=failure_reason,
            ))
    return PolicyEvaluation(
        policy_name=policy_name,
        policy_type=policy_type,
        attempted_episodes=len(episodes),
        success_rate=float(np.mean([item.success for item in episodes])),
        mean_episode_return=float(np.mean([item.episode_return for item in episodes])),
        mean_final_target_error_m=float(np.mean([item.final_target_error_m for item in episodes])),
        mean_episode_length=float(np.mean([item.episode_length for item in episodes])),
        failure_counts=failure_counts,
        constraint_violations=sum(item.constraint_violations for item in episodes),
        episodes=tuple(episodes),
    )


def evaluate_reach_policies(
    model: Any,
    task_config: PandaReachConfig,
    *,
    episodes: int,
    seed: int,
    model_path: str,
    experience_store: JsonlExperienceStore | None = None,
    transition_recorder: TransitionRecorder | None = None,
) -> ReachEvaluationRecord:
    """Evaluate heuristic and trained policies on the same held-out target seeds."""
    if episodes < 1 or not 0 <= seed <= 4_294_967_295:
        raise ValueError("episodes must be positive and seed must fit an unsigned 32-bit integer")
    evaluation_id = uuid4()
    eval_seeds = tuple((seed + index) % (4_294_967_295 + 1) for index in range(episodes))
    env = PandaReachEnv(task_config)
    try:
        env.reset(seed=seed)
        baseline = _evaluate_one_policy(
            env,
            policy_name="ik_heuristic_baseline",
            policy_type="heuristic_ik",
            model=None,
            seeds=eval_seeds,
            experiment_id=evaluation_id,
            experience_store=experience_store,
            transition_recorder=transition_recorder,
            task_config=task_config,
        )
        trained = _evaluate_one_policy(
            env,
            policy_name="ppo_policy",
            policy_type="ppo",
            model=model,
            seeds=eval_seeds,
            experiment_id=evaluation_id,
            experience_store=experience_store,
            transition_recorder=transition_recorder,
            task_config=task_config,
        )
        return ReachEvaluationRecord(
            comparison_id=evaluation_id,
            seed=seed,
            task_config=task_config,
            model_path=model_path,
            baseline=baseline,
            trained_policy=trained,
        )
    finally:
        env.close()


def load_ppo_model(path: str | Path, *, device: str = "cpu") -> Any:
    """Load a PPO checkpoint with a clear optional-dependency error."""
    PPO, _, _ = _require_sb3()
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"PPO checkpoint not found: {checkpoint}")
    return PPO.load(str(checkpoint), device=device)
