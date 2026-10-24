"""Bounded uncertainty-aware Panda reach experiments using existing runner APIs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
from typing import Any, Literal
from uuid import UUID

import numpy as np
import mujoco
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from roboforge.environments.panda_reach import PandaReachConfig, PandaReachEnv
from roboforge.experiments.experience import JsonlExperienceStore
from roboforge.experiments.runner import ExperimentRunner
from roboforge.experiments.spec import ControllerConfig, ControllerType, ExperimentSpec, TaskType
from roboforge.experiments.transitions import JsonlTransitionStore, TransitionDataset, TransitionRecord, TransitionRecorder
from roboforge.experiments.world_model import (
    EnsembleRidgeWorldModel,
    extract_reach_transitions,
    reach_state_features,
    train_world_model,
)


class ReachCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str = Field(min_length=1)
    goal_position_m: tuple[FiniteFloat, FiniteFloat, FiniteFloat]


class SelectionDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    decision_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    strategy: Literal["random", "uncertainty"]
    iteration: int = Field(ge=0)
    seed: int = Field(ge=0, le=4_294_967_295)
    candidate_generation_seed: int | None = Field(default=None, ge=0, le=4_294_967_295)
    selection_seed: int | None = Field(default=None, ge=0, le=4_294_967_295)
    scoring_reset_seed: int | None = Field(default=None, ge=0, le=4_294_967_295)
    dataset_episode_count_before: int = Field(ge=0)
    candidate_id: str
    goal_position_m: tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    score: FiniteFloat | None = None
    candidate_scores: dict[str, float]
    experiment_id: UUID | None = None
    episode_id: UUID | None = None
    success: bool | None = None
    episode_length: int | None = None
    failure_reason: str | None = None
    selection_seconds: FiniteFloat | None = Field(default=None, ge=0)
    execution_seconds: FiniteFloat | None = None
    dataset_episode_count_after: int | None = None


class JsonlDecisionStore:
    """Validated append-only log of pre-execution decisions and outcomes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(self, decision: SelectionDecision) -> None:
        validated = SelectionDecision.model_validate(decision)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(validated.model_dump_json(exclude_none=False) + "\n")
            stream.flush()


@dataclass(frozen=True)
class ReachExperimentResult:
    success: bool
    failure_reason: str | None
    episode_length: int
    phase_history: tuple[str, ...]
    final_target_error_m: float


class PandaReachExperimentTask:
    """Heuristic reach episode adapter consumed by ExperimentRunner."""

    def __init__(self, config: PandaReachConfig, goal_position_m: tuple[float, float, float]) -> None:
        self.config = config
        self.goal_position_m = goal_position_m
        self.env = PandaReachEnv(config)
        self._recorder: TransitionRecorder | None = None
        self._experiment_id: UUID | None = None
        self._episode_id: UUID | None = None
        self._seed: int | None = None

    def configure_transition_recording(
        self, *, recorder: TransitionRecorder | None, experiment_id: UUID, episode_id: UUID, seed: int | None,
    ) -> None:
        self._recorder = recorder
        self._experiment_id = experiment_id
        self._episode_id = episode_id
        self._seed = seed

    def run(self, *, seed: int | None = None) -> ReachExperimentResult:
        observation, info = self.env.reset(seed=seed, options={"goal_position_m": self.goal_position_m})
        episode_length = 0
        terminated = truncated = False
        for index in range(self.config.max_episode_steps):
            action = self.env.heuristic_action()
            following, _reward, terminated, truncated, info = self.env.step(action)
            if self._recorder is not None:
                assert self._experiment_id is not None and self._episode_id is not None
                self._recorder.write(TransitionRecord.from_transition(
                    experiment_id=self._experiment_id,
                    episode_id=self._episode_id,
                    transition_index=index,
                    seed=self._seed,
                    observation=observation,
                    action=action,
                    next_observation=following,
                    reward=_reward,
                    terminated=terminated,
                    truncated=truncated,
                    simulation_time_s=info.get("simulation_time"),
                    simulation_timestep_s=float(self.env.sim.model.opt.timestep * self.env.sim.n_substeps),
                    environment_step=info.get("step_count"),
                    task_phase="REACHING",
                    task_success=info.get("is_success"),
                    task_failure_reason=info.get("failure_reason"),
                ))
            observation = following
            episode_length = index + 1
            if terminated or truncated:
                break
        success = bool(info.get("is_success", False))
        failure = info.get("failure_reason")
        if not success and failure is None:
            failure = "time limit reached before reaching target" if truncated else "episode ended without reaching target"
        return ReachExperimentResult(
            success=success,
            failure_reason=failure,
            episode_length=episode_length,
            phase_history=("REACHING", "SUCCESS" if success else "FAILURE"),
            final_target_error_m=float(info["distance_to_goal_m"]),
        )


def generate_reach_candidates(
    config: PandaReachConfig,
    *,
    count: int,
    seed: int,
    excluded: set[str] | None = None,
    excluded_points: tuple[tuple[float, float, float], ...] = (),
) -> tuple[ReachCandidate, ...]:
    """Generate reproducible, in-bounds target candidates; IDs encode rounded targets."""
    if count < 1:
        raise ValueError("candidate count must be positive")
    low = np.asarray(config.goal_low_m, dtype=np.float64)
    high = np.asarray(config.goal_high_m, dtype=np.float64)
    if not np.all(low < high) or np.any(low < config.workspace_low_m) or np.any(high > config.workspace_high_m):
        raise ValueError("goal bounds are invalid for the configured Cartesian workspace")
    rng = np.random.default_rng(seed)
    blocked = excluded or set()
    blocked_points = []
    for item in blocked:
        try:
            blocked_points.append(np.asarray([float(value) for value in item.split(",")], dtype=np.float64))
        except ValueError:
            continue
    for item in excluded_points:
        point = np.asarray(item, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("excluded target positions must be finite XYZ vectors")
        blocked_points.append(point)
    result: list[ReachCandidate] = []
    attempts = 0
    while len(result) < count and attempts < count * 50:
        point = np.round(rng.uniform(low, high), 5)
        key = ",".join(f"{value:.5f}" for value in point)
        attempts += 1
        if key in blocked or any(np.linalg.norm(point - prior) < 0.015 for prior in blocked_points):
            continue
        blocked.add(key)
        blocked_points.append(point)
        result.append(ReachCandidate(candidate_id=key, goal_position_m=tuple(float(value) for value in point)))
    if len(result) < count:
        raise RuntimeError("could not generate enough distinct reachable targets")
    return tuple(result)


def _recorded_goal_positions(dataset: TransitionDataset, config: PandaReachConfig) -> tuple[tuple[float, float, float], ...]:
    """Recover previous target positions from recorded goal deltas and MuJoCo state."""
    first_by_episode: dict[str, TransitionRecord] = {}
    for record in dataset.records:
        if record.task_phase == "REACHING" and record.transition_index == 0:
            first_by_episode.setdefault(str(record.episode_id), record)
    if not first_by_episode:
        return ()
    env = PandaReachEnv(config)
    targets = []
    try:
        sim = env.sim
        for record in first_by_episode.values():
            observation = np.asarray(record.observation, dtype=np.float64)
            if observation.shape != (37,):
                continue
            sim.data.qpos[:] = observation[:sim.model.nq]
            sim.data.qvel[:] = observation[sim.model.nq:sim.model.nq + sim.model.nv]
            mujoco.mj_forward(sim.model, sim.data)
            goal = sim.data.xpos[env._hand_body_id] + observation[sim.model.nq + sim.model.nv:][:3]
            targets.append(tuple(float(value) for value in goal))
    finally:
        env.close()
    return tuple(targets)


def score_candidates(
    model: EnsembleRidgeWorldModel,
    config: PandaReachConfig,
    candidates: tuple[ReachCandidate, ...],
    *,
    reset_seed: int,
) -> dict[str, float]:
    """Score target candidates using only reset state, goal, action, and prior model."""
    env = PandaReachEnv(config)
    scores = {}
    try:
        for candidate in candidates:
            observation, _ = env.reset(seed=reset_seed, options={"goal_position_m": candidate.goal_position_m})
            action = env.heuristic_action()
            scores[candidate.candidate_id] = model.uncertainty(reach_state_features(observation), action)
    finally:
        env.close()
    return scores


def select_candidate(
    candidates: tuple[ReachCandidate, ...],
    *,
    strategy: Literal["random", "uncertainty"],
    seed: int,
    scores: dict[str, float] | None = None,
) -> ReachCandidate:
    """Select randomly or by prior ensemble disagreement; no outcomes are accepted."""
    if not candidates:
        raise ValueError("candidate pool must be non-empty")
    if strategy == "random":
        return candidates[int(np.random.default_rng(seed).integers(len(candidates)))]
    if scores is None or set(scores) != {candidate.candidate_id for candidate in candidates}:
        raise ValueError("uncertainty selection requires one pre-execution score per candidate")
    if any(not np.isfinite(value) or value < 0 for value in scores.values()):
        raise ValueError("candidate uncertainty scores must be finite and non-negative")
    return max(candidates, key=lambda item: (scores[item.candidate_id], item.candidate_id))


def _execute_candidate(
    candidate: ReachCandidate,
    *,
    task_config: PandaReachConfig,
    seed: int,
    experience_store: JsonlExperienceStore,
    transition_store: JsonlTransitionStore,
) -> tuple[Any, float]:
    task = PandaReachExperimentTask(task_config, candidate.goal_position_m)
    spec = ExperimentSpec(
        task=TaskType.PANDA_REACH,
        episodes=1,
        seed=seed,
        controller=ControllerConfig(controller_type=ControllerType.OPERATIONAL_SPACE),
    )
    started = time.perf_counter()
    try:
        result = ExperimentRunner(
            task,
            spec,
            environment_id="mujoco_panda_reach",
            task_config={**task_config.model_dump(mode="json"), "goal_position_m": candidate.goal_position_m},
            recorder=experience_store,
            transition_recorder=transition_store,
        ).run()
    finally:
        task.env.close()
    return result, time.perf_counter() - started


def run_active_learning(
    *,
    experience_path: str | Path,
    transition_path: str | Path,
    artifact_dir: str | Path,
    strategy: Literal["random", "uncertainty"] = "uncertainty",
    task_config: PandaReachConfig | None = None,
    seed: int = 0,
    bootstrap_episodes: int = 5,
    iterations: int = 3,
    candidate_count: int = 16,
    model_members: int = 5,
) -> dict[str, Any]:
    """Run a bounded bootstrap, selection, execution, and model-refit loop."""
    if not 0 <= seed <= 4_294_967_295:
        raise ValueError("seed must fit an unsigned 32-bit integer")
    if bootstrap_episodes < 5 or bootstrap_episodes > 100 or iterations < 0 or iterations > 100 or not 2 <= candidate_count <= 256:
        raise ValueError("bootstrap_episodes must be [5, 100], iterations [0, 100], and candidate_count [2, 256]")
    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"artifact directory must be empty: {output_dir}")
    config = task_config or PandaReachConfig(max_episode_steps=60)
    experiences = JsonlExperienceStore(experience_path)
    transitions = JsonlTransitionStore(transition_path)
    decisions = JsonlDecisionStore(output_dir / "selection_decisions.jsonl")

    def current_dataset() -> TransitionDataset:
        return TransitionDataset.from_jsonl(transition_path)

    def refresh_manifest() -> None:
        current_dataset().write_manifest(Path(transition_path).with_suffix(".manifest.json"))

    dataset = current_dataset()
    existing_episodes = {
        str(record.episode_id) for record in dataset.records if record.task_phase == "REACHING"
    }
    bootstrap_candidates = generate_reach_candidates(
        config,
        count=bootstrap_episodes,
        seed=seed,
        excluded_points=_recorded_goal_positions(dataset, config),
    )
    for index in range(min(len(existing_episodes), bootstrap_episodes), bootstrap_episodes):
        candidate = bootstrap_candidates[index]
        run, elapsed = _execute_candidate(
            candidate,
            task_config=config,
            seed=(seed + index) % (4_294_967_295 + 1),
            experience_store=experiences,
            transition_store=transitions,
        )
        record = run.records[0]
        decisions.write(SelectionDecision(
            decision_id=f"bootstrap-{index}-{candidate.candidate_id}",
            strategy="random",
            iteration=index,
            seed=(seed + index) % (4_294_967_295 + 1),
            candidate_generation_seed=seed,
            dataset_episode_count_before=len(existing_episodes) + index - min(len(existing_episodes), bootstrap_episodes),
            candidate_id=candidate.candidate_id,
            goal_position_m=candidate.goal_position_m,
            candidate_scores={},
            experiment_id=record.experiment_id,
            episode_id=record.episode_id,
            success=record.success,
            episode_length=record.episode_length,
            failure_reason=record.failure_reason,
            execution_seconds=elapsed,
            dataset_episode_count_after=len(existing_episodes) + index - min(len(existing_episodes), bootstrap_episodes) + 1,
        ))
    dataset = current_dataset()
    refresh_manifest()
    training_started = time.perf_counter()
    model, report, arrays, _splits = train_world_model(dataset, seed=seed, members=model_members)
    bootstrap_training_seconds = time.perf_counter() - training_started
    evaluation_scale = model.delta_scale.copy()
    (output_dir / "bootstrap_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    model.save(output_dir / "world_model_bootstrap.npz", provenance={
        "transition_path": str(Path(transition_path).resolve()),
        "dataset_sha256": report["dataset_sha256"],
        "training_report": report,
    })

    history = [{
        "phase": "bootstrap",
        "test": report["test"],
        "test_episode_ids": report["split_episode_ids"]["test"],
        "transition_count": len(arrays.deltas),
        "training_seconds": bootstrap_training_seconds,
    }]
    for iteration in range(iterations):
        dataset = current_dataset()
        refresh_manifest()
        arrays_now = extract_reach_transitions(dataset)
        excluded = {
            record.candidate_id for record in decisions_records(output_dir / "selection_decisions.jsonl")
        }
        excluded.update(candidate.candidate_id for candidate in bootstrap_candidates)
        selection_started = time.perf_counter()
        candidate_generation_seed = (seed + iteration + 10_000) % (4_294_967_295 + 1)
        candidates = generate_reach_candidates(
            config,
            count=candidate_count,
            seed=candidate_generation_seed,
            excluded=excluded,
            excluded_points=_recorded_goal_positions(dataset, config),
        )
        scoring_reset_seed = (seed + iteration + 20_000) % (4_294_967_295 + 1)
        scores = None if strategy == "random" else score_candidates(
            model, config, candidates, reset_seed=scoring_reset_seed,
        )
        selection_seed = (seed + iteration + 30_000) % (4_294_967_295 + 1)
        chosen = select_candidate(candidates, strategy=strategy, seed=selection_seed, scores=scores)
        selection_seconds = time.perf_counter() - selection_started
        decision_id = f"selected-{strategy}-{iteration}-{chosen.candidate_id}"
        decision = SelectionDecision(
            decision_id=decision_id,
            strategy=strategy,
            iteration=iteration,
            seed=(seed + iteration + bootstrap_episodes) % (4_294_967_295 + 1),
            candidate_generation_seed=candidate_generation_seed,
            selection_seed=selection_seed,
            scoring_reset_seed=scoring_reset_seed if strategy == "uncertainty" else None,
            dataset_episode_count_before=len(np.unique(arrays_now.episode_ids)),
            candidate_id=chosen.candidate_id,
            goal_position_m=chosen.goal_position_m,
            score=None if scores is None else scores[chosen.candidate_id],
            candidate_scores={} if scores is None else scores,
        )
        try:
            run, elapsed = _execute_candidate(
                chosen,
                task_config=config,
                seed=decision.seed,
                experience_store=experiences,
                transition_store=transitions,
            )
            result_record = run.records[0]
            decisions.write(decision.model_copy(update={
                "experiment_id": result_record.experiment_id,
                "episode_id": result_record.episode_id,
                "success": result_record.success,
                "episode_length": result_record.episode_length,
                "failure_reason": result_record.failure_reason,
                "selection_seconds": selection_seconds,
                "execution_seconds": elapsed,
                "dataset_episode_count_after": len(np.unique(extract_reach_transitions(current_dataset()).episode_ids)),
            }))
        except Exception as exc:
            decisions.write(decision.model_copy(update={
                "failure_reason": f"execution_error: {type(exc).__name__}: {exc}",
                "selection_seconds": selection_seconds,
            }))
            history.append({"phase": "selection", "iteration": iteration, "execution_error": str(exc)})
            continue
        dataset = current_dataset()
        refresh_manifest()
        training_started = time.perf_counter()
        model, report, arrays, _splits = train_world_model(
            dataset,
            seed=seed,
            members=model_members,
            fixed_validation_episodes=tuple(report["split_episode_ids"]["validation"]),
            fixed_test_episodes=tuple(report["split_episode_ids"]["test"]),
            evaluation_scale=evaluation_scale,
        )
        training_seconds = time.perf_counter() - training_started
        (output_dir / f"iteration_{iteration:03d}_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        model.save(output_dir / f"world_model_{iteration:03d}.npz", provenance={
            "transition_path": str(Path(transition_path).resolve()),
            "dataset_sha256": report["dataset_sha256"],
            "training_report": report,
        })
        history.append({
            "phase": "selection",
            "iteration": iteration,
            "candidate_id": chosen.candidate_id,
            "score": None if scores is None else scores[chosen.candidate_id],
            "success": result_record.success,
            "execution_seconds": elapsed,
            "training_seconds": training_seconds,
            "test": report["test"],
            "test_episode_ids": report["split_episode_ids"]["test"],
            "transition_count": len(arrays.deltas),
        })
    summary = {
        "schema_version": 1,
        "strategy": strategy,
        "seed": seed,
        "bootstrap_episodes": bootstrap_episodes,
        "requested_iterations": iterations,
        "history": history,
        "experience_path": str(Path(experience_path).resolve()),
        "transition_path": str(Path(transition_path).resolve()),
        "artifact_dir": str(output_dir.resolve()),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def decisions_records(path: Path) -> list[SelectionDecision]:
    if not path.exists():
        return []
    return [SelectionDecision.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines()]


def compare_selection_strategies(
    *,
    output_dir: str | Path,
    seeds: tuple[int, ...] = (0, 1, 2),
    bootstrap_episodes: int = 5,
    iterations: int = 3,
    candidate_count: int = 16,
    task_config: PandaReachConfig | None = None,
) -> dict[str, Any]:
    """Paired random-vs-uncertainty trials cloned from identical bootstrap runs."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError(f"comparison output directory must be empty: {root}")
    if not seeds:
        raise ValueError("at least one comparison seed is required")
    if any(not 0 <= seed <= 4_294_967_295 for seed in seeds):
        raise ValueError("all comparison seeds must fit an unsigned 32-bit integer")
    config = task_config or PandaReachConfig(max_episode_steps=60)
    comparisons = []
    for seed in seeds:
        seed_root = root / f"seed_{seed}"
        bootstrap_dir = seed_root / "bootstrap"
        bootstrap = run_active_learning(
            experience_path=bootstrap_dir / "experiences.jsonl",
            transition_path=bootstrap_dir / "transitions.jsonl",
            artifact_dir=bootstrap_dir,
            strategy="random",
            task_config=config,
            seed=seed,
            bootstrap_episodes=bootstrap_episodes,
            iterations=0,
            candidate_count=candidate_count,
        )
        initial_metrics = bootstrap["history"][0]["test"]
        arms = {}
        for strategy in ("random", "uncertainty"):
            arm_dir = seed_root / strategy
            experience_path = seed_root / f"{strategy}_experiences.jsonl"
            transition_path = seed_root / f"{strategy}_transitions.jsonl"
            shutil.copyfile(bootstrap_dir / "experiences.jsonl", experience_path)
            shutil.copyfile(bootstrap_dir / "transitions.jsonl", transition_path)
            run = run_active_learning(
                experience_path=experience_path,
                transition_path=transition_path,
                artifact_dir=arm_dir,
                strategy=strategy,
                task_config=config,
                seed=seed,
                bootstrap_episodes=bootstrap_episodes,
                iterations=iterations,
                candidate_count=candidate_count,
            )
            selection_history = run["history"][1:]
            evaluated_history = [item for item in run["history"] if "test" in item]
            final_history = evaluated_history[-1]
            final_test = final_history["test"]
            decisions = decisions_records(arm_dir / "selection_decisions.jsonl")
            selected_decisions = [item for item in decisions if item.decision_id.startswith("selected-")]
            arms[strategy] = {
                "initial_test_rmse": initial_metrics["learned"]["normalized_aggregate_rmse"],
                "final_test_rmse": final_test["learned"]["normalized_aggregate_rmse"],
                "test_rmse_change": (
                    final_test["learned"]["normalized_aggregate_rmse"]
                    - initial_metrics["learned"]["normalized_aggregate_rmse"]
                ),
                "initial_raw_test_rmse": initial_metrics["learned"]["aggregate_rmse"],
                "final_raw_test_rmse": final_test["learned"]["aggregate_rmse"],
                "initial_persistence_rmse": initial_metrics["persistence_baseline"]["aggregate_rmse"],
                "final_persistence_rmse": final_test["persistence_baseline"]["aggregate_rmse"],
                "test_episode_ids": final_history["test_episode_ids"],
                "experiment_count": len(selection_history),
                "completed_experiments": sum(item.episode_id is not None for item in selected_decisions),
                "transition_count": final_history["transition_count"],
                "failed_experiments": sum(
                    item.success is False or (item.failure_reason or "").startswith("execution_error:")
                    for item in selected_decisions
                ),
                "invalid_candidates": 0,
                "experiment_seconds": float(sum(item.execution_seconds or 0.0 for item in decisions)),
                "selection_seconds": float(sum(item.selection_seconds or 0.0 for item in decisions)),
                "training_seconds": float(sum(item.get("training_seconds", 0.0) for item in evaluated_history)),
                "run_summary": str((arm_dir / "run_summary.json").resolve()),
            }
        comparisons.append({
            "seed": seed,
            "initial_test_rmse": initial_metrics["learned"]["normalized_aggregate_rmse"],
            "initial_raw_test_rmse": initial_metrics["learned"]["aggregate_rmse"],
            "arms": arms,
        })
    report = {
        "schema_version": 1,
        "protocol": "paired strategies; same deterministic bootstrap target sequence and seeds per trial; fixed bootstrap validation/test episodes",
        "bootstrap_episodes": bootstrap_episodes,
        "selection_iterations": iterations,
        "candidate_count": candidate_count,
        "seeds": list(seeds),
        "comparisons": comparisons,
        "interpretation": "descriptive simulation comparison; no improvement is assumed or guaranteed",
    }
    (root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
