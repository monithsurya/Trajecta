"""Dependency-injected experiment orchestration for reusable task runners."""

from __future__ import annotations

from dataclasses import dataclass
import time
from uuid import uuid4
from typing import Any, Protocol

from roboforge.experiments.evaluation import EvaluationSummary, ExperimentEvaluator
from roboforge.experiments.experience import (
    ExperienceRecord,
    ExperienceRecorder,
    experience_from_result,
)
from roboforge.experiments.spec import ExperimentSpec, MAX_SEED_VALUE
from roboforge.experiments.transitions import TransitionRecorder


class EpisodeTask(Protocol):
    """Task execution interface expected by the experiment runner."""

    def run(self, *, seed: int | None) -> Any:
        """Run one episode and return an explicit result object."""


@dataclass(frozen=True)
class ExperimentRun:
    summary: EvaluationSummary
    records: tuple[ExperienceRecord, ...]


class ExperimentRunner:
    """Run a configured batch and inject evaluation/recording dependencies."""

    def __init__(
        self,
        task: EpisodeTask,
        spec: ExperimentSpec,
        *,
        environment_id: str,
        task_config: dict[str, Any] | None = None,
        recorder: ExperienceRecorder | None = None,
        transition_recorder: TransitionRecorder | None = None,
        evaluator: ExperimentEvaluator | None = None,
    ) -> None:
        if not environment_id.strip():
            raise ValueError("environment_id must be non-empty")
        self.task = task
        self.spec = spec
        self.environment_id = environment_id
        self.task_config = task_config
        self.recorder = recorder
        self.transition_recorder = transition_recorder
        self.evaluator = evaluator or ExperimentEvaluator()

    def run(self) -> ExperimentRun:
        results: list[Any] = []
        records: list[ExperienceRecord] = []
        experiment_config = self.spec.model_dump(mode="json")
        controller_config = self.spec.controller.model_dump(mode="json")
        experiment_id = uuid4()
        for episode_index in range(self.spec.episodes):
            seed = None if self.spec.seed is None else (self.spec.seed + episode_index) % (MAX_SEED_VALUE + 1)
            episode_id = uuid4()
            configure = getattr(self.task, "configure_transition_recording", None)
            if self.transition_recorder is not None:
                if configure is None:
                    raise TypeError("transition recording requires a task with configure_transition_recording()")
            if configure is not None:
                configure(
                    recorder=self.transition_recorder,
                    experiment_id=experiment_id,
                    episode_id=episode_id,
                    seed=seed,
                )
            started = time.monotonic()
            result = self.task.run(seed=seed)
            duration = time.monotonic() - started
            results.append(result)
            record = experience_from_result(
                result,
                task_id=self.spec.task.value,
                environment_id=self.environment_id,
                seed=seed,
                experiment_config=experiment_config,
                task_config=self.task_config,
                controller_config=controller_config,
                duration_seconds=duration,
                experiment_id=experiment_id,
                episode_id=episode_id,
            )
            records.append(record)
            if self.recorder is not None:
                self.recorder.write(record)
        return ExperimentRun(self.evaluator.evaluate(results), tuple(records))
