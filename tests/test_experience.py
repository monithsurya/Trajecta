"""Synthetic unit coverage for experiment evaluation and JSONL records."""

from dataclasses import dataclass
import json
from pathlib import Path

import pytest

from roboforge.experiments.evaluation import ExperimentEvaluator
from roboforge.experiments.experience import (
    ExperienceRecord,
    JsonlExperienceStore,
    experience_from_result,
)
from roboforge.experiments.runner import ExperimentRunner
from roboforge.experiments.spec import ControllerConfig, ControllerType, ExperimentSpec, TaskType


@dataclass
class SyntheticResult:
    success: bool
    episode_length: int | None = None
    final_placement_error_m: float | None = None
    phase_history: tuple[str, ...] = ()
    failure_reason: str | None = None
    grasp_verified: bool | None = None


def test_evaluator_handles_empty_batch_and_missing_measurements():
    summary = ExperimentEvaluator().evaluate([])
    assert summary.attempted_episodes == 0
    assert summary.successful_episodes == 0
    assert summary.success_rate is None
    assert summary.mean_episode_length is None
    assert summary.median_episode_length is None
    assert summary.mean_placement_error_m is None
    assert summary.failure_counts == {}

    summary = ExperimentEvaluator().evaluate([SyntheticResult(True), SyntheticResult(False)])
    assert summary.success_rate == 0.5
    assert summary.mean_episode_length is None
    assert summary.mean_placement_error_m is None


def test_evaluator_metrics_and_failure_categories():
    results = [
        SyntheticResult(True, 10, 0.02, ("PLACE", "SUCCESS")),
        SyntheticResult(False, 20, 0.08, ("PLACE", "FAILURE"), "cube slipped"),
        SyntheticResult(False, 30, None, ("DESCEND", "FAILURE")),
    ]
    summary = ExperimentEvaluator().evaluate(results)
    assert summary.attempted_episodes == 3
    assert summary.successful_episodes == 1
    assert summary.success_rate == pytest.approx(1 / 3)
    assert summary.mean_episode_length == 20
    assert summary.median_episode_length == 20
    assert summary.mean_placement_error_m == pytest.approx(0.05)
    assert summary.failure_counts == {"cube slipped": 1, "phase:DESCEND": 1}


def test_placement_error_is_not_measured_before_placement():
    result = SyntheticResult(False, 4, 0.5, ("APPROACH", "FAILURE"), "IK failure")
    assert ExperimentEvaluator().evaluate([result]).mean_placement_error_m is None


def test_evaluator_rejects_success_without_reported_grasp_verification():
    result = SyntheticResult(success=True, grasp_verified=False)
    with pytest.raises(ValueError, match="grasp verification"):
        ExperimentEvaluator().evaluate([result])


def test_record_round_trip_and_append_multiple(tmp_path: Path):
    store = JsonlExperienceStore(tmp_path / "nested" / "episodes.jsonl")
    first = ExperienceRecord(
        task_id="pick_and_place", environment_id="mujoco_panda_scene", seed=7,
        success=True, episode_length=570, duration_seconds=1.25,
        initial_cube_position_m=(0.45, 0, 0.725),
        final_cube_position_m=(0.64, -0.15, 0.725), placement_error_m=0.01,
        phase_history=("PLACE", "SUCCESS"),
    )
    second = experience_from_result(
        SyntheticResult(False, 20, None, ("DESCEND", "FAILURE"), "IK timeout"),
        task_id="pick_and_place", environment_id="mujoco_panda_scene", seed=8,
    )
    store.write(first)
    store.write(second)
    lines = store.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all(isinstance(json.loads(line), dict) for line in lines)
    read_back = store.read_all()
    assert read_back == [first, second]
    assert read_back[1].placement_error_m is None
    assert read_back[0].outcome == "success"
    assert read_back[1].outcome == "failure"
    assert read_back[0].schema_version == 1


def test_malformed_jsonl_has_path_and_line_number(tmp_path: Path):
    path = tmp_path / "corrupt.jsonl"
    valid = ExperienceRecord(task_id="task", environment_id="env", success=True)
    path.write_text(valid.model_dump_json() + "\n{invalid\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"corrupt.jsonl:2"):
        JsonlExperienceStore(path).read_all()


def test_unknown_schema_version_is_rejected_on_read(tmp_path: Path):
    path = tmp_path / "future.jsonl"
    path.write_text(json.dumps({"schema_version": 2}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        JsonlExperienceStore(path).read_all()


def test_missing_schema_version_is_rejected_on_read(tmp_path: Path):
    path = tmp_path / "unversioned.jsonl"
    path.write_text(json.dumps({"task_id": "task", "environment_id": "env", "success": True}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        JsonlExperienceStore(path).read_all()


def test_record_factory_preserves_only_available_task_fields():
    record = experience_from_result(
        SyntheticResult(False, phase_history=("VERIFY_GRASP", "FAILURE"), failure_reason="no contact"),
        task_id="pick_and_place",
        environment_id="synthetic-test-environment",
    )
    assert not record.success
    assert record.grasp_verified is None
    assert record.episode_length is None
    assert record.final_cube_position_m is None
    assert record.placement_error_m is None
    assert record.failure_reason == "no contact"


def test_record_factory_preserves_explicit_grasp_verification():
    result = SyntheticResult(success=True, grasp_verified=True)
    record = experience_from_result(
        result, task_id="pick_and_place", environment_id="synthetic-test-environment"
    )
    assert record.success and record.grasp_verified


class FakeBatchTask:
    def __init__(self):
        self.seeds = []

    def run(self, *, seed):
        self.seeds.append(seed)
        return SyntheticResult(
            success=seed == 10,
            episode_length=5 if seed == 10 else 7,
            failure_reason=None if seed == 10 else "synthetic failure",
        )


class MemoryRecorder:
    def __init__(self):
        self.records = []

    def write(self, record):
        self.records.append(record)


def test_runner_injects_task_evaluator_and_recorder(tmp_path: Path):
    spec = ExperimentSpec(
        task=TaskType.PICK_AND_PLACE,
        episodes=2,
        seed=10,
        controller=ControllerConfig(controller_type=ControllerType.JOINT_POSITION),
    )
    task = FakeBatchTask()
    recorder = MemoryRecorder()
    run = ExperimentRunner(
        task, spec, environment_id="synthetic-test-environment", recorder=recorder,
    ).run()
    assert task.seeds == [10, 11]
    assert len(recorder.records) == 2
    assert run.summary.attempted_episodes == 2
    assert run.summary.successful_episodes == 1
    assert run.summary.failure_counts == {"synthetic failure": 1}
    assert all(record.task_id == "pick_and_place" for record in run.records)
    assert all(record.experiment_config["episodes"] == 2 for record in run.records)
