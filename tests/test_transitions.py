"""Transition schema, JSONL storage, and task integration tests."""

import json
from uuid import UUID

import numpy as np
import pytest

from roboforge.experiments.transitions import (
    JsonlTransitionStore,
    TransitionDataset,
    TransitionRecord,
)
from roboforge.experiments.runner import ExperimentRunner
from roboforge.experiments.spec import ControllerConfig, ControllerType, ExperimentSpec, TaskType


EXPERIMENT_ID = UUID("00000000-0000-0000-0000-000000000001")
EPISODE_ID = UUID("00000000-0000-0000-0000-000000000002")


def transition(**overrides):
    values = {
        "experiment_id": EXPERIMENT_ID,
        "episode_id": EPISODE_ID,
        "transition_index": 0,
        "seed": 7,
        "observation": np.array([1, 2], dtype=np.float32),
        "action": np.array([-0.5, 1.0]),
        "next_observation": np.array([2, 3], dtype=np.float64),
        "reward": 0.0,
        "terminated": False,
        "truncated": True,
        "simulation_time_s": 0.02,
        "simulation_timestep_s": 0.02,
        "environment_step": 1,
        "task_phase": "APPROACH",
        "task_success": None,
        "task_failure_reason": None,
    }
    values.update(overrides)
    return TransitionRecord.from_transition(**values)


def test_transition_jsonl_round_trip_converts_numpy_and_preserves_flags(tmp_path):
    path = tmp_path / "steps.jsonl"
    store = JsonlTransitionStore(path)
    expected = transition()
    store.write(expected)

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["observation"] == [1.0, 2.0]
    assert raw["action"] == [-0.5, 1.0]
    assert raw["terminated"] is False
    assert raw["truncated"] is True
    assert store.read_all() == [expected]
    dataset = TransitionDataset.from_jsonl(path)
    assert dataset.manifest()["record_count"] == 1
    assert dataset.manifest()["episodes"][str(EPISODE_ID)]["record_count"] == 1


def test_transition_converts_numpy_scalar_and_rejects_non_finite_reward():
    assert transition(reward=np.float32(0.25)).reward == pytest.approx(0.25)
    with pytest.raises(ValueError):
        transition(reward=float("nan"))


@pytest.mark.parametrize("bad", [
    np.array([[1.0]]), np.array([np.inf]), np.array([1.0 + 2.0j]),
    np.array([]), [1.0, float("nan")], ["1"],
])
def test_transition_rejects_malformed_or_non_finite_vectors(bad):
    with pytest.raises(ValueError):
        transition(observation=bad)


def test_transition_dataset_requires_contiguous_indices_per_episode():
    with pytest.raises(ValueError, match="expected index 1"):
        TransitionDataset([transition(), transition(transition_index=2)])


def test_transition_store_reports_malformed_line(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"schema_version":1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"broken.jsonl:1"):
        JsonlTransitionStore(path).read_all()


def test_runner_enables_recording_and_shares_episode_ids(tmp_path):
    class RecordingTask:
        def configure_transition_recording(self, *, recorder, experiment_id, episode_id, seed):
            self.recorder = recorder
            self.context = (experiment_id, episode_id, seed)

        def run(self, *, seed):
            experiment_id, episode_id, configured_seed = self.context
            self.recorder.write(transition(
                experiment_id=experiment_id,
                episode_id=episode_id,
                seed=configured_seed,
            ))
            return type("Result", (), {"success": True, "grasp_verified": True})()

    spec = ExperimentSpec(
        task=TaskType.PICK_AND_PLACE,
        episodes=1,
        seed=9,
        controller=ControllerConfig(controller_type=ControllerType.JOINT_POSITION),
    )
    transitions = JsonlTransitionStore(tmp_path / "steps.jsonl")
    run = ExperimentRunner(
        RecordingTask(), spec, environment_id="test", transition_recorder=transitions,
    ).run()
    step = TransitionDataset.from_jsonl(transitions.path).records[0]
    assert step.experiment_id == run.records[0].experiment_id
    assert step.episode_id == run.records[0].episode_id
    assert step.seed == run.records[0].seed == 9


def test_runner_rejects_transition_recording_for_non_recording_task():
    spec = ExperimentSpec(
        task=TaskType.PICK_AND_PLACE,
        episodes=1,
        controller=ControllerConfig(controller_type=ControllerType.JOINT_POSITION),
    )
    with pytest.raises(TypeError, match="configure_transition_recording"):
        ExperimentRunner(
            object(), spec, environment_id="test", transition_recorder=JsonlTransitionStore("unused.jsonl"),
        ).run()
