"""Controller calibration validation, estimation, decisions, and MuJoCo run."""

from uuid import UUID

import numpy as np
import pytest

from roboforge.environments.mujoco_env import MujocoRobotEnvironment
from roboforge.experiments.calibration import (
    GainEvaluation,
    JsonlCalibrationStore,
    PandaControllerCalibrator,
    TrackingCalibrationSpec,
    TrackingTrial,
    compute_tracking_metrics,
    decide_candidate,
    estimate_setpoint_gain,
)
from roboforge.experiments.transitions import JsonlTransitionStore, TransitionDataset


def gain(kp, rmse, *, settled=2, failures=0, violations=0, seeds=(0, 1)):
    trials = tuple(TrackingTrial(
        episode_id=UUID(int=index + 1),
        seed=seed,
        configured_kp=kp,
        estimated_kp=kp if kp is not None else 1.0,
        tracking_rmse_rad=rmse,
        final_rmse_rad=rmse / 2,
        settling_time_s=0.1 if index < settled else None,
        settled=index < settled,
        constraint_violation_count=violations,
        completed_steps=10,
        terminated=False,
        truncated=False,
    ) for index, seed in enumerate(seeds))
    return GainEvaluation(
        configured_kp=kp,
        estimated_kp=kp if kp is not None else 1.0,
        mean_tracking_rmse_rad=rmse,
        mean_final_rmse_rad=rmse / 2,
        settled_episodes=settled,
        failure_count=failures,
        constraint_violation_count=violations,
        trials=trials,
    )


def test_calibration_spec_rejects_unbounded_or_invalid_designs():
    with pytest.raises(ValueError, match="seven Panda joints"):
        TrackingCalibrationSpec(target_offsets_rad=(0.01,))
    with pytest.raises(ValueError, match="bounded to 0.10"):
        TrackingCalibrationSpec(target_offsets_rad=(0.2, 0, 0, 0, 0, 0, 0))
    with pytest.raises(ValueError, match="candidate gains"):
        TrackingCalibrationSpec(candidate_kp=(1.2,))
    with pytest.raises(ValueError, match="settling_window"):
        TrackingCalibrationSpec(max_steps=20, settling_window=21)


def test_least_squares_estimator_recovers_known_command_gain():
    rng = np.random.default_rng(23)
    limits = np.array([[-2.0, 2.0], [-1.0, 1.0]])
    current = rng.uniform([-1.0, -0.5], [1.0, 0.5], size=(40, 2))
    target = np.tile([0.5, -0.25], (40, 1))
    known_gain = 0.37
    commanded = current + known_gain * (target - current)
    actions = 2.0 * (commanded - limits[:, 0]) / (limits[:, 1] - limits[:, 0]) - 1.0
    estimate = estimate_setpoint_gain(current, target, actions, limits)
    assert estimate == pytest.approx(known_gain, abs=1e-10)


def test_tracking_metrics_calculate_rmse_and_persistent_settling():
    errors = np.array([[0.1, 0.2], [0.01, 0.005], [0.005, -0.003]])
    rmse, final_rmse, settling_time, settled = compute_tracking_metrics(
        errors, settling_window=2, tolerance_rad=0.02, timestep_s=0.02,
    )
    assert rmse == pytest.approx(np.sqrt(np.mean(errors**2)))
    assert final_rmse == pytest.approx(np.sqrt(np.mean(errors[-2:] ** 2)))
    assert settling_time == pytest.approx(0.04)
    assert settled

    _, _, settling_time, settled = compute_tracking_metrics(
        errors, settling_window=2, tolerance_rad=0.002, timestep_s=0.02,
    )
    assert settling_time is None and not settled


def test_tracking_metrics_reject_insufficient_or_non_finite_data():
    with pytest.raises(ValueError, match="non-empty"):
        compute_tracking_metrics(np.empty((0, 7)), settling_window=1, tolerance_rad=0.01, timestep_s=0.02)
    with pytest.raises(ValueError, match="finite"):
        compute_tracking_metrics(np.array([[np.nan]]), settling_window=1, tolerance_rad=0.01, timestep_s=0.02)


def test_gain_estimator_rejects_insufficient_or_non_finite_data():
    limits = np.tile([-1.0, 1.0], (2, 1))
    with pytest.raises(ValueError, match="insufficient informative"):
        estimate_setpoint_gain(np.zeros((2, 2)), np.ones((2, 2)), np.zeros((2, 2)), limits)
    current = np.ones((10, 2))
    target = np.zeros((10, 2))
    actions = np.full((10, 2), np.nan)
    with pytest.raises(ValueError, match="finite"):
        estimate_setpoint_gain(current, target, actions, limits)


def test_candidate_acceptance_requires_gain_and_all_safety_checks():
    baseline = gain(None, 0.02)
    improved = gain(0.5, 0.018)
    best, accepted, reason = decide_candidate(baseline, [improved], minimum_improvement_rad=0.001)
    assert best is improved
    assert accepted and "accepted" in reason

    not_improved = gain(0.6, 0.0195)
    _, accepted, reason = decide_candidate(baseline, [not_improved], minimum_improvement_rad=0.001)
    assert not accepted and "below" in reason

    unsafe = gain(0.4, 0.015, violations=1)
    _, accepted, reason = decide_candidate(baseline, [unsafe], minimum_improvement_rad=0.001)
    assert not accepted and "constraint" in reason

    unsettled = gain(0.3, 0.015, settled=1)
    _, accepted, reason = decide_candidate(baseline, [unsettled], minimum_improvement_rad=0.001)
    assert not accepted and "fewer episodes" in reason

    one_episode = gain(0.2, 0.01, seeds=(0,))
    _, accepted, reason = decide_candidate(
        baseline, [one_episode], minimum_improvement_rad=0.001,
    )
    assert not accepted and "at least two paired" in reason

    unpaired = gain(0.2, 0.01, seeds=(3, 4))
    _, accepted, reason = decide_candidate(
        baseline, [unpaired], minimum_improvement_rad=0.001,
    )
    assert not accepted and "seed-matched" in reason


def test_short_mujoco_calibration_reproducible_and_persisted(tmp_path):
    spec = TrackingCalibrationSpec(
        episodes=1,
        seed=11,
        candidate_kp=(0.5,),
        max_steps=30,
        settling_window=5,
    )
    env = MujocoRobotEnvironment(max_episode_steps=31)
    try:
        transitions_path = tmp_path / "calibration-transitions.jsonl"
        calibration_path = tmp_path / "calibrations.jsonl"
        first = PandaControllerCalibrator(env).run(
            spec, transition_recorder=JsonlTransitionStore(transitions_path),
        )
        second = PandaControllerCalibrator(env).run(
            spec, transition_recorder=JsonlTransitionStore(transitions_path),
        )
        assert first.baseline.mean_tracking_rmse_rad == second.baseline.mean_tracking_rmse_rad
        assert not first.candidate_accepted
        assert first.candidates[0].estimated_kp == pytest.approx(0.5, abs=1e-4)
        assert first.baseline.estimated_kp == pytest.approx(1.0, abs=1e-4)
        assert first.transition_count == 60

        store = JsonlCalibrationStore(calibration_path)
        store.write(first)
        assert store.read_all() == [first]
        dataset = TransitionDataset.from_jsonl(transitions_path)
        assert len(dataset.records) == 120
        assert len({record.experiment_id for record in dataset.records}) == 2
        assert all(record.transition_index == 0 for record in dataset.records if record.environment_step == 1)
    finally:
        env.close()
