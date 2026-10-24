"""Fast predictive-model tests using synthetic transitions plus one MuJoCo loop."""

import json
from uuid import uuid4

import numpy as np
import pytest

from roboforge.experiments.transitions import TransitionDataset, TransitionRecord
from roboforge.experiments.world_model import (
    EnsembleRidgeWorldModel,
    REACH_FEATURE_INDICES,
    TransitionArrays,
    evaluate_model,
    extract_reach_transitions,
    split_episodes,
    state_delta,
    train_world_model,
)


def synthetic_arrays(seed=0):
    rng = np.random.default_rng(seed)
    episode_ids = []
    observations, actions, deltas = [], [], []
    weights = rng.normal(0.0, 0.1, size=(3, 5))
    for episode in range(10):
        for _step in range(12):
            observation = rng.normal(size=12)
            action = rng.uniform(-1, 1, size=3)
            delta = np.zeros(12)
            delta[:5] = action @ weights
            delta[5:] = rng.normal(0, 0.005, size=7)
            observations.append(observation)
            actions.append(action)
            deltas.append(delta)
            episode_ids.append(f"episode-{episode}")
    return TransitionArrays(
        np.asarray(observations), np.asarray(actions), np.asarray(deltas), np.asarray(episode_ids),
    )


def test_state_delta_wraps_arm_angles_and_validates_shapes():
    current = np.zeros(10)
    following = current.copy()
    current[0] = np.pi - 0.01
    following[0] = -np.pi + 0.01
    delta = state_delta(current, following)
    assert delta[0] == pytest.approx(0.02)
    with pytest.raises(ValueError, match="length"):
        state_delta(np.zeros(3), np.zeros(2))


def test_episode_splits_are_disjoint_and_never_split_an_episode():
    arrays = synthetic_arrays()
    split = split_episodes(arrays.episode_ids, seed=42)
    groups = [set(arrays.episode_ids[index]) for index in (split.train, split.validation, split.test)]
    assert all(groups[i].isdisjoint(groups[j]) for i in range(3) for j in range(i + 1, 3))
    assert set(np.concatenate((split.train, split.validation, split.test))) == set(range(len(arrays.episode_ids)))


def test_ridge_ensemble_trains_normalizes_serializes_and_beats_persistence(tmp_path):
    arrays = synthetic_arrays()
    split = split_episodes(arrays.episode_ids, seed=13)
    model = EnsembleRidgeWorldModel(members=5, seed=13).fit(arrays, split.train)
    metrics = evaluate_model(model, arrays, split.test)
    assert metrics["learned"]["aggregate_rmse"] < metrics["persistence_baseline"]["aggregate_rmse"]
    assert metrics["uncertainty_error_pearson_correlation"] is None or np.isfinite(
        metrics["uncertainty_error_pearson_correlation"]
    )
    observation, action = arrays.observations[split.test[0]], arrays.actions[split.test[0]]
    expected = model.predict(observation, action)
    assert np.isfinite(model.uncertainty(observation, action))
    path = tmp_path / "world-model.npz"
    model.save(path, provenance={"dataset": "synthetic", "seed": 13})
    reloaded = EnsembleRidgeWorldModel.load(path)
    np.testing.assert_allclose(reloaded.predict(observation, action), expected)
    with pytest.raises(FileExistsError):
        model.save(path, provenance={})
    with pytest.raises(ValueError, match="length"):
        reloaded.predict(np.zeros(2), action)


def test_extraction_uses_structured_reach_transition_fields():
    experiment_id = uuid4()
    records = []
    for episode in range(3):
        episode_id = uuid4()
        for index in range(2):
            current = np.zeros(37)
            current[-6:-3] = [0.1, 0.2, 0.3]
            following = current.copy()
            following[7] = 0.01 * (index + 1)
            records.append(TransitionRecord.from_transition(
                experiment_id=experiment_id,
                episode_id=episode_id,
                transition_index=index,
                observation=current,
                action=np.zeros(3),
                next_observation=following,
                terminated=index == 1,
                truncated=False,
                task_phase="REACHING",
            ))
    arrays = extract_reach_transitions(TransitionDataset(records))
    assert arrays.observations.shape == (6, 24)
    assert arrays.actions.shape == (6, 3)
    assert arrays.deltas[0, 0] == pytest.approx(0.01)


def test_train_world_model_reports_persistence_comparison():
    rng = np.random.default_rng(8)
    weights = rng.normal(0.0, 0.1, size=(3, 5))
    records = []
    experiment_by_episode = {}
    for episode in range(10):
        experiment_id, episode_id = uuid4(), uuid4()
        experiment_by_episode[str(episode)] = (experiment_id, episode_id)
        for index in range(12):
            compact = rng.normal(size=24)
            action = rng.uniform(-1.0, 1.0, size=3)
            delta = np.zeros(24)
            delta[:5] = action @ weights
            delta[5:] = rng.normal(0.0, 0.005, size=19)
            observation = np.zeros(37)
            following = np.zeros(37)
            observation[np.asarray(REACH_FEATURE_INDICES)] = compact
            following[np.asarray(REACH_FEATURE_INDICES)] = compact + delta
            records.append(TransitionRecord.from_transition(
                experiment_id=experiment_id,
                episode_id=episode_id,
                transition_index=index,
                observation=observation,
                action=action,
                next_observation=following,
                terminated=False,
                truncated=False,
                task_phase="REACHING",
            ))
    model, report, _, _ = train_world_model(TransitionDataset(records), seed=3, members=3)
    assert report["test"]["learned"]["aggregate_rmse"] < report["test"]["persistence_baseline"]["aggregate_rmse"]
    assert report["dataset_sha256"]
    assert model.coefficients is not None
    full_observation = np.zeros(37)
    full_observation[3] = 1.0
    predicted_full = model.predict_full_observation(full_observation, np.zeros(3))
    assert predicted_full.shape == (37,)
    np.testing.assert_array_equal(predicted_full[:7], full_observation[:7])


def test_reach_candidate_generation_and_selection_are_bounded_and_reproducible():
    from roboforge.environments.panda_reach import PandaReachConfig
    from roboforge.experiments.active_learning import (
        generate_reach_candidates,
        select_candidate,
        validation_supports_uncertainty,
    )

    config = PandaReachConfig()
    candidates = generate_reach_candidates(config, count=8, seed=99)
    assert candidates == generate_reach_candidates(config, count=8, seed=99)
    points = np.asarray([item.goal_position_m for item in candidates])
    assert np.all(points >= np.asarray(config.goal_low_m))
    assert np.all(points <= np.asarray(config.goal_high_m))
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    distances[np.diag_indices_from(distances)] = np.inf
    assert np.min(distances) >= 0.015
    scores = {item.candidate_id: float(index) for index, item in enumerate(candidates)}
    assert select_candidate(candidates, strategy="uncertainty", seed=8, scores=scores) == candidates[-1]
    assert select_candidate(candidates, strategy="random", seed=8) == select_candidate(
        candidates, strategy="random", seed=8,
    )
    with pytest.raises(ValueError, match="one pre-execution score"):
        select_candidate(candidates, strategy="uncertainty", seed=8, scores={})
    assert validation_supports_uncertainty({
        "learned": {"normalized_aggregate_rmse": 0.8},
        "persistence_baseline": {"normalized_aggregate_rmse": 1.0},
    })
    assert not validation_supports_uncertainty({
        "learned": {"normalized_aggregate_rmse": 1.1},
        "persistence_baseline": {"normalized_aggregate_rmse": 1.0},
    })


def test_active_learning_loop_bootstraps_selects_records_and_updates(tmp_path):
    from roboforge.experiments.active_learning import run_active_learning

    run = run_active_learning(
        experience_path=tmp_path / "experiences.jsonl",
        transition_path=tmp_path / "transitions.jsonl",
        artifact_dir=tmp_path / "run",
        seed=7,
        bootstrap_episodes=5,
        iterations=1,
        candidate_count=4,
        model_members=3,
    )
    assert len(run["history"]) == 2
    assert run["history"][0]["test_episode_ids"] == run["history"][1]["test_episode_ids"]
    assert (tmp_path / "run" / "world_model_000.npz").is_file()
    assert (tmp_path / "run" / "selection_decisions.jsonl").is_file()
    bootstrap_report = json.loads((tmp_path / "run" / "bootstrap_metrics.json").read_text())
    update_report = json.loads((tmp_path / "run" / "iteration_000_metrics.json").read_text())
    assert bootstrap_report["metric_scale"] == update_report["metric_scale"]
    dataset = TransitionDataset.from_jsonl(tmp_path / "transitions.jsonl")
    assert len({str(record.episode_id) for record in dataset.records}) == 6
    decisions = [
        json.loads(line)
        for line in (tmp_path / "run" / "selection_decisions.jsonl").read_text().splitlines()
    ]
    targets = np.asarray([item["goal_position_m"] for item in decisions])
    distances = np.linalg.norm(targets[:, None, :] - targets[None, :, :], axis=2)
    distances[np.diag_indices_from(distances)] = np.inf
    assert np.min(distances) >= 0.015
    from roboforge.experiments.world_model import load_world_model
    saved_model = load_world_model(tmp_path / "run" / "world_model_000.npz")
    continued = run_active_learning(
        experience_path=tmp_path / "experiences.jsonl",
        transition_path=tmp_path / "transitions.jsonl",
        artifact_dir=tmp_path / "continued",
        initial_model_path=tmp_path / "run" / "world_model_000.npz",
        seed=8,
        bootstrap_episodes=5,
        iterations=1,
        candidate_count=4,
        model_members=3,
    )
    assert continued["initial_model_path"].endswith("world_model_000.npz")
    continued_decisions = [
        json.loads(line)
        for line in (tmp_path / "continued" / "selection_decisions.jsonl").read_text().splitlines()
    ]
    selected = [item for item in continued_decisions if item["iteration"] == 0]
    assert selected[0]["model_version"] == saved_model.provenance["dataset_sha256"]


def test_random_and_uncertainty_benchmark_uses_identical_bootstrap_and_fixed_test(tmp_path):
    from roboforge.experiments.active_learning import compare_selection_strategies

    result = compare_selection_strategies(
        output_dir=tmp_path / "comparison",
        seeds=(3,),
        bootstrap_episodes=5,
        iterations=1,
        candidate_count=4,
    )
    arms = result["comparisons"][0]["arms"]
    assert arms["random"]["experiment_count"] == arms["uncertainty"]["experiment_count"] == 1
    assert arms["random"]["initial_test_rmse"] == arms["uncertainty"]["initial_test_rmse"]
    assert arms["random"]["test_episode_ids"] == arms["uncertainty"]["test_episode_ids"]
    assert (tmp_path / "comparison" / "comparison.json").is_file()
