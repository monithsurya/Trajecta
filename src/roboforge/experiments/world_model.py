"""Small NumPy ensemble world model trained from structured reach transitions.

The model is deliberately specific to PandaReachEnv's flat observation vector.
It learns next-observation deltas from (observation, action), with wrapped
differences for the seven Panda arm joint angles. Training is grouped by
episode so adjacent transitions never cross data splits.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from roboforge.experiments.transitions import TransitionDataset, TransitionRecord


MODEL_SCHEMA_VERSION = 1
MODEL_KIND = "panda_reach_ensemble_ridge_delta"
PANDA_QPOS_SIZE = 16
PANDA_QVEL_SIZE = 15
PANDA_BASE_OBSERVATION_SIZE = PANDA_QPOS_SIZE + PANDA_QVEL_SIZE
PANDA_REACH_OBSERVATION_SIZE = PANDA_BASE_OBSERVATION_SIZE + 6
# Panda's qpos begins with a 7-DoF freejoint (fixed in the scene): omit its
# position/quaternion and its six qvel values. Keep named arm/finger joints,
# their velocities, and the goal/command-relative XYZ task features.
REACH_FEATURE_INDICES = (
    *range(7, 16),
    *range(PANDA_QPOS_SIZE + 6, PANDA_QPOS_SIZE + 15),
    *range(PANDA_BASE_OBSERVATION_SIZE, PANDA_REACH_OBSERVATION_SIZE),
)
ARM_ANGLE_INDICES = tuple(range(7))
FEATURE_SCHEMA = "panda_joint_qpos_qvel_goal_delta_command_delta_v1"


def _finite_vector(value: Any, *, name: str, expected_size: int | None = None) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a non-empty finite one-dimensional vector")
    if expected_size is not None and vector.size != expected_size:
        raise ValueError(f"{name} must have length {expected_size}, got {vector.size}")
    return vector


def state_delta(current: Any, following: Any) -> np.ndarray:
    """Compute next-current, wrapping Panda arm angle differences to [-pi, pi]."""
    current_vector = _finite_vector(current, name="current state")
    following_vector = _finite_vector(following, name="next state", expected_size=current_vector.size)
    delta = following_vector - current_vector
    angle_count = min(len(ARM_ANGLE_INDICES), delta.size)
    indices = np.asarray(ARM_ANGLE_INDICES[:angle_count])
    delta[indices] = (delta[indices] + np.pi) % (2.0 * np.pi) - np.pi
    return delta


def reach_state_features(observation: Any) -> np.ndarray:
    """Project a full reach observation onto named scalar/task state fields."""
    vector = _finite_vector(
        observation,
        name="reach observation",
        expected_size=PANDA_REACH_OBSERVATION_SIZE,
    )
    return vector[np.asarray(REACH_FEATURE_INDICES, dtype=int)].copy()


@dataclass(frozen=True)
class TransitionArrays:
    observations: np.ndarray
    actions: np.ndarray
    deltas: np.ndarray
    episode_ids: np.ndarray


def extract_reach_transitions(dataset: TransitionDataset) -> TransitionArrays:
    """Extract only schema-compatible Panda reach transitions from JSONL records."""
    records = [
        item for item in dataset.records
        if item.task_phase == "REACHING" and len(item.action) == 3
    ]
    if not records:
        raise ValueError("dataset contains no Panda reach transitions (task_phase='REACHING')")
    observation_size = len(records[0].observation)
    if observation_size != PANDA_REACH_OBSERVATION_SIZE:
        raise ValueError(
            f"Panda reach observation schema requires {PANDA_REACH_OBSERVATION_SIZE} values, got {observation_size}"
        )
    compatible: list[TransitionRecord] = []
    for record in records:
        if len(record.observation) != observation_size or len(record.next_observation) != observation_size:
            continue
        compatible.append(record)
    if not compatible:
        raise ValueError("reach dataset has no transitions with a consistent observation shape")
    observations = np.asarray([reach_state_features(record.observation) for record in compatible], dtype=np.float64)
    actions = np.asarray([record.action for record in compatible], dtype=np.float64)
    deltas = np.stack([
        state_delta(reach_state_features(record.observation), reach_state_features(record.next_observation))
        for record in compatible
    ])
    episode_ids = np.asarray([str(record.episode_id) for record in compatible], dtype=str)
    if not (np.isfinite(observations).all() and np.isfinite(actions).all() and np.isfinite(deltas).all()):
        raise ValueError("reach dataset contains non-finite model features")
    return TransitionArrays(observations, actions, deltas, episode_ids)


@dataclass(frozen=True)
class EpisodeSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def split_episodes(episode_ids: Any, *, seed: int = 0) -> EpisodeSplit:
    """Deterministically split transitions by whole episode, never by row."""
    ids = np.asarray(episode_ids, dtype=str)
    unique = np.unique(ids)
    if unique.size < 3:
        raise ValueError("at least three distinct episodes are required for train/validation/test splits")
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    test_count = max(1, int(round(unique.size * 0.2)))
    validation_count = max(1, int(round(unique.size * 0.2)))
    if test_count + validation_count >= unique.size:
        test_count = validation_count = 1
    test_ids = set(shuffled[:test_count])
    validation_ids = set(shuffled[test_count:test_count + validation_count])
    train_ids = set(shuffled[test_count + validation_count:])
    return EpisodeSplit(
        train=np.flatnonzero(np.isin(ids, list(train_ids))),
        validation=np.flatnonzero(np.isin(ids, list(validation_ids))),
        test=np.flatnonzero(np.isin(ids, list(test_ids))),
    )


def _features(observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
    return np.concatenate((observations, actions), axis=1)


@dataclass(frozen=True)
class PredictionMetrics:
    sample_count: int
    per_feature_rmse: tuple[float, ...]
    aggregate_rmse: float
    aggregate_mae: float
    normalized_aggregate_rmse: float | None = None
    normalized_aggregate_mae: float | None = None


def prediction_metrics(
    expected: np.ndarray,
    predicted: np.ndarray,
    *,
    scale: np.ndarray | None = None,
) -> PredictionMetrics:
    expected_array = np.asarray(expected, dtype=np.float64)
    predicted_array = np.asarray(predicted, dtype=np.float64)
    if expected_array.ndim != 2 or predicted_array.shape != expected_array.shape or expected_array.shape[0] == 0:
        raise ValueError("expected and predicted must be non-empty matrices with the same shape")
    if not np.isfinite(expected_array).all() or not np.isfinite(predicted_array).all():
        raise ValueError("metric inputs must be finite")
    error = predicted_array - expected_array
    normalized_rmse = normalized_mae = None
    if scale is not None:
        scale_array = _finite_vector(scale, name="metric scale", expected_size=expected_array.shape[1])
        if np.any(scale_array <= 0):
            raise ValueError("metric scale values must be positive")
        normalized_error = error / scale_array
        normalized_rmse = float(np.sqrt(np.mean(normalized_error**2)))
        normalized_mae = float(np.mean(np.abs(normalized_error)))
    return PredictionMetrics(
        sample_count=int(expected_array.shape[0]),
        per_feature_rmse=tuple(float(value) for value in np.sqrt(np.mean(error**2, axis=0))),
        aggregate_rmse=float(np.sqrt(np.mean(error**2))),
        aggregate_mae=float(np.mean(np.abs(error))),
        normalized_aggregate_rmse=normalized_rmse,
        normalized_aggregate_mae=normalized_mae,
    )


class EnsembleRidgeWorldModel:
    """Bagged ridge regression with episode-level ensemble disagreement."""

    def __init__(self, *, members: int = 5, ridge: float = 1e-3, seed: int = 0) -> None:
        if not 3 <= members <= 64 or ridge <= 0 or not np.isfinite(ridge):
            raise ValueError("members must be in [3, 64] and ridge must be positive and finite")
        self.members = int(members)
        self.ridge = float(ridge)
        self.seed = int(seed)
        self.input_mean: np.ndarray | None = None
        self.input_scale: np.ndarray | None = None
        self.delta_mean: np.ndarray | None = None
        self.delta_scale: np.ndarray | None = None
        self.coefficients: np.ndarray | None = None
        self.observation_size: int | None = None
        self.action_size: int | None = None
        self.training_episode_ids: tuple[str, ...] = ()

    def fit(self, arrays: TransitionArrays, train_indices: np.ndarray) -> "EnsembleRidgeWorldModel":
        indices = np.asarray(train_indices, dtype=int)
        if indices.size < 3:
            raise ValueError("at least three training transitions are required")
        episode_ids = arrays.episode_ids[indices]
        distinct_episodes = np.unique(episode_ids)
        if distinct_episodes.size < 2:
            raise ValueError("training requires transitions from at least two episodes")
        x = _features(arrays.observations[indices], arrays.actions[indices])
        y = arrays.deltas[indices]
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("training features and targets must be finite")
        self.observation_size = int(arrays.observations.shape[1])
        self.action_size = int(arrays.actions.shape[1])
        self.input_mean = np.mean(x, axis=0)
        self.input_scale = np.std(x, axis=0)
        self.input_scale[self.input_scale < 1e-8] = 1.0
        self.delta_mean = np.mean(y, axis=0)
        self.delta_scale = np.std(y, axis=0)
        self.delta_scale[self.delta_scale < 1e-8] = 1.0
        normalized_x = (x - self.input_mean) / self.input_scale
        normalized_y = (y - self.delta_mean) / self.delta_scale
        normalized_x = np.column_stack((normalized_x, np.ones(normalized_x.shape[0])))
        rng = np.random.default_rng(self.seed)
        episode_rows = {episode: np.flatnonzero(episode_ids == episode) for episode in distinct_episodes}
        coefficients = []
        eps = np.eye(normalized_x.shape[1]) * self.ridge
        eps[-1, -1] = 0.0
        for _ in range(self.members):
            sampled_episodes = rng.choice(distinct_episodes, size=distinct_episodes.size, replace=True)
            rows = np.concatenate([episode_rows[episode] for episode in sampled_episodes])
            design = normalized_x[rows]
            target = normalized_y[rows]
            coefficients.append(np.linalg.solve(design.T @ design + eps, design.T @ target))
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        self.training_episode_ids = tuple(sorted(str(item) for item in distinct_episodes))
        return self

    def _normalized_features(self, observation: Any, action: Any) -> np.ndarray:
        if self.coefficients is None or self.input_mean is None or self.input_scale is None:
            raise RuntimeError("world model has not been fitted")
        obs = _finite_vector(observation, name="observation", expected_size=self.observation_size)
        act = _finite_vector(action, name="action", expected_size=self.action_size)
        vector = (np.concatenate((obs, act)) - self.input_mean) / self.input_scale
        return np.append(vector, 1.0)

    def predict_ensemble(self, observation: Any, action: Any) -> np.ndarray:
        """Return one reconstructed next observation per ensemble member."""
        if self.coefficients is None or self.delta_mean is None or self.delta_scale is None:
            raise RuntimeError("world model has not been fitted")
        design = self._normalized_features(observation, action)
        normalized_delta = np.einsum("d,mdo->mo", design, self.coefficients)
        predicted_delta = normalized_delta * self.delta_scale + self.delta_mean
        current = _finite_vector(observation, name="observation", expected_size=self.observation_size)
        predictions = current + predicted_delta
        for index in ARM_ANGLE_INDICES:
            if index < predictions.shape[1]:
                predictions[:, index] = (predictions[:, index] + np.pi) % (2.0 * np.pi) - np.pi
        if not np.isfinite(predictions).all():
            raise FloatingPointError("world model produced non-finite predictions")
        return predictions

    def predict(self, observation: Any, action: Any) -> np.ndarray:
        return np.mean(self.predict_ensemble(observation, action), axis=0)

    def uncertainty(self, observation: Any, action: Any) -> float:
        """Mean ensemble spread normalized by training target scales (dimensionless)."""
        if self.delta_scale is None:
            raise RuntimeError("world model has not been fitted")
        spread = np.std(self.predict_ensemble(observation, action), axis=0)
        return float(np.mean(spread / self.delta_scale))

    def save(self, path: str | Path, *, provenance: dict[str, Any]) -> None:
        target = Path(path)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite world model artifact: {target}")
        if self.coefficients is None:
            raise RuntimeError("world model has not been fitted")
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "model_kind": MODEL_KIND,
            "members": self.members,
            "ridge": self.ridge,
            "seed": self.seed,
            "observation_size": self.observation_size,
            "action_size": self.action_size,
            "angular_indices": list(ARM_ANGLE_INDICES),
            "feature_schema": FEATURE_SCHEMA,
            "training_episode_ids": list(self.training_episode_ids),
            "provenance": provenance,
        }
        with target.open("xb") as stream:
            np.savez_compressed(
                stream,
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                input_mean=self.input_mean,
                input_scale=self.input_scale,
                delta_mean=self.delta_mean,
                delta_scale=self.delta_scale,
                coefficients=self.coefficients,
            )

    @classmethod
    def load(cls, path: str | Path) -> "EnsembleRidgeWorldModel":
        with np.load(Path(path), allow_pickle=False) as artifact:
            metadata = json.loads(str(artifact["metadata"].item()))
            if metadata.get("schema_version") != MODEL_SCHEMA_VERSION or metadata.get("model_kind") != MODEL_KIND:
                raise ValueError("incompatible world model artifact schema or model kind")
            if metadata.get("feature_schema") != FEATURE_SCHEMA:
                raise ValueError("incompatible Panda reach feature schema")
            model = cls(members=int(metadata["members"]), ridge=float(metadata["ridge"]), seed=int(metadata["seed"]))
            model.observation_size = int(metadata["observation_size"])
            model.action_size = int(metadata["action_size"])
            model.input_mean = np.asarray(artifact["input_mean"], dtype=np.float64)
            model.input_scale = np.asarray(artifact["input_scale"], dtype=np.float64)
            model.delta_mean = np.asarray(artifact["delta_mean"], dtype=np.float64)
            model.delta_scale = np.asarray(artifact["delta_scale"], dtype=np.float64)
            model.coefficients = np.asarray(artifact["coefficients"], dtype=np.float64)
            model.training_episode_ids = tuple(metadata.get("training_episode_ids", ()))
        if not all(np.isfinite(array).all() for array in (
            model.input_mean, model.input_scale, model.delta_mean, model.delta_scale, model.coefficients,
        )):
            raise ValueError("world model artifact contains non-finite parameters")
        expected_input = model.observation_size + model.action_size
        expected_output = model.observation_size
        if (
            model.input_mean.shape != (expected_input,)
            or model.input_scale.shape != (expected_input,)
            or model.delta_mean.shape != (expected_output,)
            or model.delta_scale.shape != (expected_output,)
            or model.coefficients.shape != (model.members, expected_input + 1, expected_output)
            or np.any(model.input_scale <= 0)
            or np.any(model.delta_scale <= 0)
        ):
            raise ValueError("world model artifact parameter shapes or scales are incompatible")
        return model


def evaluate_model(
    model: EnsembleRidgeWorldModel,
    arrays: TransitionArrays,
    indices: np.ndarray,
    *,
    metric_scale: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compare learned mean prediction and persistence baseline on fixed rows."""
    if len(indices) == 0:
        raise ValueError("evaluation split is empty")
    expected = arrays.deltas[indices]
    prediction = np.stack([
        state_delta(
            arrays.observations[index],
            model.predict(arrays.observations[index], arrays.actions[index]),
        )
        for index in indices
    ])
    baseline = np.zeros_like(expected)
    uncertainty = np.asarray([
        model.uncertainty(arrays.observations[index], arrays.actions[index]) for index in indices
    ])
    errors = np.sqrt(np.mean((prediction - expected) ** 2, axis=1))
    correlation = None
    if len(errors) >= 2 and np.std(errors) > 0 and np.std(uncertainty) > 0:
        correlation = float(np.corrcoef(uncertainty, errors)[0, 1])
    scale = model.delta_scale if metric_scale is None else np.asarray(metric_scale, dtype=np.float64)
    return {
        "learned": prediction_metrics(expected, prediction, scale=scale).__dict__,
        "persistence_baseline": prediction_metrics(expected, baseline, scale=scale).__dict__,
        "uncertainty_error_pearson_correlation": correlation,
        "uncertainty_mean": float(np.mean(uncertainty)),
        "uncertainty_error_note": "ensemble disagreement vs per-transition RMSE; descriptive association, not calibration",
    }


def train_world_model(
    dataset: TransitionDataset,
    *,
    seed: int = 0,
    members: int = 5,
    ridge: float = 1e-3,
    fixed_validation_episodes: tuple[str, ...] | None = None,
    fixed_test_episodes: tuple[str, ...] | None = None,
    evaluation_scale: np.ndarray | None = None,
) -> tuple[EnsembleRidgeWorldModel, dict[str, Any], TransitionArrays, EpisodeSplit]:
    """Fit on train episodes and report validation/test metrics without leakage."""
    arrays = extract_reach_transitions(dataset)
    if fixed_validation_episodes is None and fixed_test_episodes is None:
        splits = split_episodes(arrays.episode_ids, seed=seed)
    elif fixed_validation_episodes is None or fixed_test_episodes is None:
        raise ValueError("fixed validation and test episode IDs must be supplied together")
    else:
        all_ids = set(np.unique(arrays.episode_ids).tolist())
        validation_ids = set(fixed_validation_episodes)
        test_ids = set(fixed_test_episodes)
        if validation_ids & test_ids or not validation_ids or not test_ids:
            raise ValueError("fixed validation/test episode IDs must be non-empty and disjoint")
        if not (validation_ids | test_ids).issubset(all_ids):
            raise ValueError("fixed validation/test episode IDs are absent from the dataset")
        train_ids = all_ids - validation_ids - test_ids
        if len(train_ids) < 2:
            raise ValueError("fixed split must leave at least two training episodes")
        splits = EpisodeSplit(
            train=np.flatnonzero(np.isin(arrays.episode_ids, list(train_ids))),
            validation=np.flatnonzero(np.isin(arrays.episode_ids, list(validation_ids))),
            test=np.flatnonzero(np.isin(arrays.episode_ids, list(test_ids))),
        )
    model = EnsembleRidgeWorldModel(members=members, ridge=ridge, seed=seed)
    model.fit(arrays, splits.train)
    metric_scale = model.delta_scale if evaluation_scale is None else np.asarray(evaluation_scale, dtype=np.float64)
    if metric_scale.shape != model.delta_scale.shape or not np.isfinite(metric_scale).all() or np.any(metric_scale <= 0):
        raise ValueError("evaluation_scale must be a finite positive vector matching the prediction features")
    reports = {
        "schema_version": 1,
        "model_kind": MODEL_KIND,
        "seed": seed,
        "metric_scale_source": "bootstrap training targets" if evaluation_scale is not None else "current training targets",
        "metric_scale": metric_scale.tolist(),
        "counts": {key: int(len(getattr(splits, key))) for key in ("train", "validation", "test")},
        "episode_counts": {
            key: int(np.unique(arrays.episode_ids[getattr(splits, key)]).size)
            for key in ("train", "validation", "test")
        },
        "validation": evaluate_model(model, arrays, splits.validation, metric_scale=metric_scale),
        "test": evaluate_model(model, arrays, splits.test, metric_scale=metric_scale),
        "split_episode_ids": {
            key: sorted(np.unique(arrays.episode_ids[getattr(splits, key)]).tolist())
            for key in ("train", "validation", "test")
        },
        "dataset_sha256": hashlib.sha256(
            "".join(record.model_dump_json() + "\n" for record in dataset.records).encode("utf-8")
        ).hexdigest(),
    }
    return model, reports, arrays, splits
