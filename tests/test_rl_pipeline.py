"""PPO configuration, paired evaluation, experience, and checkpoint tests."""

from pathlib import Path
import importlib.util
import subprocess
import sys

import numpy as np
import pytest

from roboforge.environments.panda_reach import PandaReachConfig
from roboforge.experiments.experience import JsonlExperienceStore
from roboforge.experiments.rl import (
    JsonlReachEvaluationStore,
    PPOTrainingConfig,
    evaluate_reach_policies,
    load_ppo_model,
    train_ppo,
)
from roboforge.experiments.transitions import JsonlTransitionStore, TransitionDataset


def test_ppo_config_validates_training_and_evaluation_sizes():
    config = PPOTrainingConfig(
        total_timesteps=128, n_steps=64, batch_size=32, eval_freq=64,
    )
    assert config.total_timesteps == 128
    with pytest.raises(ValueError, match="divide n_steps"):
        PPOTrainingConfig(total_timesteps=128, n_steps=64, batch_size=24, eval_freq=64)
    with pytest.raises(ValueError, match="eval_freq"):
        PPOTrainingConfig(total_timesteps=128, n_steps=64, batch_size=32, eval_freq=256)


class ZeroPolicy:
    """Minimal deterministic continuous-action policy for evaluation metrics."""

    def predict(self, observation, *, deterministic=True):
        return np.zeros(3, dtype=np.float32), None


def test_paired_evaluation_metrics_and_existing_experience_integration(tmp_path):
    experiences = JsonlExperienceStore(tmp_path / "experiences.jsonl")
    transitions = JsonlTransitionStore(tmp_path / "transitions.jsonl")
    comparison = evaluate_reach_policies(
        ZeroPolicy(),
        PandaReachConfig(max_episode_steps=30),
        episodes=2,
        seed=101,
        model_path="synthetic-policy.zip",
        experience_store=experiences,
        transition_recorder=transitions,
    )
    assert comparison.baseline.success_rate == 1.0
    assert comparison.trained_policy.success_rate == 0.0
    assert comparison.baseline.attempted_episodes == comparison.trained_policy.attempted_episodes == 2
    assert [item.seed for item in comparison.baseline.episodes] == [
        item.seed for item in comparison.trained_policy.episodes
    ]
    assert comparison.trained_policy.failure_counts == {"time_limit": 2}
    JsonlReachEvaluationStore(tmp_path / "evaluation.jsonl").write(comparison)
    assert len(JsonlReachEvaluationStore(tmp_path / "evaluation.jsonl").read_all()) == 1
    assert len(experiences.read_all()) == 4
    transition_records = TransitionDataset.from_jsonl(transitions.path).records
    assert len(transition_records) > 0
    assert len({item.episode_id for item in transition_records}) == 4


def test_short_ppo_training_saves_and_reloads_policy(tmp_path):
    if importlib.util.find_spec("stable_baselines3") is None:
        pytest.skip("install the optional rl extra to run PPO smoke training")
    runtime_check = subprocess.run(
        [sys.executable, "-c", "import torch; import stable_baselines3"],
        capture_output=True,
        text=True,
    )
    if runtime_check.returncode:
        pytest.skip(f"PPO native runtime is unavailable: {runtime_check.stderr[-500:]}")
    output_dir = tmp_path / "ppo-run"
    result = train_ppo(
        PPOTrainingConfig(
            total_timesteps=128,
            seed=17,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            eval_freq=64,
            eval_episodes=1,
            hidden_sizes=(16,),
        ),
        PandaReachConfig(max_episode_steps=20),
        output_dir,
    )
    assert result.total_timesteps >= 128
    assert Path(result.final_policy_path).is_file()
    assert Path(result.best_policy_path).is_file()
    assert (output_dir / "evaluations" / "evaluations.npz").is_file()
    model = load_ppo_model(result.final_policy_path)
    comparison = evaluate_reach_policies(
        model, PandaReachConfig(max_episode_steps=20), episodes=1, seed=900,
        model_path=result.final_policy_path,
    )
    assert comparison.trained_policy.attempted_episodes == 1
