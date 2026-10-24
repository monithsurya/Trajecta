"""Gymnasium contract, reward, determinism, and reach-task tests."""

from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

from roboforge.environments.panda_reach import (
    PandaReachConfig,
    PandaReachEnv,
    calculate_reach_reward,
)


def test_panda_reach_environment_passes_gymnasium_checker():
    env = PandaReachEnv()
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


def test_observation_action_shapes_and_reset_reproducibility():
    env = PandaReachEnv()
    try:
        first, first_info = env.reset(seed=123)
        second, second_info = env.reset(seed=123)
        assert first.shape == env.observation_space.shape
        assert env.action_space.shape == (3,)
        assert env.observation_space.contains(first)
        assert np.isfinite(first).all()
        assert np.array_equal(first, second)
        assert np.array_equal(first_info["goal_position_m"], second_info["goal_position_m"])
        assert env.action_space.contains(np.zeros(3, dtype=np.float32))
    finally:
        env.close()


def test_time_limit_is_truncation_and_success_is_termination():
    trunc_env = PandaReachEnv(PandaReachConfig(max_episode_steps=1))
    try:
        trunc_env.reset(seed=1)
        _, _, terminated, truncated, _ = trunc_env.step(np.zeros(3, dtype=np.float32))
        assert not terminated and truncated
    finally:
        trunc_env.close()

    success_config = PandaReachConfig(
        max_episode_steps=1,
        success_tolerance_m=0.2,
        goal_low_m=(0.5, -0.05, 1.1),
        goal_high_m=(0.6, 0.05, 1.2),
    )
    success_env = PandaReachEnv(success_config)
    try:
        success_env.reset(seed=1, options={"goal_position_m": (0.55, 0.0, 1.15)})
        _, _, terminated, truncated, info = success_env.step(np.zeros(3, dtype=np.float32))
        assert terminated and not truncated
        assert info["is_success"]
    finally:
        success_env.close()


def test_heuristic_reach_baseline_solves_fixed_seed_targets():
    env = PandaReachEnv()
    try:
        for seed in (0, 1, 2):
            env.reset(seed=seed)
            for _ in range(env.config.max_episode_steps):
                _, _, terminated, truncated, info = env.step(env.heuristic_action())
                if terminated or truncated:
                    break
            assert info["is_success"]
            assert info["distance_to_goal_m"] <= env.config.success_tolerance_m
    finally:
        env.close()


def test_reward_components_and_invalid_inputs():
    config = PandaReachConfig(progress_scale=10, action_penalty=0.1, success_bonus=2, failure_penalty=1)
    reward = calculate_reach_reward(
        0.5, 0.4, np.array([1.0, 0.0, 0.0]), success=True, failure=False, config=config,
    )
    assert reward == pytest.approx(10 * 0.1 - 0.1 / 3 + 2)
    with pytest.raises(ValueError, match="finite"):
        calculate_reach_reward(
            0.5, np.nan, np.zeros(3), success=False, failure=False, config=config,
        )
