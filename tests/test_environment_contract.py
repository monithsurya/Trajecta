"""Tests for the Trajecta RobotEnvironment contract and Gymnasium compliance."""

from __future__ import annotations

from typing import Any

import gymnasium.spaces as spaces
import numpy as np
import pytest

from roboforge.environments.contract import RobotEnvironment


class FakeDeterministicRobotEnv:
    """Deterministic, physics-free fake environment for contract testing."""

    def __init__(self, max_steps: int = 5, goal_pos: float = 1.0) -> None:
        self.max_steps = max_steps
        self.goal_pos = goal_pos
        self._step_count = 0
        self._state = np.array([0.0, 0.0], dtype=np.float32)
        self.is_closed = False
        self._observation_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

    @property
    def observation_space(self) -> spaces.Box:
        return self._observation_space

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if self.is_closed:
            raise RuntimeError("Cannot reset a closed environment.")
        self._step_count = 0
        if seed is not None:
            self._state = np.array([float((seed % 100) / 100.0), 0.0], dtype=np.float32)
        elif options and "initial_state" in options:
            self._state = np.array(options["initial_state"], dtype=np.float32)
        else:
            self._state = np.array([0.0, 0.0], dtype=np.float32)
        return self._state.copy(), {"step_count": self._step_count, "seed": seed, "options": options or {}}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.is_closed:
            raise RuntimeError("Cannot step in a closed environment.")
        act = float(np.clip(action[0], -1.0, 1.0))
        self._step_count += 1
        pos, vel = float(self._state[0]), float(self._state[1])
        new_vel = float(np.clip(vel + 0.1 * act, -1.0, 1.0))
        new_pos = float(np.clip(pos + new_vel, -1.0, 1.0))
        self._state = np.array([new_pos, new_vel], dtype=np.float32)
        terminated = bool(new_pos >= self.goal_pos)
        truncated = bool(self._step_count >= self.max_steps and not terminated)
        reward = 10.0 if terminated else -0.1
        return self._state.copy(), reward, terminated, truncated, {"step_count": self._step_count, "distance_to_goal": abs(self.goal_pos - new_pos)}

    def close(self) -> None:
        self.is_closed = True


class TestRobotEnvironmentContract:
    def test_satisfies_protocol_interface(self) -> None:
        assert isinstance(FakeDeterministicRobotEnv(), RobotEnvironment)

    def test_spaces_presence_and_typing(self) -> None:
        env = FakeDeterministicRobotEnv()
        assert isinstance(env.observation_space, spaces.Space)
        assert isinstance(env.action_space, spaces.Space)
        assert isinstance(env.observation_space, spaces.Box)
        assert isinstance(env.action_space, spaces.Box)
        assert env.observation_space.contains(env.observation_space.sample())
        assert env.action_space.contains(env.action_space.sample())

    def test_reset_returns_observation_and_info(self) -> None:
        env = FakeDeterministicRobotEnv()
        obs, info = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert env.observation_space.contains(obs)
        assert isinstance(info, dict)
        assert info["seed"] == 42
        assert info["step_count"] == 0

    def test_reset_with_options(self) -> None:
        env = FakeDeterministicRobotEnv()
        obs, info = env.reset(options={"initial_state": [0.5, 0.0]})
        assert np.allclose(obs, [0.5, 0.0])

    def test_step_returns_five_part_tuple(self) -> None:
        env = FakeDeterministicRobotEnv()
        env.reset()
        obs, reward, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_distinct_termination_and_truncation_conditions(self) -> None:
        # Truncation: horizon exceeded before goal
        env = FakeDeterministicRobotEnv(max_steps=3, goal_pos=100.0)
        env.reset()
        for _ in range(2):
            _, _, t, tr, _ = env.step(np.array([0.1], dtype=np.float32))
            assert not t and not tr
        _, _, t, tr, _ = env.step(np.array([0.1], dtype=np.float32))
        assert not t and tr

        # Termination: goal reached before horizon
        env2 = FakeDeterministicRobotEnv(max_steps=100, goal_pos=0.2)
        env2.reset(options={"initial_state": [0.15, 0.0]})
        _, reward, t, tr, _ = env2.step(np.array([1.0], dtype=np.float32))
        assert t and not tr and reward == 10.0

    def test_close_marks_resources_closed(self) -> None:
        env = FakeDeterministicRobotEnv()
        env.reset()
        env.close()
        assert env.is_closed
        with pytest.raises(RuntimeError, match="closed"):
            env.step(np.array([0.0], dtype=np.float32))
        with pytest.raises(RuntimeError, match="closed"):
            env.reset()
