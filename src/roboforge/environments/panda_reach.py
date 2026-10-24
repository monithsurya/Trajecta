"""Gymnasium task wrapper for continuous-control Panda end-effector reaching."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from roboforge.controllers.end_effector_position import PandaEndEffectorPositionController
from roboforge.environments.mujoco_env import MujocoRobotEnvironment


class PandaReachConfig(BaseModel):
    """Validated task bounds and reward weights for the Panda reach task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_episode_steps: int = Field(default=100, ge=1, le=2000)
    action_step_m: FiniteFloat = Field(default=0.04, gt=0, le=0.1)
    success_tolerance_m: FiniteFloat = Field(default=0.04, gt=0, le=0.2)
    progress_scale: FiniteFloat = Field(default=10.0, gt=0, le=100)
    action_penalty: FiniteFloat = Field(default=0.01, ge=0, le=1)
    success_bonus: FiniteFloat = Field(default=2.0, gt=0, le=20)
    failure_penalty: FiniteFloat = Field(default=1.0, gt=0, le=20)
    goal_low_m: tuple[FiniteFloat, FiniteFloat, FiniteFloat] = (0.42, -0.12, 0.85)
    goal_high_m: tuple[FiniteFloat, FiniteFloat, FiniteFloat] = (0.62, 0.12, 1.05)
    workspace_low_m: tuple[FiniteFloat, FiniteFloat, FiniteFloat] = (0.30, -0.30, 0.60)
    workspace_high_m: tuple[FiniteFloat, FiniteFloat, FiniteFloat] = (0.75, 0.30, 1.40)

    @model_validator(mode="after")
    def validate_bounds(self) -> "PandaReachConfig":
        low = np.asarray(self.goal_low_m)
        high = np.asarray(self.goal_high_m)
        work_low = np.asarray(self.workspace_low_m)
        work_high = np.asarray(self.workspace_high_m)
        if np.any(low >= high):
            raise ValueError("goal_low_m must be strictly below goal_high_m")
        if np.any(work_low >= work_high):
            raise ValueError("workspace_low_m must be strictly below workspace_high_m")
        if np.any(low < work_low) or np.any(high > work_high):
            raise ValueError("goal bounds must be contained in the command workspace")
        return self


def calculate_reach_reward(
    previous_distance_m: float,
    current_distance_m: float,
    action: np.ndarray,
    *,
    success: bool,
    failure: bool,
    config: PandaReachConfig,
) -> float:
    """Distance progress plus a success bonus and normalized-action penalty."""
    action_array = np.asarray(action, dtype=np.float64)
    values = np.asarray([previous_distance_m, current_distance_m], dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("distances must be finite and non-negative")
    if action_array.shape != (3,) or not np.isfinite(action_array).all():
        raise ValueError("action must be a finite three-dimensional vector")
    if np.any(np.abs(action_array) > 1.0):
        raise ValueError("action must be in [-1, 1]")
    reward = config.progress_scale * (previous_distance_m - current_distance_m)
    reward -= config.action_penalty * float(np.mean(action_array**2))
    if success:
        reward += config.success_bonus
    if failure:
        reward -= config.failure_penalty
    return float(reward)


class PandaReachEnv(gym.Env[np.ndarray, np.ndarray]):
    """Move the Panda hand to a seeded world-frame goal using Cartesian deltas.

    The policy action is a normalized XYZ delta. A bounded Cartesian target is
    solved through the existing Panda IK and joint-position controller; the
    underlying MuJoCo actuator/action contract is unchanged.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: PandaReachConfig | None = None,
        *,
        mujoco_env: MujocoRobotEnvironment | None = None,
    ) -> None:
        super().__init__()
        self.config = config or PandaReachConfig()
        self.sim = mujoco_env or MujocoRobotEnvironment(
            max_episode_steps=self.config.max_episode_steps,
        )
        self._hand_body_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        if self._hand_body_id < 0:
            raise ValueError("PandaReachEnv requires a MuJoCo body named 'hand'")
        self.controller = PandaEndEffectorPositionController(self.sim)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.sim.observation_space.shape[0] + 6,),
            dtype=np.float32,
        )
        self._goal_position: np.ndarray | None = None
        self._commanded_position: np.ndarray | None = None
        self._hand_rotation: np.ndarray | None = None
        self._observation: np.ndarray | None = None
        self._previous_distance = 0.0
        self._step_count = 0

    def _hand_position(self) -> np.ndarray:
        return self.sim.data.xpos[self._hand_body_id].copy()

    def _observation_for(self, simulator_observation: np.ndarray) -> np.ndarray:
        assert self._goal_position is not None
        goal_delta = self._goal_position - self._hand_position()
        assert self._commanded_position is not None
        command_delta = self._commanded_position - self._hand_position()
        observation = np.concatenate((
            simulator_observation,
            goal_delta.astype(np.float32),
            command_delta.astype(np.float32),
        ))
        if observation.shape != self.observation_space.shape or not np.isfinite(observation).all():
            raise FloatingPointError("Panda reach observation has invalid shape or values")
        return observation.astype(np.float32, copy=False)

    def _info(self, **extra: Any) -> dict[str, Any]:
        assert self._goal_position is not None
        hand = self._hand_position()
        return {
            "goal_position_m": self._goal_position.copy(),
            "hand_position_m": hand,
            "distance_to_goal_m": float(np.linalg.norm(self._goal_position - hand)),
            "step_count": self._step_count,
            **extra,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        simulator_observation, info = self.sim.reset(seed=seed)
        if options and "goal_position_m" in options:
            goal = np.asarray(options["goal_position_m"], dtype=np.float64)
        else:
            low = np.asarray(self.config.goal_low_m, dtype=np.float64)
            high = np.asarray(self.config.goal_high_m, dtype=np.float64)
            goal = self.np_random.uniform(low, high)
        if goal.shape != (3,) or not np.isfinite(goal).all():
            raise ValueError("goal_position_m must be a finite XYZ vector")
        if np.any(goal < self.config.goal_low_m) or np.any(goal > self.config.goal_high_m):
            raise ValueError("goal_position_m must be within configured goal bounds")
        self._goal_position = goal.copy()
        self._commanded_position = self._hand_position()
        self._hand_rotation = self.sim.data.xmat[self._hand_body_id].reshape(3, 3).copy()
        self.controller.reset()
        self._step_count = 0
        self._previous_distance = float(np.linalg.norm(self._goal_position - self._hand_position()))
        self._observation = self._observation_for(simulator_observation)
        return self._observation.copy(), {**info, **self._info(is_success=False)}

    def _failure_step(self, action: np.ndarray, reason: str) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self._observation is not None
        reward = calculate_reach_reward(
            self._previous_distance,
            self._previous_distance,
            action,
            success=False,
            failure=True,
            config=self.config,
        )
        self._step_count += 1
        info = self._info(is_success=False, failure_reason=reason, constraint_violation=True)
        return self._observation.copy(), reward, True, False, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._observation is None or self._goal_position is None or self._commanded_position is None:
            raise RuntimeError("Call reset() before stepping PandaReachEnv")
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != (3,):
            raise ValueError(f"action must have shape (3,), got {action_array.shape}")
        if not np.isfinite(action_array).all():
            raise ValueError("action must contain only finite values")
        action_array = np.clip(action_array, self.action_space.low, self.action_space.high)
        next_command = self._commanded_position + action_array.astype(np.float64) * self.config.action_step_m
        workspace_low = np.asarray(self.config.workspace_low_m)
        workspace_high = np.asarray(self.config.workspace_high_m)
        if np.any(next_command < workspace_low) or np.any(next_command > workspace_high):
            return self._failure_step(action_array, "command exceeded the configured Cartesian workspace")

        ik_result = self.controller.set_target_position(next_command, world_rotation=self._hand_rotation)
        if not ik_result.converged:
            return self._failure_step(action_array, "IK failed to reach the commanded Cartesian target")
        self._commanded_position = next_command
        base_observation_size = self.sim.observation_space.shape[0]
        simulator_action = self.controller.compute_action(self._observation[:base_observation_size])
        simulator_observation, _, base_terminated, base_truncated, base_info = self.sim.step(simulator_action)
        self._step_count += 1
        self._observation = self._observation_for(simulator_observation)
        current_distance = float(np.linalg.norm(self._goal_position - self._hand_position()))
        success = current_distance <= self.config.success_tolerance_m
        reward = calculate_reach_reward(
            self._previous_distance,
            current_distance,
            action_array,
            success=success,
            failure=False,
            config=self.config,
        )
        self._previous_distance = current_distance
        terminated = bool(success or base_terminated)
        truncated = bool(base_truncated and not success)
        info = {
            **base_info,
            **self._info(
                is_success=success,
                failure_reason=None,
                constraint_violation=False,
                ik_position_error_m=ik_result.position_error,
            ),
        }
        return self._observation.copy(), reward, terminated, truncated, info

    def heuristic_action(self) -> np.ndarray:
        """Action for the IK-guided baseline: command directly toward the goal."""
        if self._goal_position is None or self._commanded_position is None:
            raise RuntimeError("Call reset() before requesting a baseline action")
        delta = self._goal_position - self._commanded_position
        return np.clip(delta / self.config.action_step_m, -1.0, 1.0).astype(np.float32)

    def close(self) -> None:
        self.sim.close()
