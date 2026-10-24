"""Explicit robot-backend boundary for simulation and future hardware adapters.

This module supplies only the MuJoCo implementation. Hardware adapters must
implement the protocol independently; the factory intentionally refuses to
instantiate hardware backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from roboforge.environments.mujoco_env import MujocoRobotEnvironment


@dataclass(frozen=True)
class BackendCapabilities:
    backend_id: str
    simulation: bool
    hardware_enabled: bool
    supports_reset: bool
    supports_joint_position_commands: bool
    supports_gripper_commands: bool
    supports_state_feedback: bool
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class BackendCommandResult:
    completed: bool
    timed_out: bool
    steps: int
    message: str | None = None


@runtime_checkable
class RobotBackend(Protocol):
    """Task-facing robot state/command contract; no physical driver is bundled."""

    @property
    def capabilities(self) -> BackendCapabilities: ...

    def read_state(self) -> dict[str, Any]: ...

    def command_joint_positions(
        self, positions_rad: dict[str, float], *, max_steps: int = 100, tolerance_rad: float = 0.02,
    ) -> BackendCommandResult: ...

    def command_gripper(
        self, opening_fraction: float, *, max_steps: int = 1, tolerance_m: float = 0.005,
    ) -> BackendCommandResult: ...

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> dict[str, Any]: ...

    def close(self) -> None: ...


class MujocoPandaBackend:
    """Bounded, validated Panda commands over the existing MuJoCo adapter."""

    def __init__(self, environment: MujocoRobotEnvironment) -> None:
        self.environment = environment
        self._arm_names = tuple(f"joint{i}" for i in range(1, 8))
        self._hand_body_id = environment.model.body("hand").id
        self._cube_body_id = environment.model.body("target_cube").id

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_id="mujoco_panda",
            simulation=True,
            hardware_enabled=False,
            supports_reset=True,
            supports_joint_position_commands=True,
            supports_gripper_commands=True,
            supports_state_feedback=True,
            limitations=(
                "state and command timing are simulated",
                "no physical sensing, watchdog, emergency stop, or collision guarantee",
            ),
        )

    def read_state(self) -> dict[str, Any]:
        env = self.environment
        state = env._state_info()
        state.update({
            "hand_position_m": tuple(float(v) for v in env.data.xpos[self._hand_body_id]),
            "cube_position_m": tuple(float(v) for v in env.data.xpos[self._cube_body_id]),
            "simulation_time_s": float(env.data.time),
        })
        return state

    def _hold_current_arm_action(self) -> np.ndarray:
        action = np.zeros(self.environment.action_space.shape, dtype=np.float32)
        current = self.environment._state_info()["joint_positions"]
        for spec in self.environment.actuator_specs:
            if spec["transmission"] != "joint" or spec["target"] not in self._arm_names:
                continue
            low, high = spec["control_range"]
            action[int(spec["index"])] = 2.0 * (current[spec["target"]] - low) / (high - low) - 1.0
        return action

    def command_joint_positions(
        self,
        positions_rad: dict[str, float],
        *,
        max_steps: int = 100,
        tolerance_rad: float = 0.02,
    ) -> BackendCommandResult:
        if not isinstance(positions_rad, dict) or not positions_rad:
            raise ValueError("positions_rad must be a non-empty joint-name mapping")
        if set(positions_rad) - set(self._arm_names):
            raise ValueError(f"joint command names must be among {self._arm_names}")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= 1000:
            raise ValueError("max_steps must be an integer in [1, 1000]")
        if not np.isfinite(tolerance_rad) or not 0 < tolerance_rad <= 0.1:
            raise ValueError("tolerance_rad must be finite and in (0, 0.1]")
        targets = {}
        for name, raw in positions_rad.items():
            if isinstance(raw, bool) or not np.isfinite(raw):
                raise ValueError(f"joint target for {name} must be finite")
            joint_id = self.environment.joint_index(name)
            value = float(raw)
            lower, upper = self.environment.model.jnt_range[joint_id]
            if self.environment.model.jnt_limited[joint_id] and not lower <= value <= upper:
                raise ValueError(f"joint target for {name} is outside [{lower}, {upper}]")
            targets[name] = value
        actuator_by_joint = {
            spec["target"]: spec for spec in self.environment.actuator_specs
            if spec["transmission"] == "joint"
        }
        missing = set(targets) - set(actuator_by_joint)
        if missing:
            raise ValueError(f"no joint position actuator for: {sorted(missing)}")
        for step in range(1, max_steps + 1):
            action = self._hold_current_arm_action()
            # Keep gripper open while a joint command is running.
            for spec in self.environment.actuator_specs:
                if spec["transmission"] == "tendon" and spec["target"] == "split":
                    action[int(spec["index"])] = 1.0
            for name, target in targets.items():
                spec = actuator_by_joint[name]
                low, high = spec["control_range"]
                index = int(spec["index"])
                action[index] = 2.0 * (target - low) / (high - low) - 1.0
            self.environment.step(action)
            positions = self.environment._state_info()["joint_positions"]
            if all(abs(positions[name] - target) <= tolerance_rad for name, target in targets.items()):
                return BackendCommandResult(True, False, step)
        return BackendCommandResult(False, True, max_steps, "joint command did not settle before max_steps")

    def command_gripper(
        self,
        opening_fraction: float,
        *,
        max_steps: int = 1,
        tolerance_m: float = 0.005,
    ) -> BackendCommandResult:
        if isinstance(opening_fraction, bool) or not np.isfinite(opening_fraction) or not 0.0 <= opening_fraction <= 1.0:
            raise ValueError("opening_fraction must be finite and in [0, 1]")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= 100:
            raise ValueError("max_steps must be an integer in [1, 100]")
        if not np.isfinite(tolerance_m) or not 0 < tolerance_m <= 0.02:
            raise ValueError("tolerance_m must be finite and in (0, 0.02]")
        gripper_index = next((
            int(spec["index"]) for spec in self.environment.actuator_specs
            if spec["transmission"] == "tendon" and spec["target"] == "split"
        ), None)
        if gripper_index is None:
            raise RuntimeError("backend does not expose the Panda split gripper actuator")
        action = self._hold_current_arm_action()
        action[gripper_index] = float(opening_fraction) * 2.0 - 1.0
        finger_names = ("finger_joint1", "finger_joint2")
        maximum_width = sum(
            float(self.environment.model.jnt_range[self.environment.joint_index(name), 1])
            for name in finger_names
        )
        target_width = opening_fraction * maximum_width
        for step in range(1, max_steps + 1):
            self.environment.step(action)
            state = self.read_state()
            current_width = float(state["gripper_state"]["opening_width"])
            if abs(current_width - target_width) <= tolerance_m:
                return BackendCommandResult(True, False, step)
        return BackendCommandResult(False, True, max_steps, "gripper command did not reach requested opening before max_steps")

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> dict[str, Any]:
        _observation, info = self.environment.reset(seed=seed, options=options)
        return info

    def close(self) -> None:
        self.environment.close()


def create_robot_backend(kind: str, *, environment: MujocoRobotEnvironment | None = None) -> RobotBackend:
    """Create only the explicitly supported simulation backend."""
    if kind != "mujoco":
        raise NotImplementedError("physical robot backends are not implemented or enabled in Trajecta")
    if environment is None:
        raise ValueError("the MuJoCo backend requires an explicitly constructed environment")
    return MujocoPandaBackend(environment)
