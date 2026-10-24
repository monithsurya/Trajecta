#!/usr/bin/env python3
"""Safe, small Panda joint and gripper control demonstration.

macOS launch command from the project root:
    PYTHONPATH=src ./venv/bin/python ./venv/bin/mjpython examples/control_panda.py
"""

from __future__ import annotations

import argparse
import sys
import time

import mujoco
import numpy as np

from roboforge.controllers.joint_position import JointPositionController
from roboforge.environments.mujoco_env import MujocoRobotEnvironment
from roboforge.visualization.viewer import is_running_under_mjpython


class PandaDemoController:
    """Controller-interface adapter for a small arm cycle and slow gripper demo."""

    arm_joint_names = tuple(f"joint{i}" for i in range(1, 8))

    def __init__(self, env: MujocoRobotEnvironment, observation: np.ndarray) -> None:
        layout = env.observation_layout
        specs_by_joint = {
            spec["target"]: spec
            for spec in env.actuator_specs
            if spec["transmission"] == "joint"
        }
        missing = [name for name in self.arm_joint_names if name not in specs_by_joint]
        if missing:
            raise ValueError(f"Panda arm actuators are missing for joints: {missing}")

        self._joint_actuator_indices = [int(specs_by_joint[name]["index"]) for name in self.arm_joint_names]
        self._joint_observation_indices = [layout["joint_position_indices"][name] for name in self.arm_joint_names]
        joint_limits = []
        for name in self.arm_joint_names:
            joint_id = env.joint_index(name)
            limits = tuple(float(v) for v in env.model.jnt_range[joint_id])
            if not np.allclose(limits, specs_by_joint[name]["control_range"]):
                raise ValueError(
                    f"{name} actuator control range does not match the joint limits; "
                    "the position controller cannot safely normalize this actuator"
                )
            joint_limits.append(limits)
        gripper_actuators = [
            spec for spec in env.actuator_specs
            if spec["transmission"] == "tendon" and spec["target"] == "split"
        ]
        if len(gripper_actuators) != 1:
            raise ValueError(f"Expected one Panda split-tendon actuator, found {len(gripper_actuators)}")
        self._gripper_actuator_index = int(gripper_actuators[0]["index"])
        default_action = np.zeros(env.action_space.shape, dtype=np.float32)
        default_action[self._gripper_actuator_index] = 1.0  # Panda actuator8: 255 is open.

        home = observation[self._joint_observation_indices].astype(np.float32)
        self._home = home.copy()
        delta = np.array([0.08, -0.06, 0.04, 0.03, -0.04, 0.04, 0.05], dtype=np.float32)
        ranges = np.asarray(joint_limits, dtype=np.float32)
        self._moved = np.clip(home + delta, ranges[:, 0], ranges[:, 1])
        self._arm_controller = JointPositionController(
            target_positions=home,
            joint_limits=joint_limits,
            observation_indices=self._joint_observation_indices,
            action_size=env.action_space.shape[0],
            action_indices=self._joint_actuator_indices,
            default_action=default_action,
        )
        self._elapsed = 0.0
        self._control_period = 0.02

    def reset(self) -> None:
        self._elapsed = 0.0
        self._arm_controller.set_target_positions(self._home)
        self._arm_controller.reset()

    @staticmethod
    def _blend(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
        amount = float(np.clip(fraction, 0.0, 1.0))
        return start + amount * (end - start)

    def compute_action(self, observation: np.ndarray) -> np.ndarray:
        # Two gentle arm moves (out and back), then a one-second gripper close
        # and open ramp. The arm remains at its safe home target during the grip.
        if self._elapsed < 1.5:
            target = self._blend(self._home, self._moved, self._elapsed / 1.5)
            gripper = 1.0
        elif self._elapsed < 3.0:
            target = self._blend(self._moved, self._home, (self._elapsed - 1.5) / 1.5)
            gripper = 1.0
        elif self._elapsed < 4.0:
            target = self._home
            gripper = 1.0 - 2.0 * (self._elapsed - 3.0)
        elif self._elapsed < 5.0:
            target = self._home
            gripper = -1.0 + 2.0 * (self._elapsed - 4.0)
        else:
            target = self._home
            gripper = 1.0

        self._arm_controller.set_target_positions(target)
        action = self._arm_controller.compute_action(observation)
        action[self._gripper_actuator_index] = np.clip(gripper, -1.0, 1.0)
        self._elapsed += self._control_period
        return action

    def set_control_period(self, dt: float) -> None:
        self._control_period = dt


def run_demo(*, duration: float | None = None, no_ui: bool = False) -> None:
    if sys.platform == "darwin" and not is_running_under_mjpython():
        raise RuntimeError(
            "The passive MuJoCo viewer requires mjpython on macOS. Run:\n"
            "PYTHONPATH=src ./venv/bin/python ./venv/bin/mjpython examples/control_panda.py"
        )

    env = MujocoRobotEnvironment(max_episode_steps=100_000)
    try:
        observation, _ = env.reset()
        controller = PandaDemoController(env, observation)
        dt = float(env.model.opt.timestep * env.n_substeps)
        controller.set_control_period(dt)
        start_time = time.monotonic()

        with mujoco.viewer.launch_passive(
            env.model,
            env.data,
            show_left_ui=not no_ui,
            show_right_ui=not no_ui,
        ) as viewer:
            while viewer.is_running():
                step_start = time.monotonic()
                if duration is not None and step_start - start_time >= duration:
                    break
                action = controller.compute_action(observation)
                observation, _, terminated, truncated, _ = env.step(action)
                viewer.sync()
                remaining = dt - (time.monotonic() - step_start)
                if remaining > 0:
                    time.sleep(remaining)
                if terminated or truncated:
                    break
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=None, help="Close the demo after this many seconds.")
    parser.add_argument("--no-ui", action="store_true", help="Hide MuJoCo side panels.")
    args = parser.parse_args(argv)
    run_demo(duration=args.duration, no_ui=args.no_ui)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
