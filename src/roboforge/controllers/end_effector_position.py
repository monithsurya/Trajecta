"""World-position target interface backed by MuJoCo Panda IK."""

from __future__ import annotations

import numpy as np

from roboforge.controllers.joint_position import JointPositionController
from roboforge.environments.mujoco_env import MujocoRobotEnvironment
from roboforge.environments.mujoco_ik import IKResult, MujocoPositionIK


class PandaEndEffectorPositionController:
    """Convert a world XYZ target to bounded arm actions using existing contracts.

    IK is solved on ``set_target_position``. ``compute_action`` returns the
    normal normalized environment action vector, leaving the gripper slot open.
    """

    def __init__(
        self,
        env: MujocoRobotEnvironment,
        *,
        position_tolerance: float = 0.005,
        kp: float | list[float] | np.ndarray | None = None,
    ) -> None:
        self.env = env
        self.ik = MujocoPositionIK(env, position_tolerance=position_tolerance)
        specs = {s["target"]: s for s in env.actuator_specs if s["transmission"] == "joint"}
        self.action_indices = np.array([int(specs[n]["index"]) for n in self.ik.arm_joint_names])
        self.observation_indices = np.array([
            env.observation_layout["joint_position_indices"][n] for n in self.ik.arm_joint_names
        ])
        limits = self.ik.limits
        default_action = np.zeros(env.action_space.shape, dtype=np.float32)
        # Keep the independent gripper actuator in the open state when present.
        for spec in env.actuator_specs:
            if spec["transmission"] == "tendon" and spec["target"] == "split":
                default_action[int(spec["index"])] = 1.0
        self._controller = JointPositionController(
            np.zeros(7), kp=kp, joint_limits=limits,
            observation_indices=self.observation_indices,
            action_size=env.action_space.shape[0], action_indices=self.action_indices,
            default_action=default_action,
        )
        self.result: IKResult | None = None

    @property
    def kp(self) -> np.ndarray | None:
        """Return the joint target blend gain used by the wrapped controller."""
        return self._controller.kp

    def set_target_position(
        self,
        world_position_m: np.ndarray,
        *,
        world_rotation: np.ndarray | None = None,
    ) -> IKResult:
        """Set desired hand pose in world coordinates; position is meters."""
        self.result = self.ik.solve(world_position_m, target_rotation=world_rotation)
        self._controller.set_target_positions(self.result.joint_positions)
        return self.result

    def set_joint_target(self, joint_positions_rad: np.ndarray) -> None:
        """Set a bounded intermediate arm target, useful for gradual execution."""
        q = np.asarray(joint_positions_rad, dtype=np.float64)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("joint_positions_rad must be seven finite angles")
        self._controller.set_target_positions(np.clip(q, self.ik.limits[:, 0], self.ik.limits[:, 1]))

    def compute_action(self, observation: np.ndarray) -> np.ndarray:
        return self._controller.compute_action(observation)

    def reset(self) -> None:
        self._controller.reset()
