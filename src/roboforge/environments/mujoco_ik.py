"""MuJoCo damped-least-squares IK for the Panda hand body frame.

Positions and errors are expressed in world coordinates in meters. Joint
targets are radians, ordered by ``joint1`` through ``joint7``. The configured
end-effector frame is the existing ``hand`` body origin (the loaded Panda MJCF
contains no site in this scene); it is the palm frame, not the fingertip
midpoint. IK uses private scratch data and does not modify environment state.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from roboforge.environments.mujoco_env import MujocoRobotEnvironment


@dataclass(frozen=True)
class IKResult:
    joint_positions: np.ndarray
    converged: bool
    position_error: float
    iterations: int
    achieved_position: np.ndarray
    orientation_error: float = 0.0


class MujocoPositionIK:
    """Position-only Panda IK using MuJoCo's body Jacobian and DLS updates."""

    arm_joint_names = tuple(f"joint{i}" for i in range(1, 8))

    def __init__(
        self,
        env: MujocoRobotEnvironment,
        *,
        end_effector_body: str = "hand",
        damping: float = 0.04,
        position_tolerance: float = 0.005,
        max_iterations: int = 200,
        max_joint_step: float = 0.15,
    ) -> None:
        if damping <= 0 or position_tolerance <= 0 or max_iterations <= 0 or max_joint_step <= 0:
            raise ValueError("IK damping, tolerance, iterations, and step must be positive")
        self.env = env
        self.model = env.model
        self.body_name = end_effector_body
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, end_effector_body)
        if self.body_id < 0:
            raise ValueError(f"No MuJoCo body named {end_effector_body!r}")

        self.joint_ids = np.array([env.joint_index(n) for n in self.arm_joint_names], dtype=int)
        self.qpos_indices = self.model.jnt_qposadr[self.joint_ids].astype(int)
        self.dof_indices = self.model.jnt_dofadr[self.joint_ids].astype(int)
        if not np.all(self.model.jnt_limited[self.joint_ids]):
            raise ValueError("Every Panda arm joint must have a finite joint limit")
        self.limits = self.model.jnt_range[self.joint_ids].copy()
        specs = {s["target"]: s for s in env.actuator_specs if s["transmission"] == "joint"}
        for name, limits in zip(self.arm_joint_names, self.limits, strict=True):
            spec = specs.get(name)
            if spec is None or not np.allclose(spec["control_range"], limits, atol=1e-5):
                raise ValueError(f"No position-range-compatible actuator for arm joint {name!r}")
            # The existing Panda file uses fixed-gain, affine-bias position
            # servos: ctrl is a joint position setpoint, not torque/velocity.
            if (spec["dynamics_type"] != "mjDYN_NONE"
                    or spec["gain_type"] != "mjGAIN_FIXED"
                    or spec["bias_type"] != "mjBIAS_AFFINE"
                    or not np.isclose(spec["bias_parameters"][1], -spec["gain_parameters"][0])):
                raise ValueError(f"Actuator for {name!r} is not the expected Panda position servo")
        self.damping = float(damping)
        self.position_tolerance = float(position_tolerance)
        self.max_iterations = int(max_iterations)
        self.max_joint_step = float(max_joint_step)

    def forward_position(self, joint_positions: np.ndarray, *, qpos: np.ndarray | None = None) -> np.ndarray:
        """Evaluate hand body origin in world meters for arm joints in radians."""
        joints = self._validate_joints(joint_positions)
        data = mujoco.MjData(self.model)
        data.qpos[:] = self.env.data.qpos if qpos is None else qpos
        data.qpos[self.qpos_indices] = joints
        mujoco.mj_forward(self.model, data)
        return data.xpos[self.body_id].copy()

    def solve(
        self,
        target_position: np.ndarray,
        *,
        initial_positions: np.ndarray | None = None,
        target_rotation: np.ndarray | None = None,
    ) -> IKResult:
        """Solve world-space XYZ (meters), optionally matching a world rotation.

        ``target_rotation`` is a 3x3 matrix with end-effector axes in world
        coordinates. Position-only behavior remains the default.
        """
        target = np.asarray(target_position, dtype=np.float64)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("target_position must be a finite XYZ vector in world meters")
        rotation_target = None
        if target_rotation is not None:
            rotation_target = np.asarray(target_rotation, dtype=np.float64)
            if (rotation_target.shape != (3, 3) or not np.isfinite(rotation_target).all()
                    or not np.allclose(rotation_target.T @ rotation_target, np.eye(3), atol=1e-5)
                    or not np.isclose(np.linalg.det(rotation_target), 1.0, atol=1e-5)):
                raise ValueError("target_rotation must be a finite 3x3 rotation matrix")
        q = (self.env.data.qpos[self.qpos_indices].copy() if initial_positions is None
             else self._validate_joints(initial_positions))
        q = np.clip(q, self.limits[:, 0], self.limits[:, 1])
        data = mujoco.MjData(self.model)
        data.qpos[:] = self.env.data.qpos.copy()
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        iterations = 0
        for iterations in range(self.max_iterations + 1):
            data.qpos[self.qpos_indices] = q
            mujoco.mj_forward(self.model, data)
            error = target - data.xpos[self.body_id]
            error_norm = float(np.linalg.norm(error))
            if error_norm <= self.position_tolerance and rotation_target is None:
                break
            if iterations == self.max_iterations:
                break
            mujoco.mj_jacBody(self.model, data, jacp, jacr, self.body_id)
            jacobian = jacp[:, self.dof_indices]
            if rotation_target is not None:
                rotation_current = data.xmat[self.body_id].reshape(3, 3)
                rotation_error = 0.5 * sum(
                    (np.cross(rotation_current[:, axis], rotation_target[:, axis]) for axis in range(3)),
                    start=np.zeros(3, dtype=np.float64),
                )
                rotation_angle = float(np.arccos(np.clip(
                    (np.trace(rotation_target.T @ rotation_current) - 1.0) / 2.0, -1.0, 1.0
                )))
                if error_norm <= self.position_tolerance and rotation_angle <= 0.01:
                    break
                jacobian = np.vstack((jacobian, jacr[:, self.dof_indices]))
                error = np.concatenate((error, rotation_error))
            update = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + self.damping**2 * np.eye(jacobian.shape[0]), error
            )
            magnitude = float(np.max(np.abs(update)))
            if magnitude > self.max_joint_step:
                update *= self.max_joint_step / magnitude
            q = np.clip(q + update, self.limits[:, 0], self.limits[:, 1])
        achieved = data.xpos[self.body_id].copy()
        final_error = float(np.linalg.norm(target - achieved))
        final_orientation_error = 0.0
        if rotation_target is not None:
            rotation_current = data.xmat[self.body_id].reshape(3, 3)
            final_orientation_error = float(np.arccos(np.clip(
                (np.trace(rotation_target.T @ rotation_current) - 1.0) / 2.0, -1.0, 1.0
            )))
        converged = final_error <= self.position_tolerance and final_orientation_error <= 0.01
        return IKResult(q.copy(), converged, final_error, iterations, achieved, final_orientation_error)

    def _validate_joints(self, positions: np.ndarray) -> np.ndarray:
        q = np.asarray(positions, dtype=np.float64)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("joint positions must be seven finite angles in radians")
        return q.copy()
