"""Contact-verified Panda pick-and-place task state machine.

Task coordinates are world-frame meters. Arm targets are solved by the
injected position controller and executed through the environment action API.
The cube is never repositioned by task code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

import mujoco
import numpy as np


class TaskPhase(str, Enum):
    APPROACH = "APPROACH"
    ALIGN = "ALIGN"
    DESCEND = "DESCEND"
    CLOSE_GRIPPER = "CLOSE_GRIPPER"
    VERIFY_GRASP = "VERIFY_GRASP"
    LIFT = "LIFT"
    TRANSPORT = "TRANSPORT"
    PLACE = "PLACE"
    RELEASE = "RELEASE"
    VERIFY_PLACEMENT = "VERIFY_PLACEMENT"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


@dataclass(frozen=True)
class PickPlaceConfig:
    cube_target_position_m: tuple[float, float, float] | None = None
    destination_xy_m: tuple[float, float] = (0.65, -0.15)
    cube_body_name: str = "target_cube"
    cube_geom_name: str = "cube_geom"
    table_geom_name: str = "table"
    safe_approach_height_m: float = 0.12
    lift_height_m: float = 0.12
    retreat_height_m: float = 0.12
    hand_to_fingertip_m: float = 0.103
    position_tolerance_m: float = 0.015
    placement_tolerance_m: float = 0.045
    minimum_lift_m: float = 0.025
    minimum_contact_force_n: float = 0.01
    minimum_opening_m: float = 0.065
    motion_duration_s: float = 1.5
    close_settle_s: float = 0.8
    grasp_verify_s: float = 0.2
    release_duration_s: float = 0.8
    placement_settle_s: float = 0.4
    max_joint_step_rad: float = 0.025
    phase_timeouts_s: Mapping[str, float] = field(default_factory=lambda: {
        "APPROACH": 8.0, "ALIGN": 5.0, "DESCEND": 6.0,
        "CLOSE_GRIPPER": 3.0, "VERIFY_GRASP": 2.0, "LIFT": 6.0,
        "TRANSPORT": 8.0, "PLACE": 6.0, "RELEASE": 3.0,
        "VERIFY_PLACEMENT": 5.0,
    })

    def __post_init__(self) -> None:
        if len(self.destination_xy_m) != 2 or not np.isfinite(self.destination_xy_m).all():
            raise ValueError("destination_xy_m must contain two finite world coordinates")
        if self.cube_target_position_m is not None and (
            len(self.cube_target_position_m) != 3 or not np.isfinite(self.cube_target_position_m).all()
        ):
            raise ValueError("cube_target_position_m must be three finite world coordinates")
        for name in (
            "safe_approach_height_m", "lift_height_m", "retreat_height_m",
            "hand_to_fingertip_m", "position_tolerance_m", "placement_tolerance_m",
            "minimum_lift_m", "minimum_contact_force_n", "minimum_opening_m",
            "motion_duration_s", "close_settle_s", "grasp_verify_s",
            "release_duration_s", "placement_settle_s", "max_joint_step_rad",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite value")
        missing = {phase.value for phase in TaskPhase if phase not in (TaskPhase.SUCCESS, TaskPhase.FAILURE)} - set(self.phase_timeouts_s)
        if missing:
            raise ValueError(f"phase_timeouts_s is missing phases: {sorted(missing)}")
        if any(not np.isfinite(v) or v <= 0 for v in self.phase_timeouts_s.values()):
            raise ValueError("all phase timeouts must be positive and finite")


@dataclass(frozen=True)
class PickPlaceResult:
    success: bool
    failure_reason: str | None
    phase_history: tuple[str, ...]
    episode_length: int
    initial_cube_position_m: tuple[float, float, float]
    final_cube_position_m: tuple[float, float, float]
    final_placement_error_m: float | None
    grasp_verified: bool


@dataclass(frozen=True)
class TaskStep:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    phase: TaskPhase
    result: PickPlaceResult | None


class PickPlaceTask:
    """Incremental task runner, injected with an environment and controller."""

    def __init__(self, env: Any, controller: Any, config: PickPlaceConfig | None = None) -> None:
        self.env = env
        self.controller = controller
        self.config = config or PickPlaceConfig()
        self.result: PickPlaceResult | None = None
        self.phase = TaskPhase.FAILURE
        self.phase_history: list[str] = []
        self.episode_length = 0
        self.phase_elapsed = 0.0
        self._obs: np.ndarray | None = None
        self._target_q: np.ndarray | None = None
        self._phase_start_q: np.ndarray | None = None
        self._phase_target: np.ndarray | None = None
        self._grasp_verified = False
        self._grasp_confirmed_at: float | None = None
        self._cube_initial: np.ndarray | None = None
        self._cube_position_before_lift: np.ndarray | None = None

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        """Reset to the model home key and initialize targets from live geometry."""
        self._obs, _ = self.env.reset(seed=seed)
        self.controller.reset()
        model, data = self.env.model, self.env.data
        self._cube_body_id = self._named_id(mujoco.mjtObj.mjOBJ_BODY, self.config.cube_body_name)
        self._cube_geom_id = self._named_id(mujoco.mjtObj.mjOBJ_GEOM, self.config.cube_geom_name)
        self._table_geom_id = self._named_id(mujoco.mjtObj.mjOBJ_GEOM, self.config.table_geom_name)
        self._cube_body = int(model.geom_bodyid[self._cube_geom_id])
        self._left_body = self._named_id(mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self._right_body = self._named_id(mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        self._hand_body = self.controller.ik.body_id
        self._cube_half_height = float(model.geom_size[self._cube_geom_id, 2])
        self._table_top = float(model.geom_pos[self._table_geom_id, 2] + model.geom_size[self._table_geom_id, 2])
        self._cube_initial = data.xpos[self._cube_body_id].copy()
        if self.config.cube_target_position_m is not None:
            configured_cube_target = np.asarray(self.config.cube_target_position_m, dtype=np.float64)
            if np.linalg.norm(self._cube_initial - configured_cube_target) > self.config.position_tolerance_m:
                raise ValueError(
                    "cube_target_position_m does not match the cube's actual reset position; "
                    "the task will not move the cube to satisfy it"
                )
        self._destination = np.array([
            self.config.destination_xy_m[0], self.config.destination_xy_m[1],
            self._table_top + self._cube_half_height,
        ], dtype=np.float64)
        table_center = model.geom_pos[self._table_geom_id]
        table_half_size = model.geom_size[self._table_geom_id]
        for axis in (0, 1):
            lower = table_center[axis] - table_half_size[axis] + self._cube_half_height
            upper = table_center[axis] + table_half_size[axis] - self._cube_half_height
            if not lower <= self._destination[axis] <= upper:
                raise ValueError("destination_xy_m must keep the cube fully within the table footprint")
        self._hand_rotation = data.xmat[self._hand_body].reshape(3, 3).copy()
        self._arm_qpos_indices = self.controller.ik.qpos_indices.copy()
        self._arm_dof_indices = self.controller.ik.dof_indices.copy()
        self._gripper_action_index = self._find_gripper_action()
        self._grasp_verified = False
        self._grasp_confirmed_at = None
        self._cube_position_before_lift = None
        self.result = None
        self.phase_history = []
        self.episode_length = 0
        self._enter_phase(TaskPhase.APPROACH)
        return self._obs.copy()

    def _named_id(self, kind: mujoco.mjtObj, name: str) -> int:
        result = mujoco.mj_name2id(self.env.model, kind, name)
        if result < 0:
            raise ValueError(f"PickPlaceTask requires a model object named {name!r}")
        return result

    def _find_gripper_action(self) -> int:
        for spec in self.env.actuator_specs:
            if spec["transmission"] == "tendon" and spec["target"] == "split":
                return int(spec["index"])
        raise ValueError("PickPlaceTask requires the Panda 'split' tendon actuator")

    def _cube_position(self) -> np.ndarray:
        return self.env.data.xpos[self._cube_body_id].copy()

    def _hand_target(self, cube_center: np.ndarray, *, extra_height: float = 0.0) -> np.ndarray:
        return cube_center + np.array([
            0.0, 0.0, self.config.hand_to_fingertip_m + extra_height,
        ])

    def _phase_target_for(self, phase: TaskPhase) -> np.ndarray | None:
        cube = self._cube_initial
        if phase in (TaskPhase.APPROACH, TaskPhase.ALIGN, TaskPhase.DESCEND):
            return self._hand_target(cube, extra_height=(
                self.config.safe_approach_height_m
                if phase in (TaskPhase.APPROACH, TaskPhase.ALIGN) else 0.0
            ))
        if phase == TaskPhase.LIFT:
            return self._hand_target(self._cube_position_before_lift, extra_height=self.config.lift_height_m)
        if phase == TaskPhase.TRANSPORT:
            return self._hand_target(self._destination, extra_height=self.config.lift_height_m)
        if phase == TaskPhase.PLACE:
            return self._hand_target(self._destination)
        if phase == TaskPhase.VERIFY_PLACEMENT:
            return self._hand_target(self._destination, extra_height=self.config.retreat_height_m)
        return None

    def _enter_phase(self, phase: TaskPhase) -> None:
        self.phase = phase
        self.phase_elapsed = 0.0
        self.phase_history.append(phase.value)
        self._phase_target = self._phase_target_for(phase)
        self._target_q = None
        self._phase_start_q = self.env.data.qpos[self._arm_qpos_indices].copy() if hasattr(self, "_arm_qpos_indices") else None
        if self._phase_target is not None:
            solution = self.controller.set_target_position(
                self._phase_target, world_rotation=self._hand_rotation
            )
            if not solution.converged:
                self._finish(False, f"IK did not converge for {phase.value}: position error {solution.position_error:.4f} m, orientation error {solution.orientation_error:.3f} rad")
                return
            self._target_q = solution.joint_positions.copy()
            self._phase_start_q = self.env.data.qpos[self._arm_qpos_indices].copy()

    def _contact_forces(self) -> dict[int, float]:
        forces = {self._left_body: 0.0, self._right_body: 0.0}
        data, model = self.env.data, self.env.model
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            geom_a, geom_b = int(contact.geom1), int(contact.geom2)
            body_a, body_b = int(model.geom_bodyid[geom_a]), int(model.geom_bodyid[geom_b])
            if self._cube_body not in (body_a, body_b):
                continue
            finger_body = body_b if body_a == self._cube_body else body_a
            if finger_body not in forces:
                continue
            wrench = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, contact_id, wrench)
            forces[finger_body] = max(forces[finger_body], float(wrench[0]))
        return forces

    def _bilateral_grasp_contacts(self) -> bool:
        forces = self._contact_forces()
        return all(force >= self.config.minimum_contact_force_n for force in forces.values())

    def _table_supports_cube(self) -> bool:
        data, model = self.env.data, self.env.model
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            bodies = (int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2]))
            geoms = (int(contact.geom1), int(contact.geom2))
            if self._cube_body in bodies and self._table_geom_id in geoms:
                return True
        return False

    def _finish(self, success: bool, reason: str | None = None) -> None:
        final = self._cube_position()
        placement_phases = {TaskPhase.PLACE.value, TaskPhase.RELEASE.value, TaskPhase.VERIFY_PLACEMENT.value}
        placement_attempted = self.phase.value in placement_phases or any(
            phase in placement_phases for phase in self.phase_history
        )
        placement_error = (
            float(np.linalg.norm(final[:2] - self._destination[:2]))
            if placement_attempted else None
        )
        phase = TaskPhase.SUCCESS if success else TaskPhase.FAILURE
        self.phase = phase
        self.phase_history.append(phase.value)
        self.result = PickPlaceResult(
            success=success,
            failure_reason=reason,
            phase_history=tuple(self.phase_history),
            episode_length=self.episode_length,
            initial_cube_position_m=tuple(float(x) for x in self._cube_initial),
            final_cube_position_m=tuple(float(x) for x in final),
            final_placement_error_m=placement_error,
            grasp_verified=self._grasp_verified,
        )

    def _advance(self) -> None:
        transitions = {
            TaskPhase.APPROACH: TaskPhase.ALIGN,
            TaskPhase.ALIGN: TaskPhase.DESCEND,
            TaskPhase.DESCEND: TaskPhase.CLOSE_GRIPPER,
            TaskPhase.CLOSE_GRIPPER: TaskPhase.VERIFY_GRASP,
            TaskPhase.VERIFY_GRASP: TaskPhase.LIFT,
            TaskPhase.LIFT: TaskPhase.TRANSPORT,
            TaskPhase.TRANSPORT: TaskPhase.PLACE,
            TaskPhase.PLACE: TaskPhase.RELEASE,
            TaskPhase.RELEASE: TaskPhase.VERIFY_PLACEMENT,
            TaskPhase.VERIFY_PLACEMENT: TaskPhase.SUCCESS,
        }
        next_phase = transitions.get(self.phase)
        if next_phase is None:
            return
        if self.phase == TaskPhase.CLOSE_GRIPPER:
            if not self._bilateral_grasp_contacts():
                self._finish(False, "gripper closure did not produce contact on both fingers")
                return
            self._grasp_confirmed_at = self.phase_elapsed
        elif self.phase == TaskPhase.VERIFY_GRASP:
            if not self._bilateral_grasp_contacts():
                self._finish(False, "bilateral cube contact was not maintained during grasp verification")
                return
            self._grasp_verified = True
            self._cube_position_before_lift = self._cube_position()
        elif self.phase == TaskPhase.LIFT:
            lifted = float(self._cube_position()[2] - self._cube_position_before_lift[2])
            if not self._bilateral_grasp_contacts() or lifted < self.config.minimum_lift_m:
                self._grasp_verified = False
                self._finish(False, f"cube did not follow the gripper during lift (rise {lifted:.3f} m; bilateral contact={self._bilateral_grasp_contacts()})")
                return
        elif self.phase == TaskPhase.RELEASE:
            if not self._table_supports_cube():
                self._finish(False, "cube was not supported by the table after release")
                return
        elif self.phase == TaskPhase.VERIFY_PLACEMENT:
            cube = self._cube_position()
            error = float(np.linalg.norm(cube[:2] - self._destination[:2]))
            if error > self.config.placement_tolerance_m:
                self._finish(False, f"released cube missed destination by {error:.3f} m")
                return
            if cube[2] < self._table_top + self._cube_half_height - 0.015 or not self._table_supports_cube():
                self._finish(False, "cube is not stably resting on the table at the destination")
                return
            if any(force >= self.config.minimum_contact_force_n for force in self._contact_forces().values()):
                self._finish(False, "cube remains in contact with a gripper finger after release")
                return
            self._finish(True)
            return
        self._enter_phase(next_phase)

    def step(self) -> TaskStep:
        if self._obs is None:
            raise RuntimeError("Call reset() before stepping the task")
        if self.result is not None:
            return TaskStep(self._obs.copy(), 0.0, False, False, {}, self.phase, self.result)
        dt = float(self.env.model.opt.timestep * self.env.n_substeps)
        self.phase_elapsed += dt
        action = self.controller.compute_action(self._obs)
        if self._target_q is not None:
            fraction = min(self.phase_elapsed / self.config.motion_duration_s, 1.0)
            delta = self._target_q - self._phase_start_q
            max_delta = float(np.max(np.abs(delta)))
            if max_delta > 0:
                bounded_fraction = (self.phase_elapsed / dt) * self.config.max_joint_step_rad / max_delta
                fraction = min(fraction, bounded_fraction)
            self.controller.set_joint_target(self._phase_start_q + fraction * delta)
            action = self.controller.compute_action(self._obs)
        action[self._gripper_action_index] = 1.0 if self.phase in (
            TaskPhase.APPROACH, TaskPhase.ALIGN, TaskPhase.DESCEND,
            TaskPhase.RELEASE, TaskPhase.VERIFY_PLACEMENT,
        ) else -1.0
        self._obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_length += 1
        self.phase_elapsed += 0.0
        self._check_phase_progress(dt)
        return TaskStep(self._obs.copy(), float(reward), bool(terminated), bool(truncated), info, self.phase, self.result)

    def _check_phase_progress(self, dt: float) -> None:
        if self.result is not None:
            return
        timeout = float(self.config.phase_timeouts_s[self.phase.value])
        if self.phase_elapsed >= timeout:
            self._finish(False, f"{self.phase.value} timed out after {timeout:.1f} s")
            return
        if self.phase in (TaskPhase.CLOSE_GRIPPER, TaskPhase.VERIFY_GRASP):
            if not self._bilateral_grasp_contacts():
                return
            required = self.config.close_settle_s if self.phase == TaskPhase.CLOSE_GRIPPER else self.config.grasp_verify_s
            if self.phase_elapsed >= required:
                self._advance()
            return
        if self.phase == TaskPhase.RELEASE:
            finger_positions = self.env.data.qpos[[
                self.env.model.jnt_qposadr[self.env.joint_index("finger_joint1")],
                self.env.model.jnt_qposadr[self.env.joint_index("finger_joint2")],
            ]]
            if self.phase_elapsed >= self.config.release_duration_s and float(np.sum(finger_positions)) >= self.config.minimum_opening_m:
                self._advance()
            return
        if self.phase == TaskPhase.VERIFY_PLACEMENT:
            if self.phase_elapsed >= self.config.placement_settle_s:
                self._advance()
            return
        if self._phase_target is not None and self._target_q is not None:
            error = float(np.linalg.norm(self.env.data.xpos[self._hand_body] - self._phase_target))
            qvel = self.env.data.qvel[self._arm_dof_indices]
            stable = float(np.max(np.abs(qvel))) < 0.25
            motion_done = self.phase_elapsed >= self.config.motion_duration_s
            if motion_done and error <= self.config.position_tolerance_m and stable:
                if self.phase == TaskPhase.ALIGN and not np.allclose(self.env.data.xmat[self._hand_body].reshape(3, 3), self._hand_rotation, atol=0.03):
                    self._finish(False, "gripper failed to align with the configured grasp orientation")
                else:
                    if self.phase == TaskPhase.LIFT:
                        self._grasp_verified = (
                            self._bilateral_grasp_contacts()
                            and self._cube_position()[2] - self._cube_position_before_lift[2] >= self.config.minimum_lift_m
                        )
                        if not self._grasp_verified:
                            self._finish(False, "cube did not follow the gripper during lift")
                            return
                    self._advance()

    def run(self, *, seed: int | None = None, max_steps: int = 5000) -> PickPlaceResult:
        self.reset(seed=seed)
        for _ in range(max_steps):
            step = self.step()
            if step.result is not None:
                return step.result
            if step.terminated or step.truncated:
                self._finish(False, "environment terminated or truncated before task completion")
                return self.result
        self._finish(False, f"task exceeded max_steps={max_steps}")
        return self.result
