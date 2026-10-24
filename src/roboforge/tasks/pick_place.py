"""Contact-verified Panda pick-and-place task state machine.

Task coordinates are world-frame meters. Arm targets are solved by the
injected position controller and executed through the environment action API.
The cube is never repositioned by task code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping
from uuid import UUID

import mujoco
import numpy as np

from roboforge.experiments.transitions import TransitionRecord, TransitionRecorder


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
    cube_start_position_m: tuple[float, float, float] | None = None
    cube_target_position_m: tuple[float, float, float] | None = None
    cube_start_bounds_m: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    destination_xy_m: tuple[float, float] = (0.65, -0.15)
    destination_bounds_xy_m: tuple[tuple[float, float], tuple[float, float]] | None = None
    robot_initial_joint_positions_rad: tuple[float, ...] | None = None
    robot_joint_perturbation_rad: float = 0.0
    cube_mass_scale_range: tuple[float, float] = (1.0, 1.0)
    cube_friction_scale_range: tuple[float, float] = (1.0, 1.0)
    max_recovery_attempts: int = 0
    max_replans_per_phase: int = 2
    object_replan_threshold_m: float = 0.01
    recovery_grasp_offsets_m: tuple[float, ...] = (0.0, -0.005, 0.005)
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
        if self.destination_bounds_xy_m is not None:
            if len(self.destination_bounds_xy_m) != 2 or any(len(vector) != 2 for vector in self.destination_bounds_xy_m):
                raise ValueError("destination_bounds_xy_m must contain finite XY low/high vectors")
            low, high = (np.asarray(vector, dtype=np.float64) for vector in self.destination_bounds_xy_m)
            if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low >= high):
                raise ValueError("destination_bounds_xy_m low values must be below high values")
        for name in ("cube_start_position_m", "cube_target_position_m"):
            position = getattr(self, name)
            if position is not None and (len(position) != 3 or not np.isfinite(position).all()):
                raise ValueError(f"{name} must be three finite world coordinates")
        if self.cube_start_position_m is not None and self.cube_target_position_m is not None and not np.allclose(
            self.cube_start_position_m, self.cube_target_position_m,
        ):
            raise ValueError("cube_start_position_m and legacy cube_target_position_m cannot disagree")
        if self.cube_start_bounds_m is not None:
            if len(self.cube_start_bounds_m) != 2 or any(len(vector) != 3 for vector in self.cube_start_bounds_m):
                raise ValueError("cube_start_bounds_m must contain finite XYZ low/high vectors")
            low, high = (np.asarray(vector, dtype=np.float64) for vector in self.cube_start_bounds_m)
            if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low > high) or np.all(low == high):
                raise ValueError("cube_start_bounds_m low values must be finite and no greater than high values")
        if self.robot_initial_joint_positions_rad is not None and (
            len(self.robot_initial_joint_positions_rad) != 7
            or not np.isfinite(self.robot_initial_joint_positions_rad).all()
        ):
            raise ValueError("robot_initial_joint_positions_rad must contain seven finite values")
        if not np.isfinite(self.robot_joint_perturbation_rad) or not 0 <= self.robot_joint_perturbation_rad <= 0.15:
            raise ValueError("robot_joint_perturbation_rad must be in [0, 0.15]")
        for name in ("cube_mass_scale_range", "cube_friction_scale_range"):
            value = getattr(self, name)
            if len(value) != 2 or not np.isfinite(value).all() or not 0.5 <= value[0] <= value[1] <= 1.5:
                raise ValueError(f"{name} must be ordered finite bounds within [0.5, 1.5]")
        if isinstance(self.max_recovery_attempts, bool) or not 0 <= self.max_recovery_attempts <= 3:
            raise ValueError("max_recovery_attempts must be between 0 and 3")
        if isinstance(self.max_replans_per_phase, bool) or not 0 <= self.max_replans_per_phase <= 5:
            raise ValueError("max_replans_per_phase must be between 0 and 5")
        if not np.isfinite(self.object_replan_threshold_m) or not 0.002 <= self.object_replan_threshold_m <= 0.05:
            raise ValueError("object_replan_threshold_m must be in [0.002, 0.05]")
        if not self.recovery_grasp_offsets_m or any(
            not np.isfinite(value) or abs(value) > 0.01 for value in self.recovery_grasp_offsets_m
        ):
            raise ValueError("recovery_grasp_offsets_m must contain bounded offsets within +/-0.01 m")
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
    recovery_attempts: int = 0
    replanning_count: int = 0
    failure_category: str | None = None
    condition_parameters: dict[str, Any] = field(default_factory=dict)


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
        self._transition_recorder: TransitionRecorder | None = None
        self._transition_experiment_id: UUID | None = None
        self._transition_episode_id: UUID | None = None
        self._transition_seed: int | None = None
        self._transition_index = 0
        self.condition_parameters: dict[str, Any] = {}
        self.replanning_count = 0
        self._phase_replans = 0
        self._phase_cube_reference: np.ndarray | None = None

    def configure_transition_recording(
        self, *, recorder: TransitionRecorder | None, experiment_id: UUID, episode_id: UUID,
        seed: int | None,
    ) -> None:
        """Set optional recording for the next run; IDs match its episode summary."""
        self._transition_recorder = recorder
        self._transition_experiment_id = experiment_id
        self._transition_episode_id = episode_id
        self._transition_seed = seed
        self._transition_index = 0

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        """Reset to the model home key and initialize targets from live geometry."""
        rng = np.random.default_rng(seed)
        destination_xy = self.config.destination_xy_m
        if self.config.destination_bounds_xy_m is not None:
            low, high = self.config.destination_bounds_xy_m
            destination_xy = tuple(float(value) for value in rng.uniform(low, high))
        cube_position = self.config.cube_start_position_m or self.config.cube_target_position_m
        if cube_position is None and self.config.cube_start_bounds_m is not None:
            low, high = self.config.cube_start_bounds_m
            cube_position = tuple(float(value) for value in rng.uniform(low, high))
        robot_positions = self.config.robot_initial_joint_positions_rad
        if robot_positions is None and self.config.robot_joint_perturbation_rad > 0:
            home_key = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_KEY, "home")
            home_qpos = self.env.model.key_qpos[home_key] if home_key >= 0 else self.env.model.qpos0
            joint_names = tuple(f"joint{i}" for i in range(1, 8))
            nominal = []
            for name in joint_names:
                joint_id = self.env.joint_index(name)
                nominal.append(home_qpos[self.env.model.jnt_qposadr[joint_id]])
            robot_positions = tuple(float(value) for value in (
                np.asarray(nominal) + rng.uniform(
                    -self.config.robot_joint_perturbation_rad,
                    self.config.robot_joint_perturbation_rad,
                    size=7,
                )
            ))
        mass_scale = float(rng.uniform(*self.config.cube_mass_scale_range))
        friction_scale = float(rng.uniform(*self.config.cube_friction_scale_range))
        reset_options = {"physics_variation": {
            "cube_mass_scale": mass_scale,
            "cube_friction_scale": friction_scale,
        }}
        if cube_position is not None:
            reset_options["cube_position_m"] = cube_position
        if robot_positions is not None:
            reset_options["joint_positions"] = {
                f"joint{index + 1}": value for index, value in enumerate(robot_positions)
            }
        if cube_position is not None or robot_positions is not None or mass_scale != 1.0 or friction_scale != 1.0:
            self._obs, _reset_info = self.env.reset(seed=seed, options=reset_options)
        else:
            self._obs, _reset_info = self.env.reset(seed=seed)
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
        self.condition_parameters = {
            "seed": seed,
            "cube_position_m": tuple(float(value) for value in self._cube_position()),
            "destination_xy_m": tuple(float(value) for value in destination_xy),
            "robot_initial_joint_positions_rad": {
                f"joint{index + 1}": float(self.env.data.qpos[self.env.model.jnt_qposadr[self.env.joint_index(f"joint{index + 1}")]])
                for index in range(7)
            },
            "cube_mass_scale": mass_scale,
            "cube_friction_scale": friction_scale,
            "replanning_count": 0,
        }
        self.controller.reset()
        model, data = self.env.model, self.env.data
        self._cube_initial = data.xpos[self._cube_body_id].copy()
        self._validate_cube_position(self._cube_initial)
        self._destination = np.array([
            destination_xy[0], destination_xy[1],
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
        self.replanning_count = 0
        self._phase_replans = 0
        self._grasp_confirmed_at = None
        self._cube_position_before_lift = None
        self.result = None
        self.phase_history = []
        self.episode_length = 0
        self._enter_phase(TaskPhase.APPROACH)
        return self._obs.copy()

    def _validate_cube_position(self, position: np.ndarray) -> None:
        table_center = self.env.model.geom_pos[self._table_geom_id]
        table_half_size = self.env.model.geom_size[self._table_geom_id]
        if any(abs(position[axis] - table_center[axis]) > table_half_size[axis] - self._cube_half_height for axis in (0, 1)):
            raise ValueError("cube start position must keep the block fully within the table footprint")
        if abs(position[2] - (self._table_top + self._cube_half_height)) > 0.02:
            raise ValueError("cube start position must place the block on the table surface")

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
        cube = self._cube_position()
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
        if phase != self.phase:
            self._phase_replans = 0
        self.phase = phase
        self.phase_elapsed = 0.0
        self.phase_history.append(phase.value)
        self._phase_target = self._phase_target_for(phase)
        self._phase_cube_reference = self._cube_position().copy()
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
        failure_category = None if success else self.classify_failure(reason or "", self.phase)
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
            failure_category=failure_category,
            condition_parameters={**self.condition_parameters, "replanning_count": self.replanning_count},
            replanning_count=self.replanning_count,
        )

    @staticmethod
    def classify_failure(reason: str, phase: TaskPhase) -> str:
        """Coarse outcome category for recovery/evaluation, not a causal diagnosis."""
        lowered = reason.lower()
        if any(token in lowered for token in ("grasp", "finger contact", "contact on both")):
            return "missed_grasp"
        if any(token in lowered for token in ("ik", "timed out", "timeout", "reach")):
            return "failed_reach"
        if any(token in lowered for token in ("placement", "destination", "table after release", "released cube")):
            return "placement_failure"
        if phase in (TaskPhase.CLOSE_GRIPPER, TaskPhase.VERIFY_GRASP, TaskPhase.LIFT):
            return "missed_grasp"
        if phase in (TaskPhase.PLACE, TaskPhase.RELEASE, TaskPhase.VERIFY_PLACEMENT, TaskPhase.TRANSPORT):
            return "placement_failure"
        return "task_failure"

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
        if self.phase in (TaskPhase.APPROACH, TaskPhase.ALIGN, TaskPhase.DESCEND) and self._phase_cube_reference is not None:
            current_cube = self._cube_position()
            if np.linalg.norm(current_cube - self._phase_cube_reference) > self.config.object_replan_threshold_m:
                if self._phase_replans >= self.config.max_replans_per_phase:
                    self._finish(False, f"object moved during {self.phase.value}; replan budget exhausted")
                    return TaskStep(
                        self._obs.copy(), 0.0, True, False,
                        {"failure_reason": self.result.failure_reason}, self.phase, self.result,
                    )
                self._phase_replans += 1
                self.replanning_count += 1
                self.condition_parameters["replanning_count"] = self.replanning_count
                self._enter_phase(self.phase)
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
        observation = self._obs.copy()
        applied_action = np.asarray(action).copy()
        action_phase = self.phase.value
        self._obs, reward, terminated, truncated, info = self.env.step(applied_action)
        self.episode_length += 1
        try:
            self.phase_elapsed += 0.0
            self._check_phase_progress(dt)
            if self.result is None and (terminated or truncated):
                self._finish(False, "environment terminated or truncated before task completion")
        finally:
            # Persist even if task-side post-step evaluation raises: env.step has
            # already completed, so omitting this transition would hide data.
            if self._transition_recorder is not None:
                assert self._transition_experiment_id is not None and self._transition_episode_id is not None
                record = TransitionRecord.from_transition(
                    experiment_id=self._transition_experiment_id,
                    episode_id=self._transition_episode_id,
                    transition_index=self._transition_index,
                    seed=self._transition_seed,
                    observation=observation,
                    action=applied_action,
                    next_observation=self._obs,
                    # Preserve the environment reward; task success is separate.
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    simulation_time_s=info.get("simulation_time"),
                    simulation_timestep_s=dt,
                    environment_step=info.get("step_count"),
                    task_phase=action_phase,
                    task_success=self.result.success if self.result is not None else None,
                    task_failure_reason=self.result.failure_reason if self.result is not None else None,
                )
                self._transition_recorder.write(record)
                self._transition_index += 1
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

    def _run_once(self, *, seed: int | None, max_steps: int) -> PickPlaceResult:
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

    def run(self, *, seed: int | None = None, max_steps: int = 5000) -> PickPlaceResult:
        """Run a bounded episode and retry only classified recoverable failures."""
        base_config = self.config
        recovery_outcomes: list[str] = []
        attempt_conditions: list[dict[str, Any]] = []
        total_steps = 0
        result: PickPlaceResult | None = None
        try:
            for attempt in range(base_config.max_recovery_attempts + 1):
                offset = base_config.recovery_grasp_offsets_m[
                    min(attempt, len(base_config.recovery_grasp_offsets_m) - 1)
                ]
                from dataclasses import replace
                self.config = replace(
                    base_config,
                    hand_to_fingertip_m=base_config.hand_to_fingertip_m + offset,
                    max_recovery_attempts=0,
                )
                retry_seed = None if seed is None else (seed + attempt) % (4_294_967_295 + 1)
                result = self._run_once(seed=retry_seed, max_steps=max_steps)
                total_steps += result.episode_length
                attempt_conditions.append(dict(result.condition_parameters))
                if result.success:
                    return replace(
                        result,
                        episode_length=total_steps,
                        recovery_attempts=attempt,
                        replanning_count=result.replanning_count,
                        condition_parameters={
                            **result.condition_parameters,
                            "recovery_outcomes": tuple(recovery_outcomes),
                            "recovery_attempts": attempt,
                            "attempt_conditions": tuple(attempt_conditions),
                        },
                    )
                category = result.failure_category or self.classify_failure(
                    result.failure_reason or "", self.phase,
                )
                recovery_outcomes.append(category)
                if category not in {"missed_grasp", "failed_reach", "placement_failure"}:
                    break
            assert result is not None
            return replace(
                result,
                episode_length=total_steps,
                recovery_attempts=max(0, len(recovery_outcomes) - 1),
                replanning_count=result.replanning_count,
                failure_category=result.failure_category or "task_failure",
                condition_parameters={
                    **result.condition_parameters,
                    "recovery_outcomes": tuple(recovery_outcomes),
                    "recovery_attempts": max(0, len(recovery_outcomes) - 1),
                    "attempt_conditions": tuple(attempt_conditions),
                },
            )
        finally:
            self.config = base_config
