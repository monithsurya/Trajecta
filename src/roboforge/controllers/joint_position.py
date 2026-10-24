"""Deterministic joint-space position controller for articulated robot arms."""

from __future__ import annotations

from typing import Any

import numpy as np

from roboforge.controllers.contract import Controller
from roboforge.experiments.spec import ControllerConfig, ControllerType

# Default physical joint limits (radians) matching the Trajecta 2-DOF simple arm:
# shoulder_pitch in [-pi/2, pi/2], elbow_pitch in [-2pi/3, 2pi/3]
DEFAULT_JOINT_LIMITS: tuple[tuple[float, float], ...] = (
    (-1.5708, 1.5708),
    (-2.0944, 2.0944),
)


class JointPositionController:
    """A deterministic joint-space position controller.

    This controller maps selected joint positions from an observation into bounded
    normalized action slots. By default it preserves the original first-N input and
    output layout. Optional observation/action indices allow it to control a named
    subset of a larger environment action (such as the Panda arm while retaining a
    separate gripper command) without importing a physics engine.

    Target Mapping:
        Given target joint positions q_target in [q_min, q_max] and optional proportional
        gain kp, the controller computes the desired setpoint:
            q_cmd = q_current + kp * (q_target - q_current)  [if kp is specified]
            q_cmd = q_target                                [if kp is None]

        The commanded joint position is then normalized to the [-1.0, 1.0] action range:
            action_i = 2.0 * (q_cmd_i - q_min_i) / (q_max_i - q_min_i) - 1.0
            action = clip(action, -1.0, 1.0)
    """

    def __init__(
        self,
        target_positions: list[float] | np.ndarray,
        kp: float | list[float] | np.ndarray | None = None,
        joint_limits: tuple[tuple[float, float], ...] | list[tuple[float, float]] | np.ndarray = DEFAULT_JOINT_LIMITS,
        *,
        observation_indices: list[int] | tuple[int, ...] | None = None,
        action_size: int | None = None,
        action_indices: list[int] | tuple[int, ...] | None = None,
        default_action: list[float] | np.ndarray | None = None,
    ) -> None:
        """Initialize the joint position controller.

        Args:
            target_positions: Target joint angles in radians. Length determines num_joints.
            kp: Optional proportional gain (scalar or per-joint array, must be >= 0).
                Controls the rate of convergence towards the target positions.
            joint_limits: (min, max) limits for each joint in radians.
            observation_indices: Indices of each controlled joint position in the
                observation. Defaults to the original first-N scalar layout.
            action_size: Full action vector length when this controller controls
                only selected action slots (for example Panda arm actuators plus
                a separate gripper command).
            action_indices: Action slots for the controlled joints. Defaults to
                the first N slots.
            default_action: Values for non-controlled action slots. Defaults to
                zeros; values must be finite and in [-1, 1].
        """
        targets = np.asarray(target_positions, dtype=np.float32)
        if targets.ndim != 1:
            raise ValueError(f"target_positions must be 1D; got shape {targets.shape}")
        if len(targets) == 0:
            raise ValueError("target_positions cannot be empty")

        num_joints = len(targets)
        limits_arr = np.asarray(joint_limits, dtype=np.float32)
        if limits_arr.shape != (num_joints, 2):
            raise ValueError(
                f"joint_limits must have shape ({num_joints}, 2); got {limits_arr.shape}"
            )
        if np.any(limits_arr[:, 0] >= limits_arr[:, 1]):
            raise ValueError("joint_limits minimums must be strictly less than maximums")

        self._num_joints = num_joints
        self._q_min = limits_arr[:, 0]
        self._q_max = limits_arr[:, 1]
        self._target_positions = np.clip(targets, self._q_min, self._q_max)

        self._observation_indices = np.asarray(
            observation_indices if observation_indices is not None else range(num_joints),
            dtype=np.int64,
        )
        self._action_indices = np.asarray(
            action_indices if action_indices is not None else range(num_joints),
            dtype=np.int64,
        )
        if self._observation_indices.shape != (num_joints,) or np.any(self._observation_indices < 0):
            raise ValueError("observation_indices must provide one non-negative index per target joint")
        if self._action_indices.shape != (num_joints,) or np.any(self._action_indices < 0):
            raise ValueError("action_indices must provide one non-negative index per target joint")
        if len(np.unique(self._action_indices)) != num_joints:
            raise ValueError("action_indices must be unique")
        inferred_action_size = int(self._action_indices.max()) + 1
        self._action_size = inferred_action_size if action_size is None else int(action_size)
        if self._action_size < inferred_action_size:
            raise ValueError("action_size must include all action_indices")
        if np.any(self._action_indices >= self._action_size):
            raise ValueError("action_indices must be smaller than action_size")
        if default_action is None:
            self._default_action = np.zeros(self._action_size, dtype=np.float32)
        else:
            self._default_action = np.asarray(default_action, dtype=np.float32)
            if self._default_action.shape != (self._action_size,):
                raise ValueError(f"default_action must have shape ({self._action_size},)")
            if not np.isfinite(self._default_action).all() or np.any(np.abs(self._default_action) > 1.0):
                raise ValueError("default_action values must be finite and in [-1, 1]")

        if kp is not None:
            kp_arr = np.asarray(kp, dtype=np.float32)
            if kp_arr.ndim == 0:
                kp_arr = np.full(num_joints, fill_value=float(kp_arr), dtype=np.float32)
            elif kp_arr.shape != (num_joints,):
                raise ValueError(
                    f"kp array length ({len(kp_arr)}) must match num_joints ({num_joints})"
                )
            if np.any(kp_arr < 0.0):
                raise ValueError(f"kp values must be non-negative; got {kp}")
            self._kp: np.ndarray | None = kp_arr
        else:
            self._kp = None

    @property
    def num_joints(self) -> int:
        """Return the number of controlled joints."""
        return self._num_joints

    @property
    def target_positions(self) -> np.ndarray:
        """Return the target joint angles in radians."""
        return self._target_positions.copy()

    @property
    def kp(self) -> np.ndarray | None:
        """Return proportional gains, or None if direct setpoint mapping is active."""
        return self._kp.copy() if self._kp is not None else None

    @property
    def joint_limits(self) -> np.ndarray:
        """Return joint limits array of shape (num_joints, 2)."""
        return np.column_stack([self._q_min, self._q_max])

    def set_target_positions(self, target_positions: list[float] | np.ndarray) -> None:
        """Update target joint positions.

        Args:
            target_positions: Target joint angles in radians, length must match num_joints.
        """
        targets = np.asarray(target_positions, dtype=np.float32)
        if targets.shape != (self._num_joints,):
            raise ValueError(
                f"target_positions shape must be ({self._num_joints},); got {targets.shape}"
            )
        self._target_positions = np.clip(targets, self._q_min, self._q_max)

    @classmethod
    def from_config(
        cls,
        config: ControllerConfig,
        num_joints: int = 2,
    ) -> JointPositionController:
        """Construct a JointPositionController from a validated ControllerConfig.

        Args:
            config: Validated ControllerConfig from an ExperimentSpec.
            num_joints: Expected number of joints (defaults to 2 for Trajecta simple arm).

        Returns:
            A configured JointPositionController instance.
        """
        if config.controller_type != ControllerType.JOINT_POSITION:
            raise ValueError(
                f"Expected ControllerType.JOINT_POSITION; got {config.controller_type}"
            )

        target_positions = config.extra_params.get(
            "target_positions", [0.0] * num_joints
        )
        joint_limits = config.extra_params.get(
            "joint_limits", DEFAULT_JOINT_LIMITS[:num_joints]
        )

        return cls(
            target_positions=target_positions,
            kp=config.kp,
            joint_limits=joint_limits,
        )

    def compute_action(self, observation: np.ndarray) -> np.ndarray:
        """Compute an action conforming to [-1.0, 1.0]^num_joints from observation.

        Args:
            observation: 1D array of environment observations. The first `num_joints`
                elements must represent current joint positions.

        Returns:
            1D float32 NumPy array of shape (action_size,), bounded by [-1.0, 1.0].
        """
        if not isinstance(observation, np.ndarray):
            raise TypeError(
                f"observation must be a numpy.ndarray; got {type(observation).__name__}"
            )
        if observation.ndim != 1:
            raise ValueError(
                f"observation must be 1D; got ndim={observation.ndim} with shape {observation.shape}"
            )
        if len(observation) <= int(self._observation_indices.max()):
            raise ValueError(
                f"observation length ({len(observation)}) is smaller than num_joints "
                f"or does not contain required index {int(self._observation_indices.max())}"
            )

        q_current = observation[self._observation_indices].astype(np.float32)
        if not np.all(np.isfinite(q_current)):
            raise ValueError(f"observation contains non-finite values: {q_current}")

        if self._kp is not None:
            # Proportional tracking step: q_cmd = q_current + kp * (q_target - q_current)
            error = self._target_positions - q_current
            q_cmd = q_current + self._kp * error
        else:
            # Direct setpoint normalization
            q_cmd = self._target_positions

        # Normalize q_cmd from [q_min, q_max] to [-1.0, 1.0]
        q_span = self._q_max - self._q_min
        action = 2.0 * (q_cmd - self._q_min) / q_span - 1.0

        normalized = np.clip(action, -1.0, 1.0).astype(np.float32)
        full_action = self._default_action.copy()
        full_action[self._action_indices] = normalized
        return full_action

    def reset(self) -> None:
        """Reset internal controller state (stateless, provided for protocol compliance)."""
        pass

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        """Enable functional-style callable syntax `action = controller(obs)`."""
        return self.compute_action(observation)
