"""MuJoCo-backed robot environment adapter for Trajecta."""

from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Any

import gymnasium.spaces as spaces
import mujoco
import numpy as np

from roboforge.environments.contract import RobotEnvironment


def _enum_name(enum_type: Any, value: int) -> str:
    """Return a readable MuJoCo enum member name for model metadata."""
    for name in dir(enum_type):
        if name.startswith("mj"):
            try:
                if int(getattr(enum_type, name)) == int(value):
                    return name
            except (TypeError, ValueError):
                continue
    return f"unknown({value})"


def get_default_asset_path() -> Path:
    """Return the filesystem path to the packaged Panda scene."""
    traversable = importlib.resources.files("roboforge.environments.assets").joinpath("panda", "panda.xml")
    return Path(str(traversable))


class MujocoRobotEnvironment:
    """MuJoCo simulator environment conforming to the RobotEnvironment protocol.

    Action contract:
        Actions are normalized per MuJoCo actuator, in model actuator order.
        Each action maps linearly to that actuator's actual ``ctrlrange``; the
        environment does not assume that a control value is a joint position.
        Actuator transmission, gain, bias, and control range are available in
        ``actuator_specs`` and the physical command is returned in step info.

    Observation contract:
        A float32 concatenation of MuJoCo ``qpos`` followed by ``qvel``.
        ``observation_layout`` provides the named scalar-joint addresses. Free
        joints occupy their MuJoCo-sized qpos/qvel spans and are not flattened
        into scalar entries in ``joint_positions``/``joint_velocities``.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        xml_string: str | None = None,
        n_substeps: int = 10,
        max_episode_steps: int = 500,
    ) -> None:
        if xml_string is not None:
            self._model = mujoco.MjModel.from_xml_string(xml_string)
        else:
            resolved_path = Path(model_path) if model_path is not None else get_default_asset_path()
            if not resolved_path.exists():
                raise FileNotFoundError(f"MuJoCo model file not found at: {resolved_path}")
            self._model = mujoco.MjModel.from_xml_path(str(resolved_path))

        self._data = mujoco.MjData(self._model)
        self.n_substeps = n_substeps
        self.max_episode_steps = max_episode_steps
        self._step_count = 0
        self._is_closed = False
        self._rng = np.random.default_rng()

        self._ctrl_ranges = self._model.actuator_ctrlrange.copy()
        nu = self._model.nu
        if nu and not bool(np.all(self._model.actuator_ctrllimited)):
            raise ValueError("Every actuator must define a finite ctrlrange for normalized actions.")
        if not np.isfinite(self._ctrl_ranges).all():
            raise ValueError("Actuator control ranges must be finite.")
        self._action_space = spaces.Box(low=-1.0, high=1.0, shape=(nu,), dtype=np.float32)
        obs_dim = int(self._model.nq + self._model.nv)
        self._observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data

    @property
    def is_closed(self) -> bool:
        return self._is_closed

    @property
    def observation_space(self) -> spaces.Box:
        return self._observation_space

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    @property
    def actuator_names(self) -> tuple[str | None, ...]:
        """Names in the exact action-vector / ``data.ctrl`` ordering."""
        return tuple(
            mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self._model.nu)
        )

    @property
    def joint_names(self) -> tuple[str | None, ...]:
        """MuJoCo joint names in model order (free joints may be unnamed)."""
        return tuple(
            mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self._model.njnt)
        )

    @property
    def actuator_specs(self) -> tuple[dict[str, Any], ...]:
        """Actuator transmission and control metadata in action-vector order."""
        specs: list[dict[str, Any]] = []
        for actuator_id, name in enumerate(self.actuator_names):
            transmission_id = int(self._model.actuator_trntype[actuator_id])
            if transmission_id == int(mujoco.mjtTrn.mjTRN_JOINT):
                transmission = "joint"
                target = mujoco.mj_id2name(
                    self._model, mujoco.mjtObj.mjOBJ_JOINT,
                    int(self._model.actuator_trnid[actuator_id, 0]),
                )
            elif transmission_id == int(mujoco.mjtTrn.mjTRN_TENDON):
                transmission = "tendon"
                target = mujoco.mj_id2name(
                    self._model, mujoco.mjtObj.mjOBJ_TENDON,
                    int(self._model.actuator_trnid[actuator_id, 0]),
                )
            else:
                transmission = f"MuJoCo transmission type {transmission_id}"
                target = None
            specs.append({
                "index": actuator_id,
                "name": name,
                "transmission": transmission,
                "target": target,
                "control_range": tuple(float(v) for v in self._ctrl_ranges[actuator_id]),
                "dynamics_type": _enum_name(mujoco.mjtDyn, self._model.actuator_dyntype[actuator_id]),
                "gain_type": _enum_name(mujoco.mjtGain, self._model.actuator_gaintype[actuator_id]),
                "bias_type": _enum_name(mujoco.mjtBias, self._model.actuator_biastype[actuator_id]),
                "gain_parameters": tuple(float(v) for v in self._model.actuator_gainprm[actuator_id, :3]),
                "bias_parameters": tuple(float(v) for v in self._model.actuator_biasprm[actuator_id, :3]),
            })
        return tuple(specs)

    @property
    def observation_layout(self) -> dict[str, Any]:
        """Slices and named scalar-joint indices in the flat observation."""
        positions: dict[str, int] = {}
        velocities: dict[str, int] = {}
        for joint_id, name in enumerate(self.joint_names):
            if name is None:
                continue
            joint_type = int(self._model.jnt_type[joint_id])
            if joint_type in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                positions[name] = int(self._model.jnt_qposadr[joint_id])
                velocities[name] = int(self._model.nq + self._model.jnt_dofadr[joint_id])
        return {
            "qpos": slice(0, self._model.nq),
            "qvel": slice(self._model.nq, self._model.nq + self._model.nv),
            "joint_position_indices": positions,
            "joint_velocity_observation_indices": velocities,
        }

    def actuator_index(self, name: str) -> int:
        """Resolve an actuator by model name; never rely on a guessed index."""
        try:
            return self.actuator_names.index(name)
        except ValueError as exc:
            raise KeyError(f"No actuator named {name!r}; available: {self.actuator_names}") from exc

    def joint_index(self, name: str) -> int:
        """Resolve a joint by model name."""
        try:
            return self.joint_names.index(name)
        except ValueError as exc:
            raise KeyError(f"No joint named {name!r}; available: {self.joint_names}") from exc

    def _get_obs(self) -> np.ndarray:
        observation = np.concatenate([
            self._data.qpos.ravel().astype(np.float32),
            self._data.qvel.ravel().astype(np.float32),
        ])
        if not np.isfinite(observation).all():
            raise FloatingPointError("Non-finite state in MuJoCo observation.")
        return observation

    def _state_info(self) -> dict[str, Any]:
        layout = self.observation_layout
        qpos = self._data.qpos
        qvel = self._data.qvel
        joint_positions = {name: float(qpos[index]) for name, index in layout["joint_position_indices"].items()}
        joint_velocities = {
            name: float(qvel[index - self._model.nq])
            for name, index in layout["joint_velocity_observation_indices"].items()
        }
        finger_names = ("finger_joint1", "finger_joint2")
        gripper_positions = {
            name: joint_positions[name]
            for name in finger_names
            if name in joint_positions
        }
        gripper_state: dict[str, Any] = {"finger_positions": gripper_positions}
        if len(gripper_positions) == 2:
            gripper_state["opening_width"] = float(sum(gripper_positions.values()))
        return {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "gripper_state": gripper_state,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self._is_closed:
            raise RuntimeError("Cannot reset a closed environment.")
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        # Menagerie Panda supplies a named home keyframe; use it so restricted
        # joints (notably Panda joint4) start at valid positions. Other models
        # retain MuJoCo's ordinary zero-state reset.
        home_key = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_key >= 0:
            mujoco.mj_resetDataKeyframe(self._model, self._data, home_key)
        else:
            mujoco.mj_resetData(self._model, self._data)
        if options and "qpos" in options:
            qpos = np.asarray(options["qpos"], dtype=np.float64)
            if qpos.shape == self._data.qpos.shape:
                self._data.qpos[:] = qpos
        if options and "qvel" in options:
            qvel = np.asarray(options["qvel"], dtype=np.float64)
            if qvel.shape == self._data.qvel.shape:
                self._data.qvel[:] = qvel
        mujoco.mj_forward(self._model, self._data)
        info: dict[str, Any] = {
            "simulation_time": float(self._data.time),
            "step_count": self._step_count,
            "seed": seed,
            "is_success": False,
            **self._state_info(),
        }
        return self._get_obs(), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._is_closed:
            raise RuntimeError("Cannot step in a closed environment.")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self._action_space.shape:
            raise ValueError(f"action must have shape {self._action_space.shape}; got {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError("action must contain only finite values")
        clipped = np.clip(action, self._action_space.low, self._action_space.high)
        ctrl_min = self._ctrl_ranges[:, 0]
        ctrl_max = self._ctrl_ranges[:, 1]
        ctrl_values = ctrl_min + (clipped + 1.0) * 0.5 * (ctrl_max - ctrl_min)
        self._data.ctrl[:] = ctrl_values
        for _ in range(self.n_substeps):
            mujoco.mj_step(self._model, self._data)
        self._step_count += 1
        obs = self._get_obs()
        is_finite = bool(np.isfinite(obs).all())
        if not is_finite:
            raise FloatingPointError("Non-finite state in MuJoCo step.")
        terminated = False
        truncated = bool(self._step_count >= self.max_episode_steps)
        info: dict[str, Any] = {
            "simulation_time": float(self._data.time),
            "step_count": self._step_count,
            "applied_ctrl": ctrl_values.copy(),
            "is_finite": is_finite,
            "actuator_commands": {
                name if name is not None else f"actuator_{i}": float(ctrl_values[i])
                for i, name in enumerate(self.actuator_names)
            },
            **self._state_info(),
        }
        return obs, 0.0, terminated, truncated, info

    def close(self) -> None:
        self._is_closed = True
