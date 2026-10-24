"""Integration checks for the packaged MuJoCo Menagerie Panda scene."""

from __future__ import annotations

import numpy as np
import mujoco
import pytest

from roboforge.controllers.joint_position import JointPositionController
from roboforge.environments.mujoco_env import MujocoRobotEnvironment, get_default_asset_path


def test_default_scene_is_panda_with_table_and_target_cube() -> None:
    assert get_default_asset_path().name == "panda.xml"
    env = MujocoRobotEnvironment()
    try:
        assert env.model.nu == 8  # seven arm position actuators plus gripper tendon actuator
        assert env.model.nq == 16  # 7 arm, 2 finger, and 7 free cube coordinates
        actuator_names = [mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(env.model.nu)]
        assert actuator_names == [f"actuator{i}" for i in range(1, 9)]
        assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "table") >= 0
        assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "target_cube") >= 0
    finally:
        env.close()


def test_panda_reset_step_spaces_and_gripper_mapping() -> None:
    env = MujocoRobotEnvironment(n_substeps=2, max_episode_steps=2)
    try:
        obs0, info = env.reset(seed=17)
        assert obs0.shape == (env.model.nq + env.model.nv,)
        assert env.observation_space.contains(obs0)
        assert info["step_count"] == 0

        # Normalized arm actions map per actuator; gripper -1 closes (ctrl 0),
        # while +1 opens (ctrl 255) for the Menagerie tendon actuator.
        action = np.zeros(env.model.nu, dtype=np.float32)
        action[-1] = 1.0
        obs1, _, terminated, truncated, step_info = env.step(action)
        assert not terminated and not truncated
        assert env.data.time > 0
        assert obs1.shape == env.observation_space.shape
        assert np.all(np.isfinite(obs1))
        assert np.isclose(step_info["applied_ctrl"][-1], 255.0)

        action[-1] = -1.0
        _, _, _, truncated, step_info = env.step(action)
        assert truncated
        assert np.isclose(step_info["applied_ctrl"][-1], 0.0)
        assert step_info["actuator_commands"]["actuator8"] == 0.0

        obs_reset, reset_info = env.reset(seed=17)
        assert reset_info["step_count"] == 0
        assert np.allclose(obs_reset, obs0)
    finally:
        env.close()


def test_actuator_and_observation_metadata_matches_loaded_model() -> None:
    env = MujocoRobotEnvironment()
    try:
        specs = env.actuator_specs
        assert [spec["target"] for spec in specs[:7]] == [f"joint{i}" for i in range(1, 8)]
        assert specs[7]["transmission"] == "tendon"
        assert specs[7]["target"] == "split"
        assert all(spec["dynamics_type"] == "mjDYN_NONE" for spec in specs)
        assert all(spec["bias_type"] == "mjBIAS_AFFINE" for spec in specs)
        assert [spec["control_range"] for spec in specs] == [
            tuple(float(v) for v in row) for row in env.model.actuator_ctrlrange
        ]
        layout = env.observation_layout
        # Cube free joint precedes Panda in model qpos; use named indices so its
        # translation/quaternion are never mistaken for arm joints.
        assert layout["joint_position_indices"]["joint1"] == int(env.model.jnt_qposadr[env.joint_index("joint1")])
        assert layout["joint_position_indices"]["joint1"] == 7
        assert layout["joint_velocity_observation_indices"]["joint1"] == env.model.nq + 6
        obs, info = env.reset(seed=4)
        assert env.observation_space.contains(obs)
        assert np.isfinite(obs).all()
        assert set(info["joint_positions"]) >= {f"joint{i}" for i in range(1, 8)}
        for name in (f"joint{i}" for i in range(1, 8)):
            joint_id = env.joint_index(name)
            assert env.model.jnt_range[joint_id, 0] <= info["joint_positions"][name] <= env.model.jnt_range[joint_id, 1]
        assert set(info["gripper_state"]["finger_positions"]) == {"finger_joint1", "finger_joint2"}
    finally:
        env.close()


def test_action_shape_finiteness_and_actuator_range_clipping() -> None:
    env = MujocoRobotEnvironment(n_substeps=1)
    try:
        env.reset()
        with pytest.raises(ValueError, match="shape"):
            env.step(np.zeros(env.model.nu - 1, dtype=np.float32))
        bad = np.zeros(env.model.nu, dtype=np.float32)
        bad[0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            env.step(bad)

        action = np.full(env.model.nu, 4.0, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert np.allclose(info["applied_ctrl"], env.model.actuator_ctrlrange[:, 1])
    finally:
        env.close()


def test_joint_position_controller_maps_named_panda_joints_to_matching_actuators() -> None:
    env = MujocoRobotEnvironment()
    try:
        obs, _ = env.reset()
        layout = env.observation_layout
        joint_names = [f"joint{i}" for i in range(1, 8)]
        actuator_by_joint = {
            spec["target"]: spec
            for spec in env.actuator_specs
            if spec["transmission"] == "joint"
        }
        limits = [
            tuple(float(v) for v in env.model.jnt_range[env.joint_index(name)])
            for name in joint_names
        ]
        controller = JointPositionController(
            target_positions=[obs[layout["joint_position_indices"][name]] for name in joint_names],
            joint_limits=limits,
            observation_indices=[layout["joint_position_indices"][name] for name in joint_names],
            action_size=env.action_space.shape[0],
            action_indices=[actuator_by_joint[name]["index"] for name in joint_names],
            default_action=[0, 0, 0, 0, 0, 0, 0, 1],
        )
        action = controller.compute_action(obs)
        assert action.shape == env.action_space.shape
        assert action.dtype == np.float32
        assert env.action_space.contains(action)
        assert action[-1] == 1.0  # Uncontrolled tendon actuator stays explicitly open.
        _, _, _, _, info = env.step(action)
        assert np.isfinite(info["applied_ctrl"]).all()
        assert info["actuator_commands"]["actuator8"] == env.model.actuator_ctrlrange[7, 1]
    finally:
        env.close()
