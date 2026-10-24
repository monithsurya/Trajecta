"""Tests for the MuJoCo robot environment adapter."""

from __future__ import annotations

import gymnasium.spaces as spaces
import mujoco
import numpy as np
import pytest

from roboforge.environments.contract import RobotEnvironment
from roboforge.environments.mujoco_env import MujocoRobotEnvironment, get_default_asset_path


class TestMujocoRobotEnvironment:

    def test_default_asset_exists_and_loads(self) -> None:
        asset_path = get_default_asset_path()
        assert asset_path.exists(), f"Asset path {asset_path} does not exist"
        model = mujoco.MjModel.from_xml_path(str(asset_path))
        assert model.nu >= 2
        assert model.nq >= 2

    def test_satisfies_robot_environment_protocol(self) -> None:
        env = MujocoRobotEnvironment()
        try:
            assert isinstance(env, RobotEnvironment)
        finally:
            env.close()

    def test_spaces_presence_and_structure(self) -> None:
        env = MujocoRobotEnvironment()
        try:
            assert isinstance(env.action_space, spaces.Box)
            assert env.action_space.shape == (env.model.nu,)
            assert np.all(env.action_space.low == -1.0)
            assert np.all(env.action_space.high == 1.0)
            assert isinstance(env.observation_space, spaces.Box)
            assert env.observation_space.shape == (env.model.nq + env.model.nv,)
        finally:
            env.close()

    def test_reset_returns_valid_observation_and_info(self) -> None:
        env = MujocoRobotEnvironment()
        try:
            obs, info = env.reset(seed=123)
            assert isinstance(obs, np.ndarray)
            assert obs.dtype == np.float32
            assert obs.shape == env.observation_space.shape
            assert env.observation_space.contains(obs)
            assert np.all(np.isfinite(obs))
            assert info["step_count"] == 0
            assert info["seed"] == 123
        finally:
            env.close()

    def test_step_returns_five_part_contract(self) -> None:
        env = MujocoRobotEnvironment()
        try:
            env.reset()
            obs, reward, terminated, truncated, info = env.step(np.zeros(env.model.nu, dtype=np.float32))
            assert obs.shape == env.observation_space.shape
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert "applied_ctrl" in info
            assert "simulation_time" in info
        finally:
            env.close()

    def test_action_advances_simulation_with_finite_states(self) -> None:
        env = MujocoRobotEnvironment(n_substeps=10)
        try:
            env.reset(seed=42)
            t0 = env.data.time
            obs, _, _, _, info = env.step(np.ones(env.model.nu, dtype=np.float32))
            assert env.data.time > t0
            assert np.isclose(env.data.time - t0, 0.02, atol=1e-5)
            assert np.all(np.isfinite(obs))
            assert info["is_finite"] is True
            assert np.allclose(info["applied_ctrl"], env.model.actuator_ctrlrange[:, 1])
        finally:
            env.close()

    def test_reset_restores_initial_state(self) -> None:
        env = MujocoRobotEnvironment()
        try:
            obs_init, _ = env.reset(seed=42)
            for _ in range(5):
                env.step(np.ones(env.model.nu, dtype=np.float32))
            assert not np.allclose(obs_init, env._get_obs())
            obs_reset, info = env.reset(seed=42)
            assert np.allclose(obs_init, obs_reset)
            assert info["step_count"] == 0
            assert env.data.time == 0.0
        finally:
            env.close()

    def test_truncation_on_max_steps(self) -> None:
        env = MujocoRobotEnvironment(max_episode_steps=3)
        try:
            env.reset()
            action = np.zeros(env.model.nu, dtype=np.float32)
            for _ in range(2):
                _, _, t, tr, _ = env.step(action)
                assert not t and not tr
            _, _, t, tr, info = env.step(action)
            assert not t and tr and info["step_count"] == 3
        finally:
            env.close()

    def test_seeded_cube_and_physics_variations_are_recorded_and_reset_to_nominal(self) -> None:
        env = MujocoRobotEnvironment()
        try:
            body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "target_cube")
            geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
            nominal_mass = float(env.model.body_mass[body_id])
            nominal_friction = env.model.geom_friction[geom_id].copy()
            obs_a, info_a = env.reset(
                seed=5,
                options={
                    "cube_position_m": (0.48, 0.05, 0.725),
                    "joint_positions": {"joint1": 0.1},
                    "physics_variation": {"cube_mass_scale": 1.2, "cube_friction_scale": 0.8},
                },
            )
            assert info_a["physics_variation"] == {"cube_mass_scale": 1.2, "cube_friction_scale": 0.8}
            assert env.model.body_mass[body_id] == pytest.approx(nominal_mass * 1.2)
            assert env.model.geom_friction[geom_id] == pytest.approx(nominal_friction * 0.8)
            assert env.data.qpos[env.model.jnt_qposadr[env.joint_index("joint1")]] == pytest.approx(0.1)
            obs_b, _ = env.reset(seed=5)
            assert env.model.body_mass[body_id] == pytest.approx(nominal_mass)
            assert env.model.geom_friction[geom_id] == pytest.approx(nominal_friction)
            assert not np.allclose(obs_a, obs_b)
        finally:
            env.close()

    def test_reset_rejects_invalid_joint_limits_and_randomization(self) -> None:
        env = MujocoRobotEnvironment()
        try:
            with pytest.raises(ValueError, match="outside"):
                env.reset(options={"joint_positions": {"joint4": 0.0}})
            with pytest.raises(ValueError, match=r"\[0.5, 1.5\]"):
                env.reset(options={"physics_variation": {"cube_mass_scale": 3.0}})
        finally:
            env.close()

    def test_close_releases_safely(self) -> None:
        env = MujocoRobotEnvironment()
        env.reset()
        env.close()
        assert env.is_closed
        with pytest.raises(RuntimeError, match="closed"):
            env.step(np.zeros(env.model.nu, dtype=np.float32))
        with pytest.raises(RuntimeError, match="closed"):
            env.reset()
        env.close()  # idempotent
