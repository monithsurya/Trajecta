"""Tests for Panda world-position IK and its established action interface."""

import mujoco
import numpy as np
import pytest

from roboforge.controllers.end_effector_position import PandaEndEffectorPositionController
from roboforge.environments.mujoco_env import MujocoRobotEnvironment
from roboforge.environments.mujoco_ik import MujocoPositionIK


@pytest.fixture
def env():
    instance = MujocoRobotEnvironment(max_episode_steps=100)
    instance.reset(seed=11)
    yield instance
    instance.close()


def test_reachable_target_converges_within_joint_limits(env):
    ik = MujocoPositionIK(env, position_tolerance=0.003)
    start = env.data.qpos[ik.qpos_indices].copy()
    target_q = start + np.array([0.04, -0.03, 0.025, 0.0, 0.02, -0.015, 0.03])
    target = ik.forward_position(target_q)
    result = ik.solve(target)
    assert result.converged
    assert result.position_error <= 0.003
    assert result.iterations <= ik.max_iterations
    assert np.isfinite(result.joint_positions).all()
    assert np.all(result.joint_positions >= ik.limits[:, 0])
    assert np.all(result.joint_positions <= ik.limits[:, 1])


def test_unreachable_target_returns_bounded_finite_best_effort(env):
    ik = MujocoPositionIK(env, max_iterations=30)
    result = ik.solve(np.array([100.0, -100.0, 100.0]))
    assert not result.converged
    assert result.iterations == 30
    assert np.isfinite(result.joint_positions).all()
    assert np.isfinite(result.achieved_position).all()
    assert np.isfinite(result.position_error)
    assert np.all(result.joint_positions >= ik.limits[:, 0])
    assert np.all(result.joint_positions <= ik.limits[:, 1])


@pytest.mark.parametrize("target", [[1, 2], [np.nan, 0, 0], [0, np.inf, 0]])
def test_invalid_target_is_rejected(env, target):
    with pytest.raises(ValueError):
        MujocoPositionIK(env).solve(np.asarray(target))


def test_invalid_initial_joint_positions_are_rejected(env):
    with pytest.raises(ValueError):
        MujocoPositionIK(env).solve([0.5, 0.0, 0.8], initial_positions=np.zeros(6))


def test_ik_and_forward_evaluation_preserve_simulator_state(env):
    ik = MujocoPositionIK(env)
    env.data.ctrl[:] = np.linspace(-0.5, 0.5, env.model.nu)
    mujoco.mj_step(env.model, env.data)
    names = ("qpos", "qvel", "ctrl", "qacc", "xpos")
    before = {name: getattr(env.data, name).copy() for name in names}
    time_before = env.data.time
    current = env.data.qpos[ik.qpos_indices].copy()
    target = ik.forward_position(current + np.array([0.02, 0, 0, 0, 0, 0, 0]))
    ik.solve(target)
    for name in names:
        np.testing.assert_array_equal(getattr(env.data, name), before[name])
    assert env.data.time == time_before


def test_world_position_controller_outputs_environment_action(env):
    _, _ = env.reset()
    controller = PandaEndEffectorPositionController(env)
    q = env.data.qpos[controller.ik.qpos_indices].copy()
    target = controller.ik.forward_position(q + np.array([0.025, 0, 0, 0, 0, 0, 0]))
    result = controller.set_target_position(target)
    assert result.converged
    action = controller.compute_action(env._get_obs())
    assert action.shape == env.action_space.shape
    assert env.action_space.contains(action)
    gripper = next(s for s in env.actuator_specs if s["transmission"] == "tendon")
    assert action[gripper["index"]] == 1.0
    obs, _, _, _, _ = env.step(action)
    assert env.observation_space.contains(obs)
    assert np.isfinite(obs).all()


def test_pose_ik_preserves_requested_gripper_orientation(env):
    ik = MujocoPositionIK(env)
    rotation = env.data.xmat[ik.body_id].reshape(3, 3).copy()
    result = ik.solve(np.array([0.45, 0.0, 0.85]), target_rotation=rotation)
    assert result.converged
    assert result.position_error <= ik.position_tolerance
    assert result.orientation_error <= 0.01
    achieved = ik.forward_position(result.joint_positions)
    np.testing.assert_allclose(achieved, [0.45, 0.0, 0.85], atol=ik.position_tolerance)
