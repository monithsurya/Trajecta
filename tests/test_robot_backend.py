"""Backend contract tests: MuJoCo is exercised, hardware remains unavailable."""

import pytest

from roboforge.environments.mujoco_env import MujocoRobotEnvironment
from roboforge.environments.robot_backend import (
    MujocoPandaBackend,
    RobotBackend,
    create_robot_backend,
)


def test_mujoco_backend_exposes_simulation_only_capabilities_and_state():
    env = MujocoRobotEnvironment()
    backend = create_robot_backend("mujoco", environment=env)
    try:
        assert isinstance(backend, RobotBackend)
        assert backend.capabilities.simulation
        assert not backend.capabilities.hardware_enabled
        state = backend.read_state()
        assert len(state["joint_positions"]) == 9
        assert len(state["hand_position_m"]) == 3
        assert len(state["cube_position_m"]) == 3
    finally:
        backend.close()


def test_backend_rejects_invalid_joint_gripper_commands_and_hardware():
    env = MujocoRobotEnvironment()
    backend = MujocoPandaBackend(env)
    try:
        with pytest.raises(ValueError, match="outside"):
            backend.command_joint_positions({"joint4": 0.0})
        with pytest.raises(ValueError, match="finite"):
            backend.command_joint_positions({"joint1": float("nan")})
        with pytest.raises(ValueError, match="opening_fraction"):
            backend.command_gripper(1.2)
        with pytest.raises(NotImplementedError, match="not implemented"):
            create_robot_backend("hardware")
    finally:
        env.close()


def test_backend_joint_command_is_bounded_and_reports_timeout():
    env = MujocoRobotEnvironment(n_substeps=1)
    backend = MujocoPandaBackend(env)
    try:
        state = backend.read_state()
        target = state["joint_positions"]["joint1"] + 0.05
        result = backend.command_joint_positions({"joint1": target}, max_steps=1)
        assert result.steps == 1
        assert result.completed or result.timed_out
        assert result.completed != result.timed_out
        with pytest.raises(ValueError, match="max_steps"):
            backend.command_joint_positions({"joint1": target}, max_steps=0)
    finally:
        env.close()
