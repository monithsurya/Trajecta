"""Non-graphical safety checks for the Panda control demonstration."""

from __future__ import annotations

import numpy as np

from examples.control_panda import PandaDemoController
from roboforge.environments.mujoco_env import MujocoRobotEnvironment


def test_demo_produces_bounded_environment_actions_and_returns_to_open_home() -> None:
    env = MujocoRobotEnvironment(max_episode_steps=100_000)
    try:
        observation, reset_info = env.reset(seed=9)
        controller = PandaDemoController(env, observation)
        dt = env.model.opt.timestep * env.n_substeps
        controller.set_control_period(dt)
        gripper_index = next(
            spec["index"] for spec in env.actuator_specs
            if spec["transmission"] == "tendon" and spec["target"] == "split"
        )
        home = np.array([reset_info["joint_positions"][f"joint{i}"] for i in range(1, 8)])
        sampled_gripper_commands: dict[int, float] = {}
        sampled_arm_states: dict[int, np.ndarray] = {}
        sampled_opening_width: dict[int, float] = {}
        for step in range(252):
            action = controller.compute_action(observation)
            assert env.action_space.contains(action)
            assert np.isfinite(action).all()
            observation, _, _, _, info = env.step(action)
            if step in (0, 200, 250, 251):
                sampled_gripper_commands[step] = float(action[gripper_index])
            if step in (74, 149):
                sampled_arm_states[step] = np.array([
                    info["joint_positions"][f"joint{i}"] for i in range(1, 8)
                ])
            if step in (199, 251):
                sampled_opening_width[step] = info["gripper_state"]["opening_width"]

        assert np.max(np.abs(sampled_arm_states[74] - home)) > 0.03
        assert np.allclose(sampled_arm_states[149], home, atol=0.03)
        assert sampled_gripper_commands[0] == 1.0
        assert sampled_gripper_commands[200] == -1.0
        assert sampled_gripper_commands[250] == 1.0
        assert sampled_gripper_commands[251] == 1.0
        assert env.observation_space.contains(observation)
        assert np.isfinite(observation).all()
        assert sampled_opening_width[199] < 0.02
        assert sampled_opening_width[251] > sampled_opening_width[199]
    finally:
        env.close()
