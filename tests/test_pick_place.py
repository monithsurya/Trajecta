"""Pick-and-place state-machine tests plus a deterministic MuJoCo trial."""

from types import SimpleNamespace

import numpy as np
import pytest
import mujoco

from roboforge.controllers.end_effector_position import PandaEndEffectorPositionController
from roboforge.environments.mujoco_env import MujocoRobotEnvironment
from roboforge.experiments.evaluation import ExperimentEvaluator
from roboforge.experiments.experience import experience_from_result
from roboforge.tasks.pick_place import PickPlaceConfig, PickPlaceTask, TaskPhase


class FakeEnvironment:
    """Fake step interface over a loaded model for focused task logic tests."""

    def __init__(self):
        self.sim = MujocoRobotEnvironment(max_episode_steps=20)
        self.model, self.data = self.sim.model, self.sim.data
        self.n_substeps = self.sim.n_substeps
        self.actuator_specs = self.sim.actuator_specs
        self.actions = []

    def reset(self, *, seed=None):
        return self.sim.reset(seed=seed)

    def joint_index(self, name):
        return self.sim.joint_index(name)

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        return self.sim._get_obs(), 0.0, False, False, {}

    def close(self):
        self.sim.close()


class FakeController:
    def __init__(self, env, *, converged=True):
        self.env = env
        arm_joint_ids = np.array([env.joint_index(f"joint{i}") for i in range(1, 8)])
        self.ik = SimpleNamespace(
            body_id=mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "hand"),
            qpos_indices=env.model.jnt_qposadr[arm_joint_ids].copy(),
            dof_indices=env.model.jnt_dofadr[arm_joint_ids].copy(),
        )
        self.converged = converged
        self.target = None

    def reset(self):
        pass

    def set_target_position(self, position, *, world_rotation=None):
        self.target = np.asarray(position).copy()
        return SimpleNamespace(
            converged=self.converged,
            position_error=0.0 if self.converged else 0.2,
            orientation_error=0.0 if self.converged else 0.5,
            joint_positions=self.env.data.qpos[self.ik.qpos_indices].copy(),
        )

    def set_joint_target(self, positions):
        pass

    def compute_action(self, observation):
        return np.zeros(self.env.model.nu, dtype=np.float32)


@pytest.fixture
def fake_task():
    env = FakeEnvironment()
    controller = FakeController(env)
    task = PickPlaceTask(env, controller)
    task.reset(seed=1)
    yield task, env
    env.close()


def test_fake_components_advance_and_fail_on_missing_finger_contact(fake_task):
    task, _ = fake_task
    assert task.phase == TaskPhase.APPROACH
    task._advance()
    assert task.phase == TaskPhase.ALIGN
    task.phase = TaskPhase.CLOSE_GRIPPER
    task._bilateral_grasp_contacts = lambda: False
    task._advance()
    assert task.result is not None
    assert not task.result.success
    assert "both fingers" in task.result.failure_reason


def test_fake_controller_ik_failure_is_reported(fake_task):
    task, _ = fake_task
    task.controller.converged = False
    task.reset(seed=1)
    assert task.result is not None
    assert not task.result.success
    assert "IK did not converge" in task.result.failure_reason


def test_fake_release_action_uses_open_gripper_command(fake_task):
    task, env = fake_task
    task.phase = TaskPhase.RELEASE
    task._phase_target = None
    task._target_q = None
    task._check_phase_progress = lambda dt: None
    task.step()
    assert env.actions[-1][task._gripper_action_index] == 1.0


def test_pick_place_config_rejects_invalid_destination():
    with pytest.raises(ValueError):
        PickPlaceConfig(destination_xy_m=(np.nan, 0.0))


def test_real_scene_completes_contact_verified_pick_and_place():
    env = MujocoRobotEnvironment(n_substeps=10, max_episode_steps=10_000)
    try:
        task = PickPlaceTask(env, PandaEndEffectorPositionController(env))
        result = task.run(seed=4, max_steps=4_000)
        assert result.success, result.failure_reason
        assert result.grasp_verified
        assert result.final_placement_error_m <= task.config.placement_tolerance_m
        assert result.phase_history[-1] == TaskPhase.SUCCESS.value
        assert result.episode_length > 0
        record = experience_from_result(
            result, task_id="pick_and_place", environment_id="mujoco_panda_scene", seed=4,
        )
        assert record.grasp_verified is True
        assert ExperimentEvaluator().evaluate([result]).successful_episodes == 1
    finally:
        env.close()
