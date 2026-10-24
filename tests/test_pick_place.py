"""Pick-and-place state-machine tests plus a deterministic MuJoCo trial."""

from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
import mujoco

from roboforge.controllers.end_effector_position import PandaEndEffectorPositionController
from roboforge.environments.mujoco_env import MujocoRobotEnvironment
from roboforge.experiments.evaluation import ExperimentEvaluator
from roboforge.experiments.experience import experience_from_result
from roboforge.experiments.transitions import JsonlTransitionStore, TransitionDataset
from roboforge.tasks.pick_place import PickPlaceConfig, PickPlaceResult, PickPlaceTask, TaskPhase


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


def test_pick_place_replans_from_live_cube_position_with_bounded_count(fake_task):
    task, env = fake_task
    cube_body = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "target_cube")
    cube_joint = int(env.model.body_jntadr[cube_body])
    qpos_address = int(env.model.jnt_qposadr[cube_joint])
    env.data.qpos[qpos_address] += 0.02
    mujoco.mj_forward(env.model, env.data)
    before = len(task.phase_history)
    task.step()
    assert task.replanning_count == 1
    assert len(task.phase_history) == before + 1


def test_step_recording_is_optional_and_persists_completed_step_on_task_error(fake_task, tmp_path):
    task, env = fake_task
    task._check_phase_progress = lambda dt: None
    task.step()
    assert task._transition_recorder is None

    task.reset(seed=2)
    store = JsonlTransitionStore(tmp_path / "transitions.jsonl")
    task.configure_transition_recording(
        recorder=store, experiment_id=uuid4(), episode_id=uuid4(), seed=2,
    )
    observation = task._obs.copy()
    task._check_phase_progress = lambda dt: (_ for _ in ()).throw(RuntimeError("synthetic task error"))
    with pytest.raises(RuntimeError, match="synthetic task error"):
        task.step()
    recorded = store.read_all()[0]
    assert len(store.read_all()) == 1
    assert recorded.observation == tuple(float(x) for x in observation)
    assert recorded.action == tuple(float(x) for x in env.actions[-1])


def test_pick_place_config_rejects_invalid_destination():
    with pytest.raises(ValueError):
        PickPlaceConfig(destination_xy_m=(np.nan, 0.0))


def test_pick_place_config_rejects_unsafe_variation_and_recovery_bounds():
    with pytest.raises(ValueError, match="mass_scale_range"):
        PickPlaceConfig(cube_mass_scale_range=(0.1, 1.0))
    with pytest.raises(ValueError, match="max_recovery_attempts"):
        PickPlaceConfig(max_recovery_attempts=4)
    with pytest.raises(ValueError, match="joint_perturbation"):
        PickPlaceConfig(robot_joint_perturbation_rad=0.5)


def test_randomized_reset_conditions_are_seeded_and_recorded():
    config = PickPlaceConfig(
        cube_start_bounds_m=((0.40, -0.08, 0.725), (0.50, 0.08, 0.725)),
        destination_bounds_xy_m=((0.62, -0.18), (0.68, -0.12)),
        robot_joint_perturbation_rad=0.02,
        cube_mass_scale_range=(0.9, 1.1),
        cube_friction_scale_range=(0.9, 1.1),
    )
    env = MujocoRobotEnvironment()
    try:
        task = PickPlaceTask(env, PandaEndEffectorPositionController(env), config)
        task.reset(seed=21)
        first = dict(task.condition_parameters)
        task.reset(seed=21)
        assert task.condition_parameters == first
        assert 0.9 <= task.condition_parameters["cube_mass_scale"] <= 1.1
        assert 0.9 <= task.condition_parameters["cube_friction_scale"] <= 1.1
        assert 0.40 <= task.condition_parameters["cube_position_m"][0] <= 0.50
        assert 0.62 <= task.condition_parameters["destination_xy_m"][0] <= 0.68
    finally:
        env.close()


def test_bounded_recovery_retries_only_classified_failure(monkeypatch):
    env = MujocoRobotEnvironment()
    try:
        task = PickPlaceTask(
            env,
            PandaEndEffectorPositionController(env),
            PickPlaceConfig(max_recovery_attempts=2),
        )
        outcomes = iter((
            PickPlaceResult(False, "no bilateral finger contact", ("VERIFY_GRASP",), 10, (0, 0, 0), (0, 0, 0), None, False, failure_category="missed_grasp"),
            PickPlaceResult(True, None, ("SUCCESS",), 12, (0, 0, 0), (0, 0, 0), 0.01, True),
        ))
        attempts = []

        def fake_run_once(*, seed, max_steps):
            attempts.append((seed, task.config.hand_to_fingertip_m))
            return next(outcomes)

        monkeypatch.setattr(task, "_run_once", fake_run_once)
        result = task.run(seed=30)
        assert result.success
        assert result.recovery_attempts == 1
        assert result.episode_length == 22
        assert attempts[0][0] == 30 and attempts[1][0] == 31
        assert attempts[0][1] != attempts[1][1]
    finally:
        env.close()


def test_real_scene_completes_and_records_contact_verified_pick_and_place(tmp_path):
    env = MujocoRobotEnvironment(n_substeps=10, max_episode_steps=10_000)
    try:
        task = PickPlaceTask(env, PandaEndEffectorPositionController(env))
        transition_store = JsonlTransitionStore(tmp_path / "transitions.jsonl")
        experiment_id, episode_id = uuid4(), uuid4()
        task.configure_transition_recording(
            recorder=transition_store, experiment_id=experiment_id,
            episode_id=episode_id, seed=4,
        )
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
        assert record.failure_category is None
        assert record.recovery_attempts == 0
        assert record.replanning_count == 0
        assert record.condition_parameters["cube_mass_scale"] == pytest.approx(1.0)
        assert ExperimentEvaluator().evaluate([result]).successful_episodes == 1
        transitions = TransitionDataset.from_jsonl(transition_store.path).records
        assert len(transitions) == result.episode_length
        assert [item.transition_index for item in transitions] == list(range(result.episode_length))
        assert all(item.experiment_id == experiment_id and item.episode_id == episode_id for item in transitions)
        assert transitions[0].observation != transitions[0].next_observation
        assert transitions[-1].task_success is True
        assert transitions[-1].task_phase == TaskPhase.VERIFY_PLACEMENT.value
    finally:
        env.close()
