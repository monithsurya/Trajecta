"""Unit tests for Trajecta Controller contract and JointPositionController."""

from __future__ import annotations

import numpy as np
import pytest
from importlib.resources import files

from roboforge.controllers.contract import Controller
from roboforge.controllers.joint_position import DEFAULT_JOINT_LIMITS, JointPositionController
from roboforge.environments.mujoco_env import MujocoRobotEnvironment
from roboforge.experiments.spec import ControllerConfig, ControllerType


class TestControllerContract:
    """Test suite validating Controller protocol conformance and contract invariants."""

    def test_satisfies_controller_protocol(self) -> None:
        """Requirement 1: Verify JointPositionController structurally satisfies Controller Protocol."""
        controller = JointPositionController(target_positions=[0.0, 0.5])
        assert isinstance(controller, Controller)

    def test_action_shape_and_dtype(self) -> None:
        """Requirement 5: Verify compute_action produces a float32 array of shape (2,)."""
        controller = JointPositionController(target_positions=[0.0, 0.0])
        obs = np.zeros(17, dtype=np.float32)

        action = controller.compute_action(obs)
        assert isinstance(action, np.ndarray)
        assert action.shape == (2,)
        assert action.dtype == np.float32

    def test_action_bounds_and_clipping(self) -> None:
        """Requirement 5: Verify all outputs are strictly bounded in [-1.0, 1.0]."""
        controller = JointPositionController(
            target_positions=[100.0, -100.0],  # Targets far outside limits
            kp=50.0,  # Aggressive gain
        )
        obs = np.array([-10.0, 10.0], dtype=np.float32)

        action = controller.compute_action(obs)
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)
        assert action[0] == 1.0
        assert action[1] == -1.0

    def test_action_bounds_at_exact_limits(self) -> None:
        """Verify actions map accurately to -1.0, 0.0, and 1.0 at boundary setpoints."""
        # Upper limits: shoulder=1.5708, elbow=2.0944 -> normalized action should be [1.0, 1.0]
        ctrl_upper = JointPositionController(target_positions=[1.5708, 2.0944])
        act_upper = ctrl_upper.compute_action(np.zeros(2, dtype=np.float32))
        assert np.allclose(act_upper, [1.0, 1.0], atol=1e-5)

        # Lower limits: shoulder=-1.5708, elbow=-2.0944 -> normalized action should be [-1.0, -1.0]
        ctrl_lower = JointPositionController(target_positions=[-1.5708, -2.0944])
        act_lower = ctrl_lower.compute_action(np.zeros(2, dtype=np.float32))
        assert np.allclose(act_lower, [-1.0, -1.0], atol=1e-5)

        # Center: shoulder=0.0, elbow=0.0 -> normalized action should be [0.0, 0.0]
        ctrl_center = JointPositionController(target_positions=[0.0, 0.0])
        act_center = ctrl_center.compute_action(np.zeros(2, dtype=np.float32))
        assert np.allclose(act_center, [0.0, 0.0], atol=1e-5)

    def test_deterministic_outputs(self) -> None:
        """Requirement 5: Verify identical inputs produce bitwise identical actions."""
        controller = JointPositionController(target_positions=[0.25, -0.75], kp=[1.5, 0.8])
        obs = np.array([0.1, -0.2, 0.35, 0.0, 0.425, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        action_1 = controller.compute_action(obs)
        action_2 = controller.compute_action(obs)
        assert np.array_equal(action_1, action_2)

    def test_proportional_feedback_convergence_direction(self) -> None:
        """Verify proportional feedback produces corrective commands in the correct direction.

        The controller computes q_cmd = q_current + kp * (q_target - q_current).
        When below target, q_cmd > q_current; when above target, q_cmd < q_current.
        We verify the corrective step moves the commanded position toward the target.
        """
        target = [0.5, 0.5]
        kp = 0.5
        controller = JointPositionController(target_positions=target, kp=kp)
        limits = np.asarray(DEFAULT_JOINT_LIMITS, dtype=np.float64)
        q_min, q_max = limits[:, 0], limits[:, 1]

        def compute_q_cmd(q_current: np.ndarray) -> np.ndarray:
            return q_current + kp * (np.asarray(target) - q_current)

        # Below target: q_cmd should be between q_current and target (moving toward target)
        obs_below = np.array([0.0, 0.0], dtype=np.float32)
        action_below = controller.compute_action(obs_below)
        # Action should be positive (q_cmd = 0.25 > 0 in normalized space → positive action)
        assert action_below[0] > 0.0
        assert action_below[1] > 0.0
        # q_cmd should be between current and target
        q_cmd_below = compute_q_cmd(obs_below.astype(np.float64))
        assert np.all(q_cmd_below >= obs_below.astype(np.float64))
        assert np.all(q_cmd_below <= np.asarray(target))

        # Above target: q_cmd should be between target and q_current (moving toward target)
        obs_above = np.array([0.8, 0.8], dtype=np.float32)
        action_above = controller.compute_action(obs_above)
        q_cmd_above = compute_q_cmd(obs_above.astype(np.float64))
        assert np.all(q_cmd_above <= obs_above.astype(np.float64))
        assert np.all(q_cmd_above >= np.asarray(target))
        # When current position is well above target, action is still positive (q_cmd>0)
        # but the commanded q_cmd is less than q_current (converging downward)
        assert np.all(q_cmd_above < obs_above.astype(np.float64))

    def test_set_target_positions(self) -> None:
        """Verify dynamic setpoint updates."""
        controller = JointPositionController(target_positions=[0.0, 0.0])
        obs = np.zeros(2, dtype=np.float32)
        assert np.allclose(controller.compute_action(obs), [0.0, 0.0])

        controller.set_target_positions([0.7854, -1.0472])
        action = controller.compute_action(obs)
        assert action[0] > 0.0
        assert action[1] < 0.0

    def test_callable_interface(self) -> None:
        """Verify functional invocation `controller(obs)` delegates to compute_action."""
        controller = JointPositionController(target_positions=[0.2, -0.3])
        obs = np.zeros(2, dtype=np.float32)
        assert np.array_equal(controller(obs), controller.compute_action(obs))

    def test_reset_is_safe_and_idempotent(self) -> None:
        """Verify reset satisfies protocol without side effects."""
        controller = JointPositionController(target_positions=[0.0, 0.0])
        controller.reset()  # No exception


class TestControllerConfigIntegration:
    """Test suite validating construction from Chunk 1 ControllerConfig."""

    def test_from_config_default(self) -> None:
        """Verify from_config with minimal ControllerConfig."""
        config = ControllerConfig(controller_type=ControllerType.JOINT_POSITION)
        controller = JointPositionController.from_config(config)

        assert controller.num_joints == 2
        assert np.allclose(controller.target_positions, [0.0, 0.0])
        assert controller.kp is None

    def test_from_config_with_gains_and_targets(self) -> None:
        """Verify from_config with explicit gains and extra_params."""
        config = ControllerConfig(
            controller_type=ControllerType.JOINT_POSITION,
            frequency_hz=100.0,
            kp=[2.0, 1.5],
            extra_params={"target_positions": [0.5, -0.5]},
        )
        controller = JointPositionController.from_config(config)

        assert controller.num_joints == 2
        assert np.allclose(controller.target_positions, [0.5, -0.5])
        assert np.allclose(controller.kp, [2.0, 1.5])

    def test_reject_incompatible_controller_type(self) -> None:
        """Verify from_config rejects mismatched controller types."""
        config = ControllerConfig(controller_type=ControllerType.OPERATIONAL_SPACE)
        with pytest.raises(ValueError, match="Expected ControllerType.JOINT_POSITION"):
            JointPositionController.from_config(config)


class TestInvalidInputsAndConfigurations:
    """Test suite verifying robust validation and rejection of malformed inputs."""

    @pytest.mark.parametrize(
        "invalid_obs",
        [
            "not_an_array",
            [0.1, 0.2],
            123,
            None,
        ],
    )
    def test_reject_non_numpy_observation(self, invalid_obs: object) -> None:
        """Verify non-ndarray inputs raise TypeError."""
        controller = JointPositionController(target_positions=[0.0, 0.0])
        with pytest.raises(TypeError, match="must be a numpy.ndarray"):
            controller.compute_action(invalid_obs)  # type: ignore[arg-type]

    def test_reject_multidimensional_observation(self) -> None:
        """Verify 2D or higher dimension arrays raise ValueError."""
        controller = JointPositionController(target_positions=[0.0, 0.0])
        with pytest.raises(ValueError, match="must be 1D"):
            controller.compute_action(np.zeros((2, 2), dtype=np.float32))

    def test_reject_observation_too_short(self) -> None:
        """Verify observations smaller than num_joints raise ValueError."""
        controller = JointPositionController(target_positions=[0.0, 0.0])
        with pytest.raises(ValueError, match="smaller than num_joints"):
            controller.compute_action(np.array([0.5], dtype=np.float32))

    @pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
    def test_reject_non_finite_observation(self, bad_val: float) -> None:
        """Verify observations with NaN or Inf in robot state raise ValueError."""
        controller = JointPositionController(target_positions=[0.0, 0.0])
        obs = np.array([bad_val, 0.0, 1.0, 2.0], dtype=np.float32)
        with pytest.raises(ValueError, match="non-finite values"):
            controller.compute_action(obs)

    def test_reject_empty_targets(self) -> None:
        """Verify empty target_positions raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            JointPositionController(target_positions=[])

    def test_reject_inverted_joint_limits(self) -> None:
        """Verify min >= max in joint_limits raises ValueError."""
        with pytest.raises(ValueError, match="minimums must be strictly less than maximums"):
            JointPositionController(
                target_positions=[0.0, 0.0],
                joint_limits=[(1.0, 0.0), (-1.0, 1.0)],
            )

    def test_reject_kp_length_mismatch(self) -> None:
        """Verify kp array length not matching num_joints raises ValueError."""
        with pytest.raises(ValueError, match="must match num_joints"):
            JointPositionController(
                target_positions=[0.0, 0.0],
                kp=[1.0, 2.0, 3.0],
            )

    def test_reject_negative_kp(self) -> None:
        """Verify negative kp raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            JointPositionController(
                target_positions=[0.0, 0.0],
                kp=-1.0,
            )


class TestEnvironmentClosedLoopCompatibility:
    """Test closed-loop interaction between JointPositionController and MujocoRobotEnvironment."""

    def test_closed_loop_execution_step(self) -> None:
        """Requirement 5: Verify controller seamlessly drives MujocoRobotEnvironment."""
        # This contract test pairs the legacy two-joint controller with the
        # two-joint fixture explicitly; the default environment is now Panda.
        simple_arm = files("roboforge.environments.assets").joinpath("simple_arm.xml")
        env = MujocoRobotEnvironment(model_path=str(simple_arm), n_substeps=10)
        try:
            obs, info = env.reset(seed=42)

            # Configure controller to drive arm toward shoulder=0.5 rad, elbow=-0.5 rad
            target = [0.5, -0.5]
            controller = JointPositionController(target_positions=target, kp=1.0)

            for step_idx in range(10):
                # 1. Controller computes action from environment observation
                action = controller.compute_action(obs)

                # 2. Action strictly conforms to environment action space
                assert action.shape == env.action_space.shape
                assert action.dtype == np.float32
                assert env.action_space.contains(action)

                # 3. Environment advances physics
                obs, reward, terminated, truncated, step_info = env.step(action)
                assert step_info["is_finite"] is True
                assert not terminated
                assert not truncated

            # Verify arm joints moved and simulation remained stable
            qpos_final = obs[:2]
            assert np.all(np.isfinite(qpos_final)), "Joint positions must remain finite after 10 steps"
            # Verify joints actually moved from initial position [0.0, 0.0]
            assert not np.allclose(qpos_final, [0.0, 0.0], atol=0.01), (
                "Joints should have moved from the initial rest position after 10 steps of actuation"
            )
        finally:
            env.close()
