"""Unit tests for Trajecta experiment specification and schema validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from roboforge.experiments.spec import (
    MAX_SEED_VALUE,
    ControllerConfig,
    ControllerType,
    ExperimentSpec,
    TaskType,
)


class TestControllerConfig:
    """Test suite for ControllerConfig validation and behavior."""

    def test_valid_default_controller_config(self) -> None:
        """Verify default values when creating a valid ControllerConfig."""
        config = ControllerConfig(controller_type=ControllerType.OPERATIONAL_SPACE)
        assert config.controller_type == ControllerType.OPERATIONAL_SPACE
        assert config.frequency_hz == 50.0
        assert config.kp is None
        assert config.kd is None
        assert config.extra_params == {}

    def test_valid_controller_with_string_type_coercion(self) -> None:
        """Verify string identifiers coerce into ControllerType enum members."""
        config = ControllerConfig(controller_type="joint_position")  # type: ignore[arg-type]
        assert config.controller_type == ControllerType.JOINT_POSITION

    def test_valid_controller_with_custom_gains_and_frequency(self) -> None:
        """Verify explicit frequency, scalar gains, and per-joint gain lists."""
        config_scalar = ControllerConfig(
            controller_type=ControllerType.CARTESIAN_IMPEDANCE,
            frequency_hz=100.0,
            kp=150.0,
            kd=20.0,
            extra_params={"nullspace_damping": 0.1},
        )
        assert config_scalar.frequency_hz == 100.0
        assert config_scalar.kp == 150.0
        assert config_scalar.kd == 20.0
        assert config_scalar.extra_params["nullspace_damping"] == 0.1

        config_vector = ControllerConfig(
            controller_type=ControllerType.JOINT_POSITION,
            frequency_hz=200.0,
            kp=[100.0, 100.0, 80.0, 50.0, 20.0, 10.0, 5.0],
            kd=[10.0, 10.0, 8.0, 5.0, 2.0, 1.0, 0.5],
        )
        assert config_vector.kp == [100.0, 100.0, 80.0, 50.0, 20.0, 10.0, 5.0]
        assert config_vector.kd == [10.0, 10.0, 8.0, 5.0, 2.0, 1.0, 0.5]

    def test_controller_immutability(self) -> None:
        """Verify frozen configuration rejects runtime attribute assignment."""
        config = ControllerConfig(controller_type=ControllerType.OPERATIONAL_SPACE)
        with pytest.raises(ValidationError):
            config.frequency_hz = 100.0  # type: ignore[misc]

    def test_reject_missing_controller_type(self) -> None:
        """Verify controller_type is required."""
        with pytest.raises(ValidationError) as exc_info:
            ControllerConfig()  # type: ignore[call-arg]
        assert "controller_type" in str(exc_info.value)

    def test_reject_invalid_controller_type(self) -> None:
        """Verify unrecognized controller type strings are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ControllerConfig(controller_type="unsupported_exotic_controller")  # type: ignore[arg-type]
        assert "controller_type" in str(exc_info.value)

    @pytest.mark.parametrize("invalid_freq", [0.0, -1.0, -50.0])
    def test_reject_non_positive_frequency(self, invalid_freq: float) -> None:
        """Verify control frequency must be strictly positive (> 0)."""
        with pytest.raises(ValidationError) as exc_info:
            ControllerConfig(
                controller_type=ControllerType.OPERATIONAL_SPACE,
                frequency_hz=invalid_freq,
            )
        assert "frequency_hz" in str(exc_info.value)

    @pytest.mark.parametrize("negative_gain", [-0.01, -10.0])
    def test_reject_negative_scalar_gains(self, negative_gain: float) -> None:
        """Verify negative scalar gains are rejected."""
        with pytest.raises(ValidationError):
            ControllerConfig(
                controller_type=ControllerType.OPERATIONAL_SPACE,
                kp=negative_gain,
            )
        with pytest.raises(ValidationError):
            ControllerConfig(
                controller_type=ControllerType.OPERATIONAL_SPACE,
                kd=negative_gain,
            )

    def test_reject_negative_vector_gain_elements(self) -> None:
        """Verify negative values in gain lists are rejected."""
        with pytest.raises(ValidationError):
            ControllerConfig(
                controller_type=ControllerType.JOINT_POSITION,
                kp=[100.0, -5.0, 50.0],
            )

    def test_reject_non_numeric_gains(self) -> None:
        """Verify non-numeric gain entries are rejected."""
        with pytest.raises(ValidationError):
            ControllerConfig(
                controller_type=ControllerType.JOINT_POSITION,
                kp="high",  # type: ignore[arg-type]
            )
        with pytest.raises(ValidationError):
            ControllerConfig(
                controller_type=ControllerType.JOINT_POSITION,
                kp=[10.0, "invalid"],  # type: ignore[list-item]
            )

    def test_reject_extra_fields(self) -> None:
        """Verify extra unexpected fields are rejected by forbidden extra configuration."""
        with pytest.raises(ValidationError) as exc_info:
            ControllerConfig(
                controller_type=ControllerType.OPERATIONAL_SPACE,
                unexpected_parameter=123,  # type: ignore[call-arg]
            )
        assert "extra_forbidden" in str(exc_info.value) or "unexpected_parameter" in str(exc_info.value)


class TestExperimentSpec:
    """Test suite for ExperimentSpec schema validation and invariants."""

    @pytest.fixture
    def valid_controller(self) -> ControllerConfig:
        return ControllerConfig(controller_type=ControllerType.OPERATIONAL_SPACE)

    def test_valid_minimal_spec(self, valid_controller: ControllerConfig) -> None:
        """Verify creating a minimal valid ExperimentSpec with default seed=None."""
        spec = ExperimentSpec(
            task=TaskType.PICK_AND_PLACE,
            episodes=10,
            controller=valid_controller,
        )
        assert spec.task == TaskType.PICK_AND_PLACE
        assert spec.episodes == 10
        assert spec.seed is None
        assert spec.controller == valid_controller

    def test_valid_spec_from_dict_and_string_coercion(self) -> None:
        """Verify instantiating with raw dictionary and primitive string types."""
        raw_data = {
            "task": "pick_and_place",
            "episodes": 25,
            "seed": 42,
            "controller": {
                "controller_type": "operational_space",
                "frequency_hz": 100.0,
            },
        }
        spec = ExperimentSpec.model_validate(raw_data)
        assert spec.task == TaskType.PICK_AND_PLACE
        assert spec.episodes == 25
        assert spec.seed == 42
        assert spec.controller.controller_type == ControllerType.OPERATIONAL_SPACE
        assert spec.controller.frequency_hz == 100.0

    @pytest.mark.parametrize("seed_val", [0, 1, 42, 123456789, MAX_SEED_VALUE])
    def test_valid_seed_boundaries(
        self, seed_val: int, valid_controller: ControllerConfig
    ) -> None:
        """Verify accepted boundary seed values: 0 up to 2^32 - 1."""
        spec = ExperimentSpec(
            task=TaskType.PICK_AND_PLACE,
            episodes=5,
            seed=seed_val,
            controller=valid_controller,
        )
        assert spec.seed == seed_val

    def test_spec_immutability(self, valid_controller: ControllerConfig) -> None:
        """Verify frozen ExperimentSpec rejects attribute mutation."""
        spec = ExperimentSpec(
            task=TaskType.PICK_AND_PLACE,
            episodes=10,
            controller=valid_controller,
        )
        with pytest.raises(ValidationError):
            spec.episodes = 20  # type: ignore[misc]

    def test_spec_serialization_roundtrip(
        self, valid_controller: ControllerConfig
    ) -> None:
        """Verify JSON and dict serialization/deserialization preservation."""
        original = ExperimentSpec(
            task=TaskType.PICK_AND_PLACE,
            episodes=100,
            seed=2026,
            controller=valid_controller,
        )
        json_data = original.model_dump_json()
        restored = ExperimentSpec.model_validate_json(json_data)
        assert original == restored
        assert original.model_dump() == restored.model_dump()

    # --- Rejection of Invalid Inputs ---

    @pytest.mark.parametrize("missing_field", ["task", "episodes", "controller"])
    def test_reject_missing_required_fields(self, missing_field: str) -> None:
        """Verify that omitting any required field fails validation."""
        valid_args = {
            "task": TaskType.PICK_AND_PLACE,
            "episodes": 10,
            "controller": ControllerConfig(controller_type=ControllerType.OPERATIONAL_SPACE),
        }
        del valid_args[missing_field]
        with pytest.raises(ValidationError) as exc_info:
            ExperimentSpec(**valid_args)
        assert missing_field in str(exc_info.value)

    @pytest.mark.parametrize("invalid_task", ["invalid_task", "pick-and-place", "", 123])
    def test_reject_invalid_task_identifier(
        self, invalid_task: object, valid_controller: ControllerConfig
    ) -> None:
        """Verify invalid or unconfigured task identifiers are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ExperimentSpec(
                task=invalid_task,  # type: ignore[arg-type]
                episodes=10,
                controller=valid_controller,
            )
        assert "task" in str(exc_info.value)

    @pytest.mark.parametrize("invalid_episodes", [0, -1, -50])
    def test_reject_non_positive_episode_counts(
        self, invalid_episodes: int, valid_controller: ControllerConfig
    ) -> None:
        """Verify episode count must be strictly greater than zero."""
        with pytest.raises(ValidationError) as exc_info:
            ExperimentSpec(
                task=TaskType.PICK_AND_PLACE,
                episodes=invalid_episodes,
                controller=valid_controller,
            )
        assert "episodes" in str(exc_info.value)

    @pytest.mark.parametrize("invalid_episodes", [True, False, 3.14, "twenty"])
    def test_reject_non_integer_or_boolean_episodes(
        self, invalid_episodes: object, valid_controller: ControllerConfig
    ) -> None:
        """Verify boolean, float, and non-numeric values are rejected for episode count."""
        with pytest.raises(ValidationError) as exc_info:
            ExperimentSpec(
                task=TaskType.PICK_AND_PLACE,
                episodes=invalid_episodes,  # type: ignore[arg-type]
                controller=valid_controller,
            )
        assert "episodes" in str(exc_info.value)

    @pytest.mark.parametrize("invalid_seed", [-1, -100, MAX_SEED_VALUE + 1, 10**11])
    def test_reject_out_of_range_seeds(
        self, invalid_seed: int, valid_controller: ControllerConfig
    ) -> None:
        """Verify negative seeds and seeds exceeding 2^32 - 1 are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ExperimentSpec(
                task=TaskType.PICK_AND_PLACE,
                episodes=10,
                seed=invalid_seed,
                controller=valid_controller,
            )
        assert "seed" in str(exc_info.value)

    @pytest.mark.parametrize("invalid_seed", [True, False, "random_seed", 3.1415])
    def test_reject_non_integer_or_boolean_seeds(
        self, invalid_seed: object, valid_controller: ControllerConfig
    ) -> None:
        """Verify non-integer types and booleans are rejected for seed."""
        with pytest.raises(ValidationError) as exc_info:
            ExperimentSpec(
                task=TaskType.PICK_AND_PLACE,
                episodes=10,
                seed=invalid_seed,  # type: ignore[arg-type]
                controller=valid_controller,
            )
        assert "seed" in str(exc_info.value)

    def test_reject_malformed_controller_configuration(self) -> None:
        """Verify malformed nested controller configurations bubble up validation errors."""
        with pytest.raises(ValidationError) as exc_info:
            ExperimentSpec.model_validate(
                {
                    "task": "pick_and_place",
                    "episodes": 10,
                    "controller": {
                        "controller_type": "joint_position",
                        "frequency_hz": -10.0,  # Invalid negative frequency
                    },
                }
            )
        assert "frequency_hz" in str(exc_info.value)

    def test_reject_extra_fields(self, valid_controller: ControllerConfig) -> None:
        """Verify extra unmodeled fields on ExperimentSpec are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            ExperimentSpec(
                task=TaskType.PICK_AND_PLACE,
                episodes=10,
                controller=valid_controller,
                simulator="mujoco",  # type: ignore[call-arg]
            )
        assert "extra_forbidden" in str(exc_info.value) or "simulator" in str(exc_info.value)
