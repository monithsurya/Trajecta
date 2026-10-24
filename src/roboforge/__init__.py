"""Trajecta: A simulation-first adaptive robot-learning framework.

Core package initialization.
"""

from roboforge.controllers.contract import Controller
from roboforge.controllers.joint_position import DEFAULT_JOINT_LIMITS, JointPositionController
from roboforge.environments.contract import RobotEnvironment
from roboforge.experiments.spec import (
    MAX_SEED_VALUE,
    ControllerConfig,
    ControllerType,
    ExperimentSpec,
    TaskType,
)

__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "MujocoRobotEnvironment":
        from roboforge.environments.mujoco_env import MujocoRobotEnvironment

        return MujocoRobotEnvironment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "MAX_SEED_VALUE",
    "ControllerConfig",
    "ControllerType",
    "ExperimentSpec",
    "TaskType",
    "RobotEnvironment",
    "MujocoRobotEnvironment",
    "Controller",
    "JointPositionController",
    "DEFAULT_JOINT_LIMITS",
]
