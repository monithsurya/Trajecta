"""Trajecta environments package.

Provides the environment contract and simulator adapters.
"""

from roboforge.environments.contract import RobotEnvironment
from roboforge.environments.mujoco_env import MujocoRobotEnvironment

__all__ = [
    "RobotEnvironment",
    "MujocoRobotEnvironment",
]
