"""Controller interfaces and protocols for Trajecta.

Defines the contract for robot controllers, decoupling decision-making logic
from environment dynamics, physics simulation, and experiment orchestration.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

import numpy as np

ObsType_contra = TypeVar("ObsType_contra", contravariant=True)
ActType_co = TypeVar("ActType_co", covariant=True)


@runtime_checkable
class Controller(Protocol[ObsType_contra, ActType_co]):
    """Structural protocol defining the controller contract for Trajecta.

    A Controller consumes observations (or state estimates) and deterministically
    produces action commands conforming to the target environment's action space.

    Conforming classes do not strictly need to inherit from this protocol as long
    as they implement the required methods with matching signatures.

    Type Parameters:
        ObsType_contra: Observation type accepted (contravariant).
        ActType_co: Action type produced (covariant).
    """

    def compute_action(self, observation: ObsType_contra) -> ActType_co:
        """Compute a control action from an environment observation.

        Args:
            observation: Current observation conforming to the environment's observation space.

        Returns:
            Control action conforming to the environment's action space.
        """
        ...

    def reset(self) -> None:
        """Reset internal controller state (e.g., integrators, timers, filters)."""
        ...
