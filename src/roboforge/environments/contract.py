"""Typed robot environment contract for Trajecta.

This module defines the structural interface (Protocol) that all robot environments
must satisfy in Trajecta, following Gymnasium reset/step/close and spaces semantics.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

import gymnasium.spaces as spaces

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@runtime_checkable
class RobotEnvironment(Protocol[ObsType, ActType]):
    """Structural protocol defining the environment contract for Trajecta.

    Any environment—whether wrapping a physical robot, a simulation engine
    (e.g., MuJoCo, Isaac Lab), or a deterministic test harness—must satisfy
    this interface. Conformance is structural (duck typing with static validation),
    meaning classes do not strictly need to inherit from this protocol as long
    as they implement the required properties and methods matching Gymnasium conventions.

    Type Parameters:
        ObsType: The type of observation returned by the environment.
        ActType: The type of action accepted by the environment.
    """

    @property
    def observation_space(self) -> spaces.Space[ObsType]:
        """The Space object defining valid observations."""
        ...

    @property
    def action_space(self) -> spaces.Space[ActType]:
        """The Space object defining valid actions."""
        ...

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment to an initial state.

        Args:
            seed: Optional integer seed for deterministic state/RNG initialization.
            options: Optional dictionary of additional reset settings.

        Returns:
            A tuple of:
                - observation: The initial observation conforming to observation_space.
                - info: Auxiliary diagnostic or debugging information dictionary.
        """
        ...

    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Run one timestep of environment dynamics using the supplied action.

        Args:
            action: An action conforming to action_space.

        Returns:
            A 5-part tuple of:
                - observation: Next agent observation conforming to observation_space.
                - reward: Numeric scalar reward resulting from taking the action.
                - terminated: Boolean flag indicating if a terminal state was reached
                  (e.g., task success, goal achieved, out of bounds, robot failure).
                  Episode ended due to natural task conditions.
                - truncated: Boolean flag indicating if execution was cut short due
                  to an external boundary condition (e.g., max episode timestep limit
                  exceeded, timeout, user interrupt).
                - info: Diagnostic and debugging dictionary.
        """
        ...

    def close(self) -> None:
        """Close the environment and release any held resources.

        This should cleanly terminate any external connections, simulator handles,
        graphics pipelines, or hardware controllers.
        """
        ...
