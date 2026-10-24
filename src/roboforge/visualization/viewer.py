"""MuJoCo viewer entry points for the Trajecta Panda scene."""

from __future__ import annotations

import argparse
import sys
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from roboforge.controllers.contract import Controller
    from roboforge.environments.mujoco_env import MujocoRobotEnvironment


def is_running_under_mjpython() -> bool:
    """Return whether MuJoCo's macOS mjpython trampoline is active."""
    import mujoco.viewer

    return getattr(mujoco.viewer, "_MJPYTHON", None) is not None


def launch_interactive_viewer(
    env: MujocoRobotEnvironment | None = None,
    *,
    show_left_ui: bool = True,
    show_right_ui: bool = True,
) -> None:
    """Open MuJoCo's interactive viewer for the default Panda scene or `env`."""
    import mujoco.viewer
    from roboforge.environments.mujoco_env import MujocoRobotEnvironment

    managed_env = env if env is not None else MujocoRobotEnvironment()
    try:
        mujoco.viewer.launch(
            managed_env.model,
            managed_env.data,
            show_left_ui=show_left_ui,
            show_right_ui=show_right_ui,
        )
    finally:
        if not managed_env.is_closed:
            managed_env.close()


def launch_passive_viewer(
    env: MujocoRobotEnvironment | None = None,
    controller: Controller | None = None,
    *,
    max_duration: float | None = None,
    action: np.ndarray | None = None,
    show_left_ui: bool = True,
    show_right_ui: bool = True,
) -> None:
    """Open a passive viewer and drive simulation steps from Python."""
    import mujoco.viewer
    from roboforge.environments.mujoco_env import MujocoRobotEnvironment

    if sys.platform == "darwin" and not is_running_under_mjpython():
        raise RuntimeError(
            "Passive viewer mode on macOS requires `mjpython`. Run:\n"
            "PYTHONPATH=src ./venv/bin/mjpython -m roboforge.visualization --passive\n"
            "Or use the interactive viewer with: PYTHONPATH=src ./venv/bin/python -m roboforge.visualization"
        )

    managed_env = env if env is not None else MujocoRobotEnvironment()
    applied_action = (
        np.asarray(action, dtype=np.float32)
        if action is not None
        else np.zeros(managed_env.action_space.shape, dtype=np.float32)
    )
    try:
        obs, _ = managed_env.reset()
        if controller is not None:
            controller.reset()
        dt = float(managed_env.model.opt.timestep * managed_env.n_substeps)
        start_time = time.monotonic()
        with mujoco.viewer.launch_passive(
            managed_env.model,
            managed_env.data,
            show_left_ui=show_left_ui,
            show_right_ui=show_right_ui,
        ) as viewer:
            while viewer.is_running():
                step_start = time.monotonic()
                if max_duration is not None and step_start - start_time >= max_duration:
                    break
                act = controller.compute_action(obs) if controller is not None else applied_action
                obs, _, terminated, truncated, _ = managed_env.step(act)
                viewer.sync()
                elapsed = time.monotonic() - step_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                if terminated or truncated:
                    obs, _ = managed_env.reset()
                    if controller is not None:
                        controller.reset()
    finally:
        if not managed_env.is_closed:
            managed_env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="trajecta-viewer",
        description="Launch the Trajecta Franka Panda MuJoCo scene.",
    )
    parser.add_argument(
        "--passive",
        action="store_true",
        help="Drive steps from Python (requires mjpython on macOS).",
    )
    parser.add_argument("--duration", type=float, default=None, help="Optional run duration in seconds.")
    parser.add_argument("--no-ui", action="store_true", help="Hide both MuJoCo side panels.")
    args = parser.parse_args(argv)
    show_ui = not args.no_ui
    print("Trajecta Panda viewer: Franka Emika Panda, table, and target cube")
    if args.passive:
        launch_passive_viewer(
            max_duration=args.duration,
            show_left_ui=show_ui,
            show_right_ui=show_ui,
        )
    else:
        launch_interactive_viewer(show_left_ui=show_ui, show_right_ui=show_ui)
    return 0
