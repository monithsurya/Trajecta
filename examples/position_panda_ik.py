#!/usr/bin/env python3
"""Move the Panda hand a few centimeters with position-only inverse kinematics.

macOS (from the project root):
    PYTHONPATH=src ./venv/bin/python ./venv/bin/mjpython examples/position_panda_ik.py
"""

from __future__ import annotations

import argparse
import sys
import time

import mujoco
import numpy as np

from roboforge.controllers.end_effector_position import PandaEndEffectorPositionController
from roboforge.environments.mujoco_env import MujocoRobotEnvironment
from roboforge.visualization.viewer import is_running_under_mjpython


def run_demo(*, no_ui: bool = False, tolerance: float = 0.01) -> bool:
    if sys.platform == "darwin" and not is_running_under_mjpython():
        raise RuntimeError(
            "On macOS, launch with: PYTHONPATH=src ./venv/bin/python "
            "./venv/bin/mjpython examples/position_panda_ik.py"
        )
    env = MujocoRobotEnvironment(max_episode_steps=100_000)
    try:
        observation, _ = env.reset()
        controller = PandaEndEffectorPositionController(env, position_tolerance=min(tolerance, 0.005))
        hand_id = controller.ik.body_id
        start_q = env.data.qpos[controller.ik.qpos_indices].copy()
        start_position = env.data.xpos[hand_id].copy()
        # Build a known nearby reachable target by a small safe joint offset.
        delta = np.array([0.035, -0.025, 0.02, 0.0, 0.02, -0.015, 0.025])
        target_q = np.clip(start_q + delta, controller.ik.limits[:, 0], controller.ik.limits[:, 1])
        target = controller.ik.forward_position(target_q)
        table_top = 0.70
        if target[2] <= table_top + 0.05:
            raise RuntimeError(f"Chosen target is too close to the table: z={target[2]:.3f} m")
        result = controller.set_target_position(target)
        print(f"Hand start (world m): {start_position.round(4).tolist()}")
        print(f"Hand target (world m): {target.round(4).tolist()}")
        print(f"IK: converged={result.converged}, error={result.position_error:.4f} m, iterations={result.iterations}")
        if not result.converged:
            print("IK did not converge within its configured tolerance; stopping without motion.")
            return False

        dt = float(env.model.opt.timestep * env.n_substeps)
        move_duration = 2.5
        total_duration = 2 * move_duration + 0.5
        started = time.monotonic()
        reached = False
        with mujoco.viewer.launch_passive(
            env.model, env.data, show_left_ui=not no_ui, show_right_ui=not no_ui,
        ) as viewer:
            while viewer.is_running() and time.monotonic() - started < total_duration:
                frame_started = time.monotonic()
                elapsed = frame_started - started
                if elapsed <= move_duration:
                    fraction = np.clip(elapsed / move_duration, 0.0, 1.0)
                    joint_target = start_q + fraction * (result.joint_positions - start_q)
                else:
                    fraction = np.clip((elapsed - move_duration) / move_duration, 0.0, 1.0)
                    joint_target = result.joint_positions + fraction * (start_q - result.joint_positions)
                controller.set_joint_target(joint_target)
                action = controller.compute_action(observation)
                observation, _, terminated, truncated, _ = env.step(action)
                viewer.sync()
                if elapsed >= move_duration and not reached:
                    current_position = env.data.xpos[hand_id].copy()
                    error = float(np.linalg.norm(target - current_position))
                    reached = error <= tolerance
                    print(f"Reached target within {tolerance:.3f} m: {reached} (current error {error:.4f} m)")
                delay = dt - (time.monotonic() - frame_started)
                if delay > 0:
                    time.sleep(delay)
                if terminated or truncated:
                    break
        return reached
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-ui", action="store_true", help="Hide MuJoCo side panels.")
    parser.add_argument("--tolerance", type=float, default=0.01, help="Reached-target tolerance in meters.")
    args = parser.parse_args(argv)
    return 0 if run_demo(no_ui=args.no_ui, tolerance=args.tolerance) else 1


if __name__ == "__main__":
    raise SystemExit(main())
