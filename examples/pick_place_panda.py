#!/usr/bin/env python3
"""Run the contact-based Panda pick-and-place task in MuJoCo's viewer.

On macOS:
    PYTHONPATH=src ./venv/bin/python ./venv/bin/mjpython examples/pick_place_panda.py --vary-conditions --seed 100
"""

from __future__ import annotations

import argparse
import sys
import time

import mujoco

from roboforge.controllers.end_effector_position import PandaEndEffectorPositionController
from roboforge.environments.mujoco_env import MujocoRobotEnvironment
from roboforge.tasks.pick_place import PickPlaceConfig, PickPlaceTask
from roboforge.visualization.viewer import is_running_under_mjpython


def run_demo(*, seed: int = 0, no_ui: bool = False, vary_conditions: bool = False) -> bool:
    if sys.platform == "darwin" and not is_running_under_mjpython():
        raise RuntimeError(
            "On macOS, launch with: PYTHONPATH=src ./venv/bin/python "
            "./venv/bin/mjpython examples/pick_place_panda.py"
        )
    env = MujocoRobotEnvironment(max_episode_steps=10_000)
    controller = PandaEndEffectorPositionController(env, position_tolerance=0.005)
    config = PickPlaceConfig()
    if vary_conditions:
        config = PickPlaceConfig(
            cube_start_bounds_m=((0.40, -0.08, 0.725), (0.50, 0.08, 0.725)),
            destination_bounds_xy_m=((0.62, -0.18), (0.68, -0.12)),
            robot_joint_perturbation_rad=0.02,
            cube_mass_scale_range=(0.9, 1.1),
            cube_friction_scale_range=(0.9, 1.1),
        )
    task = PickPlaceTask(env, controller, config)
    try:
        task.reset(seed=seed)
        mode = "seeded varied conditions" if vary_conditions else "nominal conditions"
        print(f"Pick-and-place attempt (seed={seed}, {mode}); coordinates are in world meters.")
        if vary_conditions:
            print(f"Sampled conditions: {task.condition_parameters}")
        dt = float(env.model.opt.timestep * env.n_substeps)
        previous_phase = task.phase
        with mujoco.viewer.launch_passive(
            env.model, env.data, show_left_ui=not no_ui, show_right_ui=not no_ui,
        ) as viewer:
            while viewer.is_running() and task.result is None:
                started = time.monotonic()
                step = task.step()
                viewer.sync()
                if step.phase != previous_phase:
                    print(f"Phase: {step.phase.value}")
                    previous_phase = step.phase
                if step.result is not None:
                    break
                if step.terminated or step.truncated:
                    break
                delay = dt - (time.monotonic() - started)
                if delay > 0:
                    time.sleep(delay)
        if task.result is None:
            print("Viewer closed before the task finished.")
            return False
        result = task.result
        print(f"Success: {result.success}")
        if result.failure_reason:
            print(f"Failure reason: {result.failure_reason}")
        print(f"Phases: {' -> '.join(result.phase_history)}")
        print(f"Grasp verified by bilateral contacts and lift: {result.grasp_verified}")
        print(f"Replans after detected cube motion: {result.replanning_count}")
        print(f"Initial cube position (m): {result.initial_cube_position_m}")
        print(f"Final cube position (m): {result.final_cube_position_m}")
        print(f"Final horizontal placement error: {result.final_placement_error_m:.4f} m")
        print(f"Episode length: {result.episode_length} control steps")
        return result.success
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-ui", action="store_true", help="Hide MuJoCo side panels.")
    parser.add_argument(
        "--vary-conditions", action="store_true",
        help="Seed cube/goal positions, small robot joint offsets, and cube mass/friction variation.",
    )
    args = parser.parse_args(argv)
    return 0 if run_demo(seed=args.seed, no_ui=args.no_ui, vary_conditions=args.vary_conditions) else 1


if __name__ == "__main__":
    raise SystemExit(main())
