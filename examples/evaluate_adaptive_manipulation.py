#!/usr/bin/env python3
"""Paired nominal/robust Panda pick-place evaluation over seeded conditions."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

from roboforge.controllers.end_effector_position import PandaEndEffectorPositionController
from roboforge.environments.mujoco_env import MujocoRobotEnvironment
from roboforge.experiments.experience import JsonlExperienceStore
from roboforge.experiments.runner import ExperimentRunner
from roboforge.experiments.spec import ControllerConfig, ControllerType, ExperimentSpec, TaskType
from roboforge.experiments.transitions import JsonlTransitionStore
from roboforge.tasks.pick_place import PickPlaceConfig, PickPlaceTask


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/adaptive_manipulation_report.json"))
    parser.add_argument("--transitions-dir", type=Path, default=Path("artifacts/adaptive_manipulation_transitions"))
    parser.add_argument("--max-recovery-attempts", type=int, default=2)
    parser.add_argument("--vary-conditions", action="store_true")
    args = parser.parse_args(argv)
    if isinstance(args.episodes, bool) or not 1 <= args.episodes <= 100:
        parser.error("--episodes must be in [1, 100]")
    if not 0 <= args.seed <= 4_294_967_295:
        parser.error("--seed must fit an unsigned 32-bit integer")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation report: {args.output}")
    args.transitions_dir.mkdir(parents=True, exist_ok=True)
    if any(args.transitions_dir.iterdir()):
        raise FileExistsError(f"transition output directory must be empty: {args.transitions_dir}")

    variation = {}
    if args.vary_conditions:
        variation = {
            "cube_start_bounds_m": ((0.40, -0.08, 0.725), (0.50, 0.08, 0.725)),
            "destination_bounds_xy_m": ((0.62, -0.18), (0.68, -0.12)),
            "robot_joint_perturbation_rad": 0.02,
            "cube_mass_scale_range": (0.9, 1.1),
            "cube_friction_scale_range": (0.9, 1.1),
        }
    common_config = PickPlaceConfig(**variation)
    arms: dict[str, list[dict[str, object]]] = {"baseline": [], "adaptive": []}
    for episode in range(args.episodes):
        episode_seed = (args.seed + episode) % (4_294_967_295 + 1)
        for arm in ("baseline", "adaptive"):
            recovery_attempts = args.max_recovery_attempts if arm == "adaptive" else 0
            config = replace(common_config, max_recovery_attempts=recovery_attempts)
            env = MujocoRobotEnvironment(n_substeps=10, max_episode_steps=10_000)
            task = PickPlaceTask(env, PandaEndEffectorPositionController(env), config)
            spec = ExperimentSpec(
                task=TaskType.PICK_AND_PLACE,
                episodes=1,
                seed=episode_seed,
                controller=ControllerConfig(controller_type=ControllerType.OPERATIONAL_SPACE),
            )
            experience_path = args.transitions_dir / f"{arm}_experiences.jsonl"
            transition_path = args.transitions_dir / f"{arm}_transitions.jsonl"
            store = JsonlExperienceStore(experience_path)
            transitions = JsonlTransitionStore(transition_path)
            started = time.perf_counter()
            try:
                run = ExperimentRunner(
                    task,
                    spec,
                    environment_id="mujoco_panda_scene",
                    task_config=asdict(config),
                    recorder=store,
                    transition_recorder=transitions,
                ).run()
                result = run.records[0]
                arms[arm].append({
                    "seed": episode_seed,
                    "success": result.success,
                    "grasp_verified": result.grasp_verified,
                    "placement_error_m": result.placement_error_m,
                    "episode_length": result.episode_length,
                    "recovery_attempts": result.recovery_attempts,
                    "replanning_count": result.replanning_count,
                    "failure_category": result.failure_category,
                    "failure_reason": result.failure_reason,
                    "condition_parameters": result.condition_parameters,
                    "wall_time_s": time.perf_counter() - started,
                    "experiment_id": str(result.experiment_id),
                    "episode_id": str(result.episode_id),
                })
            finally:
                env.close()

    def aggregate(episodes: list[dict[str, object]]) -> dict[str, object]:
        success_count = sum(bool(item["success"]) for item in episodes)
        grasp_count = sum(bool(item["grasp_verified"]) for item in episodes)
        placement_errors = [float(item["placement_error_m"]) for item in episodes if item["placement_error_m"] is not None]
        return {
            "attempted_episodes": len(episodes),
            "success_rate": success_count / len(episodes),
            "grasp_success_rate": grasp_count / len(episodes),
            "mean_placement_error_m": sum(placement_errors) / len(placement_errors) if placement_errors else None,
            "mean_episode_length": sum(int(item["episode_length"]) for item in episodes) / len(episodes),
            "mean_recovery_attempts": sum(int(item["recovery_attempts"]) for item in episodes) / len(episodes),
            "failure_categories": {
                str(category): sum(item["failure_category"] == category for item in episodes)
                for category in {item["failure_category"] for item in episodes if item["failure_category"] is not None}
            },
            "mean_wall_time_s": sum(float(item["wall_time_s"]) for item in episodes) / len(episodes),
        }

    report = {
        "schema_version": 1,
        "protocol": "paired seeds and same initial condition/variation ranges; baseline has no retries; adaptive arm retries classified failures with bounded grasp-height offsets",
        "simulation_only": True,
        "conditions_varied": args.vary_conditions,
        "configuration": asdict(common_config),
        "baseline": {"metrics": aggregate(arms["baseline"]), "episodes": arms["baseline"]},
        "adaptive": {"metrics": aggregate(arms["adaptive"]), "episodes": arms["adaptive"]},
        "limitations": [
            "small simulation sample; no generalization claim",
            "recovery resets the episode and perturbs only grasp height within configured bounds",
            "no collision-safety guarantee or physical robot validation",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for arm in ("baseline", "adaptive"):
        print(f"{arm}: {report[arm]['metrics']}")
    print(f"Report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
