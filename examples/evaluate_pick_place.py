#!/usr/bin/env python3
"""Evaluate headless Panda episodes and append their records to JSONL."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from roboforge.controllers.end_effector_position import PandaEndEffectorPositionController
from roboforge.environments.mujoco_env import MujocoRobotEnvironment
from roboforge.experiments import (
    ControllerConfig,
    ControllerType,
    ExperimentRunner,
    ExperimentSpec,
    JsonlExperienceStore,
    JsonlTransitionStore,
    TransitionDataset,
    TaskType,
)
from roboforge.tasks.pick_place import PickPlaceConfig, PickPlaceTask


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/experiences.jsonl"))
    parser.add_argument(
        "--transitions", type=Path, default=None,
        help="optional step-level JSONL output (for example artifacts/transitions.jsonl)",
    )
    args = parser.parse_args(argv)
    spec = ExperimentSpec(
        task=TaskType.PICK_AND_PLACE,
        episodes=args.episodes,
        seed=args.seed,
        controller=ControllerConfig(
            controller_type=ControllerType.JOINT_POSITION,
            frequency_hz=50.0,
            extra_params={
                "interface": "PandaEndEffectorPositionController",
                "ik_position_tolerance_m": 0.005,
            },
        ),
    )
    env = MujocoRobotEnvironment(n_substeps=10, max_episode_steps=10_000)
    try:
        task_config = PickPlaceConfig()
        task = PickPlaceTask(env, PandaEndEffectorPositionController(env), task_config)
        store = JsonlExperienceStore(args.output)
        transition_store = JsonlTransitionStore(args.transitions) if args.transitions else None
        run = ExperimentRunner(
            task,
            spec,
            environment_id="mujoco_panda_scene",
            task_config=asdict(task_config),
            recorder=store,
            transition_recorder=transition_store,
        ).run()
        summary = run.summary
        print(f"Attempted episodes: {summary.attempted_episodes}")
        print(f"Successful episodes: {summary.successful_episodes}")
        print(f"Success rate: {summary.success_rate}")
        print(f"Mean/median episode length: {summary.mean_episode_length} / {summary.median_episode_length}")
        print(f"Mean placement error (m): {summary.mean_placement_error_m}")
        print(f"Failures: {summary.failure_counts}")
        print(f"JSONL records: {store.path.resolve()}")
        if transition_store is not None:
            dataset = TransitionDataset.from_jsonl(transition_store.path)
            manifest_path = transition_store.path.with_suffix(".manifest.json")
            dataset.write_manifest(manifest_path)
            print(f"Transition records: {len(dataset.records)}")
            print(f"Transition manifest: {manifest_path.resolve()}")
        return 0 if summary.successful_episodes == summary.attempted_episodes else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
