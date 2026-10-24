#!/usr/bin/env python3
"""Compare bounded Panda joint-position gains in repeatable MuJoCo trials."""

from __future__ import annotations

import argparse
from pathlib import Path

from roboforge.environments.mujoco_env import MujocoRobotEnvironment
from roboforge.experiments import (
    JsonlCalibrationStore,
    JsonlTransitionStore,
    PandaControllerCalibrator,
    TrackingCalibrationSpec,
    TransitionDataset,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--candidate-kp", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--output", type=Path, default=Path("artifacts/calibrations.jsonl"))
    parser.add_argument(
        "--transitions", type=Path, default=Path("artifacts/calibration_transitions.jsonl"),
    )
    args = parser.parse_args(argv)

    spec = TrackingCalibrationSpec(
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        candidate_kp=tuple(args.candidate_kp),
    )
    env = MujocoRobotEnvironment(max_episode_steps=max(spec.max_steps + 1, 500))
    try:
        calibration = PandaControllerCalibrator(env).run(
            spec,
            transition_recorder=JsonlTransitionStore(args.transitions),
        )
        JsonlCalibrationStore(args.output).write(calibration)
        transitions = TransitionDataset.from_jsonl(args.transitions)
        manifest_path = args.transitions.with_suffix(".manifest.json")
        transitions.write_manifest(manifest_path)

        print(f"Calibration ID: {calibration.calibration_id}")
        print(f"Baseline kp (None means direct target): estimated {calibration.baseline.estimated_kp:.4f}")
        print(f"Baseline joint tracking RMSE: {calibration.baseline.mean_tracking_rmse_rad:.6f} rad")
        for candidate in calibration.candidates:
            print(
                f"Candidate kp={candidate.configured_kp:.3f}: "
                f"estimated={candidate.estimated_kp:.4f}, "
                f"RMSE={candidate.mean_tracking_rmse_rad:.6f} rad, "
                f"settled={candidate.settled_episodes}/{len(candidate.trials)}, "
                f"constraints={candidate.constraint_violation_count}"
            )
        print(f"Candidate accepted: {calibration.candidate_accepted}")
        print(f"Best candidate kp: {calibration.best_candidate_kp:.3f}")
        print(f"Accepted candidate kp: {calibration.accepted_candidate_kp}")
        print(f"Decision: {calibration.acceptance_reason}")
        print(f"Calibration records: {args.output.resolve()}")
        print(f"Transition records: {args.transitions.resolve()}")
        print(f"Transition manifest: {manifest_path.resolve()}")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
