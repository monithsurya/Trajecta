#!/usr/bin/env python3
"""Run a bounded Panda reach model-learning and experiment-selection loop."""

from __future__ import annotations

import argparse
from pathlib import Path

from roboforge.experiments.active_learning import run_active_learning
from roboforge.environments.panda_reach import PandaReachConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/active_learning_uncertainty"))
    parser.add_argument("--strategy", choices=("random", "uncertainty"), default="uncertainty")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-episodes", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--max-episode-steps", type=int, default=60)
    args = parser.parse_args(argv)
    root = args.output_dir
    result = run_active_learning(
        experience_path=root / "experiences.jsonl",
        transition_path=root / "transitions.jsonl",
        artifact_dir=root,
        strategy=args.strategy,
        task_config=PandaReachConfig(max_episode_steps=args.max_episode_steps),
        seed=args.seed,
        bootstrap_episodes=args.bootstrap_episodes,
        iterations=args.iterations,
        candidate_count=args.candidate_count,
    )
    print(f"Strategy: {result['strategy']}; bounded selection iterations: {result['requested_iterations']}")
    print(f"Run summary: {root / 'run_summary.json'}")
    print(f"Metrics and model artifacts: {root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
