#!/usr/bin/env python3
"""Compare random and ensemble-uncertainty target selection from shared bootstrap data."""

from __future__ import annotations

import argparse
from pathlib import Path

from roboforge.experiments.active_learning import compare_selection_strategies
from roboforge.environments.panda_reach import PandaReachConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/selection_comparison"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--bootstrap-episodes", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--max-episode-steps", type=int, default=60)
    args = parser.parse_args(argv)
    report = compare_selection_strategies(
        output_dir=args.output_dir,
        seeds=tuple(args.seeds),
        bootstrap_episodes=args.bootstrap_episodes,
        iterations=args.iterations,
        candidate_count=args.candidate_count,
        task_config=PandaReachConfig(max_episode_steps=args.max_episode_steps),
    )
    for comparison in report["comparisons"]:
        print(f"seed {comparison['seed']}: initial normalized held-out RMSE={comparison['initial_test_rmse']:.6f}")
        for strategy, metrics in comparison["arms"].items():
            print(
                f"  {strategy}: final normalized RMSE={metrics['final_test_rmse']:.6f}, "
                f"change={metrics['test_rmse_change']:+.6f}, "
                f"episodes={metrics['experiment_count']}, transitions={metrics['transition_count']}, "
                f"failed={metrics['failed_experiments']}, experiment_seconds={metrics['experiment_seconds']:.3f}, "
                f"selection_seconds={metrics['selection_seconds']:.3f}, "
                f"training_seconds={metrics['training_seconds']:.3f}"
            )
    print(f"Comparison report: {args.output_dir / 'comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
