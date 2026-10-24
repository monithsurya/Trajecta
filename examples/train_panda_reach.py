#!/usr/bin/env python3
"""Train an optional Stable-Baselines3 PPO policy on Panda reaching."""

from __future__ import annotations

import argparse
from pathlib import Path

from roboforge.environments.panda_reach import PandaReachConfig
from roboforge.experiments.rl import PPOTrainingConfig, train_ppo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/panda_reach_ppo"))
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--resume", type=Path, default=None, help="resume from a PPO .zip into a fresh output directory")
    args = parser.parse_args(argv)

    training = PPOTrainingConfig(
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
    )
    task = PandaReachConfig(max_episode_steps=args.max_episode_steps)
    run = train_ppo(training, task, args.output_dir, resume_from=args.resume)
    print(f"PPO timesteps: {run.total_timesteps}")
    print(f"Final policy: {run.final_policy_path}")
    print(f"Best evaluated policy: {run.best_policy_path}")
    print(f"Training logs: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
