#!/usr/bin/env python3
"""Compare a saved PPO reach policy with Trajecta's IK-guided baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from roboforge.environments.panda_reach import PandaReachConfig
from roboforge.experiments.experience import JsonlExperienceStore
from roboforge.experiments.rl import JsonlReachEvaluationStore, evaluate_reach_policies, load_ppo_model
from roboforge.experiments.transitions import JsonlTransitionStore, TransitionDataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="trained policy .zip file")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1_000_000)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("artifacts/panda_reach_evaluations.jsonl"))
    parser.add_argument("--experiences", type=Path, default=Path("artifacts/panda_reach_experiences.jsonl"))
    parser.add_argument("--transitions", type=Path, default=Path("artifacts/panda_reach_eval_transitions.jsonl"))
    args = parser.parse_args(argv)

    model = load_ppo_model(args.model)
    task_config = PandaReachConfig(max_episode_steps=args.max_episode_steps)
    result = evaluate_reach_policies(
        model,
        task_config,
        episodes=args.episodes,
        seed=args.seed,
        model_path=str(args.model),
        experience_store=JsonlExperienceStore(args.experiences),
        transition_recorder=JsonlTransitionStore(args.transitions),
    )
    JsonlReachEvaluationStore(args.output).write(result)
    transition_dataset = TransitionDataset.from_jsonl(args.transitions)
    manifest_path = args.transitions.with_suffix(".manifest.json")
    transition_dataset.write_manifest(manifest_path)
    for evaluation in (result.baseline, result.trained_policy):
        print(
            f"{evaluation.policy_name}: success={evaluation.success_rate:.3f}, "
            f"return={evaluation.mean_episode_return:.3f}, "
            f"final_error={evaluation.mean_final_target_error_m:.4f} m, "
            f"length={evaluation.mean_episode_length:.1f}, "
            f"failures={evaluation.failure_counts}, "
            f"constraint_violations={evaluation.constraint_violations}"
        )
    print(f"Evaluation records: {args.output.resolve()}")
    print(f"Episode experiences: {args.experiences.resolve()}")
    print(f"Evaluation transitions: {args.transitions.resolve()}")
    print(f"Transition manifest: {manifest_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
