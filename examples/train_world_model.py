#!/usr/bin/env python3
"""Train, evaluate, and save the Panda reach world model from transition JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from roboforge.experiments.transitions import TransitionDataset
from roboforge.experiments.world_model import train_world_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="structured TransitionRecord JSONL dataset")
    parser.add_argument("--model-output", type=Path, required=True, help="new .npz model artifact path")
    parser.add_argument("--report-output", type=Path, required=True, help="new JSON metrics/provenance path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1e-3)
    args = parser.parse_args(argv)

    if args.model_output.exists() or args.report_output.exists():
        raise FileExistsError("model and report outputs must be new paths; existing artifacts are preserved")
    if args.model_output.resolve() == args.report_output.resolve():
        raise ValueError("model and report outputs must use different paths")
    dataset = TransitionDataset.from_jsonl(args.dataset)
    model, report, _arrays, _split = train_world_model(
        dataset, seed=args.seed, members=args.members, ridge=args.ridge,
    )
    provenance = {
        "dataset_path": str(args.dataset.resolve()),
        "dataset_sha256": report["dataset_sha256"],
        "training_config": {"seed": args.seed, "members": args.members, "ridge": args.ridge},
        "training_report": report,
    }
    model.save(args.model_output, provenance=provenance)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for split_name in ("validation", "test"):
        metrics = report[split_name]
        print(
            f"{split_name}: model normalized RMSE={metrics['learned']['normalized_aggregate_rmse']:.6f}, "
            f"persistence normalized RMSE={metrics['persistence_baseline']['normalized_aggregate_rmse']:.6f}, "
            f"model raw MAE={metrics['learned']['aggregate_mae']:.6f}, "
            f"model raw RMSE={metrics['learned']['aggregate_rmse']:.6f}"
        )
    print(f"Model: {args.model_output.resolve()}")
    print(f"Metrics: {args.report_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
