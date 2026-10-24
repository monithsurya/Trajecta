#!/usr/bin/env python3
"""Load a saved model and predict a full 37-value Panda reach next observation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from roboforge.experiments.world_model import load_world_model, reach_state_features


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("observation", type=Path, help="JSON array containing one full PandaReachEnv observation")
    parser.add_argument("action", type=float, nargs=3, help="normalized XYZ delta action")
    args = parser.parse_args(argv)
    observation = np.asarray(json.loads(args.observation.read_text(encoding="utf-8")), dtype=np.float64)
    model = load_world_model(args.model)
    predicted = model.predict_full_observation(observation, np.asarray(args.action, dtype=np.float64))
    print(json.dumps({
        "model_kind": "panda_reach_ensemble_ridge_delta",
        "predicted_next_observation": predicted.tolist(),
        "ensemble_disagreement": model.uncertainty(
            reach_state_features(observation),
            np.asarray(args.action, dtype=np.float64),
        ),
        "uncertainty_note": "dimensionless ensemble disagreement, not a calibrated confidence",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
