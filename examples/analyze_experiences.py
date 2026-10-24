#!/usr/bin/env python3
"""Print deterministic diagnoses for failed episodes in a JSONL file."""

from __future__ import annotations

import argparse
from pathlib import Path

from roboforge.experiments import ExperienceDataset, analyze


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, help="Validated Trajecta experience JSONL")
    args = parser.parse_args()
    dataset = ExperienceDataset.from_jsonl(args.records).filter(success=False)
    for item in analyze(dataset.records):
        print(f"{item.episode_id}  {item.category}  confidence={item.confidence}")
        for evidence in item.evidence:
            print(f"  evidence: {evidence}")
        print(f"  investigation: {item.recommended_investigation}")
    print(f"Analyzed {len(dataset.records)} failed episode(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
