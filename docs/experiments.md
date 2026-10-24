# Experiments and recorded data

## Define and run an experiment

`ExperimentSpec` validates the task, positive episode count, optional 32-bit seed, and `ControllerConfig`. `ExperimentRunner` receives an `EpisodeTask`, environment identifier, optional task metadata, recorder, and evaluator. Episode seed is `(base_seed + episode_index) mod 2^32`.

The runnable end-to-end configuration is in `examples/evaluate_pick_place.py`:

```bash
python examples/evaluate_pick_place.py --episodes 5 --seed 0 --output artifacts/experiences.jsonl
```

The script builds the bundled Panda environment, end-effector controller, `PickPlaceTask`, validated spec, and JSONL recorder. It prints attempt/success counts, episode lengths, placement error, and failures. Exit status is nonzero if any run fails.

## Task success and metrics

Success is verified by the task state machine: bilateral cube/finger contact is maintained through lift, the released cube is table-supported, lies within the horizontal placement tolerance, rests at table height, and is no longer in finger contact. A reward value alone is never treated as proof of task success.

The evaluator reports attempted and successful episodes, success fraction, mean/median control steps when available, mean placement error for placement attempts, and failure counts. Episode time is wall time in records; control-step count is not simulation seconds.

## Experience record and files

`ExperienceRecord` is a version-1 Pydantic model with episode and experiment IDs, UTC timestamp, task/environment IDs, configuration metadata, seed, explicit outcome, episode length/time, initial/final object positions, placement error, phase history, failure reason, and controller configuration when present. `JsonlExperienceStore` validates records before append and validates each line on read. A malformed record reports its path and line number. It stores no invented or reconstructed transition data.

Dataset helpers filter by task/outcome/experiment, produce a content-hashed manifest, export validated JSONL, and split at the `experiment_id` level to prevent episodes from one run crossing train/test. If there is only one run, one split may be empty; collect multiple distinct runs. The current episode-only schema is not training data for offline RL or learned dynamics.

Failure diagnoses use recorded failure text and phases to provide a category, evidence, a qualified likely cause, and an investigation suggestion. Treat the output as triage, not a proven causal explanation.

## Reproducibility limits

The runner deterministically schedules seeds, but the current Panda scene reset uses a fixed cube and goal location. Different episode seeds do not currently produce randomized task conditions. The five-record summary at `media/evaluation_summary.json` is a repeated smoke check under the same setup, not an independent or held-out benchmark. Calibration, RL, world-model, and active-selection workflows do not exist yet.
