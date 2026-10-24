# Experiments and recorded data

## Define and run an experiment

`ExperimentSpec` validates the task, positive episode count, optional 32-bit seed, and `ControllerConfig`. `ExperimentRunner` receives an `EpisodeTask`, environment identifier, optional task metadata, recorder, and evaluator. Episode seed is `(base_seed + episode_index) mod 2^32`.

The runnable end-to-end configuration is in `examples/evaluate_pick_place.py`:

```bash
python examples/evaluate_pick_place.py --episodes 5 --seed 0 --output artifacts/experiences.jsonl
```

The script builds the bundled Panda environment, end-effector controller, `PickPlaceTask`, validated spec, and JSONL recorder. It prints attempt/success counts, episode lengths, placement error, and failures. Exit status is nonzero if any run fails. Add `--transitions artifacts/transitions.jsonl` to record one transition per completed environment step alongside the episode summaries. The optional transition path is append-only; use a new path for each run if you want to keep datasets separate. Existing files are never truncated.

## Task success and metrics

Success is verified by the task state machine: bilateral cube/finger contact is maintained through lift, the released cube is table-supported, lies within the horizontal placement tolerance, rests at table height, and is no longer in finger contact. A reward value alone is never treated as proof of task success.

The evaluator reports attempted and successful episodes, success fraction, mean/median control steps when available, mean placement error for placement attempts, and failure counts. Episode time is wall time in records; control-step count is not simulation seconds.

## Experience record and files

`ExperienceRecord` is a version-1 Pydantic model with episode and experiment IDs, UTC timestamp, task/environment IDs, configuration metadata, seed, explicit outcome, episode length/time, initial/final object positions, placement error, phase history, failure reason, and controller configuration when present. `JsonlExperienceStore` validates records before append and validates each line on read. A malformed record reports its path and line number.

## Step-level transitions

`TransitionRecord` version 1 stores experiment/episode IDs, a zero-based transition index, seed, current observation, normalized action passed to `env.step`, next observation, environment reward when present, separate `terminated` and `truncated` flags, simulator time/timestep and environment step when exposed, and task phase/success/failure signals. Vectors are serialized as finite JSON number arrays. Experiment configuration is referenced through the experiment ID and remains in the episode summary rather than being repeated for every step. Missing reward stays `null`; the current MuJoCo adapter actually returns `0.0`, which is recorded as such. Task success is a separate field and is never substituted for reward.

The transition JSONL sits at the path supplied with `--transitions`, separate from the episode summary JSONL. The command writes a sibling manifest with `.manifest.json` suffix, including record counts by episode, schema versions, experiment IDs, and a content hash. `TransitionDataset.from_jsonl(path)` loads and validates the file, including contiguous per-episode transition indices. Malformed or partial JSONL lines raise with their path and line number; recording errors propagate and fail the run rather than being ignored. Each record is flushed after append. Reusing a path intentionally appends records; use a new run-specific path when you need isolation.

An illustrative record (vector values shortened):

```json
{"schema_version":1,"experiment_id":"00000000-0000-0000-0000-000000000001","episode_id":"00000000-0000-0000-0000-000000000002","transition_index":0,"seed":0,"observation":[0.1,0.2],"action":[-0.4,1.0],"next_observation":[0.11,0.19],"reward":0.0,"terminated":false,"truncated":false,"simulation_time_s":0.02,"simulation_timestep_s":0.02,"environment_step":1,"task_phase":"APPROACH","task_success":null,"task_failure_reason":null}
```

Load the recorded transitions and inspect the manifest in Python:

```python
from roboforge.experiments import TransitionDataset

dataset = TransitionDataset.from_jsonl("artifacts/transitions.jsonl")
print(dataset.manifest()["record_count"])
```

Transition capture records observed data. It does not implement reward shaping, reinforcement learning, learned dynamics, or adaptive experiment selection.

Episode dataset helpers filter by task/outcome/experiment, produce a content-hashed manifest, export validated JSONL, and split at the `experiment_id` level to prevent episodes from one run crossing train/test. If there is only one run, one split may be empty; collect multiple distinct runs. Transition capture is not itself an offline RL or learned dynamics implementation.

Failure diagnoses use recorded failure text and phases to provide a category, evidence, a qualified likely cause, and an investigation suggestion. Treat the output as triage, not a proven causal explanation.

## Reproducibility limits

The runner deterministically schedules seeds, but the current Panda scene reset uses a fixed cube and goal location. Different episode seeds do not currently produce randomized task conditions. The five-record summary at `media/evaluation_summary.json` is a repeated smoke check under the same setup, not an independent or held-out benchmark. Calibration, RL, world-model, and active-selection workflows do not exist yet.
