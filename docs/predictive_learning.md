# Predictive learning and experiment selection

This first predictive-learning loop is limited to `PandaReachEnv`. It learns one MuJoCo step of the existing reach environment and selects among bounded Cartesian goal positions. It does not model the pick-and-place contacts, learn a general robot dynamics function, or alter the current controller.

## Data contract and prediction target

The model consumes existing `TransitionRecord` JSONL rows. It accepts only records with `task_phase="REACHING"`, a three-value action, and consistent flat observation dimensions. It uses the current observation and action as model inputs and predicts:

```text
delta = next_observation - observation
next_observation_prediction = observation + predicted_delta
```

The full reach observation consists of MuJoCo `qpos` followed by `qvel`, then the goal-relative hand displacement and command-relative hand displacement that `PandaReachEnv` appends. The bundled Panda scene begins with a fixed-base MuJoCo free joint: its three position, four quaternion, and six velocity values are deliberately excluded. The model uses the nine named scalar arm/finger joint positions, nine corresponding joint velocities, and six goal/command-relative XYZ values (24 features total). The goal is represented in the current observation; the next observation is never part of the input. No block positions exist in this task. The seven Panda arm joint positions are angular and their deltas use the wrapped shortest difference in `[-pi, pi]`; predicted joint positions are wrapped on reconstruction. Other fields use ordinary differences.

Feature and target means/scales are fit using training transitions only. The deterministic split assigns whole episodes to train, validation, or test; adjacent transitions from one episode cannot leak across splits. At least five episodes are used for the bootstrap, leaving multiple training episodes plus fixed validation and test episodes. In active runs the bootstrap validation and test episode IDs and target scales used to normalize reported errors stay fixed while later selected episodes enter training. Model provenance stores the dataset SHA-256, source path, split IDs, normalization metadata, and metrics.

## Model and uncertainty

The baseline is persistence: predicted state delta is zero. The learned model is a NumPy ridge regression over standardized `[observation, action]` features. A small ensemble is fit using independent episode-level bootstrap samples. NumPy is already a required project dependency; model fitting is imported only when the experiment module is used and adds no deep-learning dependency.

The reported uncertainty score is the mean across state dimensions of ensemble prediction standard deviation divided by the training target scale. It is a dimensionless disagreement heuristic, not a calibrated confidence interval or guarantee. Test reports also include the Pearson correlation between this disagreement and per-transition prediction RMSE when the test sample permits a correlation. Treat a weak, undefined, or negative correlation as evidence against using the score as an error proxy.

Validation and test metrics include raw per-feature RMSE, raw aggregate RMSE/MAE, and normalized aggregate RMSE/MAE. The normalized aggregate divides each feature error by that feature's training target scale before aggregation, so radian, velocity, and meter values are not mixed without scaling. The persistence baseline uses the same scales. The fixed test set makes within-run changes comparable as data accumulates. A lower training error alone is not considered evidence of useful prediction.

## Candidate generation and selection

The current candidate dimension is the reach target XYZ position. Candidates are sampled with an explicit seed inside the configured goal bounds, which are validated as contained by the task workspace. Rounded target IDs and a minimum target separation avoid duplicate or near-duplicate selections within a run. The IK heuristic remains the execution controller; selection does not change controller gains or hardware behavior.

The random strategy samples the generated valid candidate pool with a seeded generator. Before uncertainty scoring, the model's validation normalized RMSE must beat the persistence baseline; otherwise the loop records a fallback and uses seeded random selection. Test metrics are never used in this decision. If the validation gate passes, the uncertainty strategy resets an environment at each candidate target, obtains the current observation and heuristic action, and ranks the prior model's ensemble disagreement for that one-step prediction. Each decision record stores the candidate pool/scoring seeds and pre-execution scores with the chosen target; its outcome fields are added after the run. The true outcome is only consumed after execution.

## Bootstrap, run, and compare

With no reach transitions, the loop first executes five seeded, bounded heuristic reach episodes and records them through the existing `ExperimentRunner`, `ExperienceRecord`, and `TransitionRecord` interfaces. It then fits the bootstrap model. Each requested iteration generates a bounded pool, selects one target, runs one episode, appends the structured episode/transition records and selection outcome, and refits the model. There is no automatic unbounded run.

```bash
python examples/run_active_learning.py \
  --output-dir artifacts/active_learning_uncertainty \
  --strategy uncertainty --seed 0 --bootstrap-episodes 5 --iterations 3
```

For an existing transition store and compatible previously trained model, pass their paths along with `--model`; the output directory must be fresh:

```bash
python examples/run_active_learning.py \
  --output-dir artifacts/active_learning_continued \
  --transition-path artifacts/previous_run/transitions.jsonl \
  --experience-path artifacts/previous_run/experiences.jsonl \
  --model artifacts/previous_run/world_model_000.npz \
  --seed 0 --iterations 3
```

The model's trained episode IDs must be present in the supplied transition dataset. Without a model, the workflow bootstraps and fits one from the data before selecting.

To train and evaluate a model explicitly from an existing transition dataset, then reload it for a full-observation prediction:

```bash
python examples/train_world_model.py artifacts/active_learning_uncertainty/transitions.jsonl \
  --model-output artifacts/reach_model.npz --report-output artifacts/reach_model_metrics.json --seed 0
python examples/predict_world_model.py artifacts/reach_model.npz artifacts/one_observation.json 0.0 0.5 -0.2
```

The training command requires at least five distinct reach episodes and refuses to overwrite model or report paths. Its report contains train/validation/test transition and episode counts, per-feature and aggregate errors, the persistence comparison, split IDs, metric scales, and transition-file SHA-256. Prediction expects one full 37-value `PandaReachEnv` observation and returns a full next observation; fixed-base fields excluded from training are preserved from the input. `load_world_model()` and `predict_full_observation()` provide the same operation as Python APIs.

Compare random and uncertainty selection under paired conditions:

```bash
python examples/compare_selection_strategies.py \
  --output-dir artifacts/selection_comparison \
  --seeds 0 1 2 --bootstrap-episodes 15 --iterations 3
```

For each seed, comparison first runs one bootstrap and copies the exact same structured data into both strategy arms. They use the same task configuration, seeds, candidate pool generation, and fixed bootstrap validation/test episodes. `comparison.json` reports held-out RMSE change and reduction per selected experiment/per measured selection, execution, and model-training second; it also records persistence error, experiment/transition counts, failures, and those timing components. Outcomes may favor either strategy; this is a small simulation study and should be run with multiple seeds and larger budgets before drawing conclusions.

The output directory must be new or empty; checkpoints, metrics, datasets, and decision logs are not written into Git-tracked source. Existing experience and transition stores are append-only and are validated when loaded. Back them up before intentionally reusing paths. Model files use versioned NPZ artifacts with JSON metadata and reject overwrites.

## Reference comparison

A local three-seed comparison used 15 bootstrap episodes, three selected episodes per arm, 16 candidates per selection, 60-step episode caps, and five ridge ensemble members. Each seed's two arms began from byte-identical bootstrap stores, used the same fixed held-out episodes, and shared its bootstrap-trained feature scales for normalized metrics. Uncertainty selection was enabled only when the model beat persistence on validation; otherwise it would fall back to random. The table reports normalized aggregate test RMSE (lower is better):

| Seed | Initial model | Persistence after data | Random after data | Uncertainty after data |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.672 | 1.373 | 0.735 | 0.478 |
| 1 | 0.913 | 1.402 | 0.755 | 0.461 |
| 2 | 0.639 | 1.426 | 0.347 | 0.401 |
| Mean final | — | 1.401 | 0.612 | 0.447 |

Both adaptive arms beat persistence on all three seeds. Uncertainty selection had lower final test error on two seeds and a lower three-seed mean; random was better on seed 2. The final datasets contained 820 transitions in the uncertainty arms versus 822 in random arms, with no failed episodes. Candidate scoring took about 0.085 seconds across three uncertainty rounds (about 0.028 seconds per round) versus 0.042 seconds total for random selection. Episode execution took about 0.05 seconds per arm/seed; four model fits took around 0.05 seconds per arm. Disagreement/error correlations on the uncertainty held-out episodes were 0.619, 0.056, and 0.261. So the mean-error result favors uncertainty in this small run, but per-seed outcome varies and disagreement-error correlation was weak for two seeds. Three trials and a small fixed test set do not establish a reliable learning-efficiency benefit or calibrated uncertainty.

An explicit training smoke run on the five-episode bootstrap dataset from the earlier run reported test normalized RMSE 13.30 for the learned model versus 1.74 for persistence. This actual recorded-data result is substantially worse than baseline; the resulting model can be loaded and queried, but that score does not support using it for uncertainty-directed selection. The validation gate keeps the active loop on random selection when this happens.

## Limits

The ensemble is a compact baseline, not a validated physical dynamics model. Disagreement can miss shared model bias and its correlation with error can be unstable on a small test set. Current selection only varies reach targets; it does not optimize task success, information gain, block placement, robot initial state, or controller parameters. The deterministic IK baseline makes this a useful low-cost data loop, but conclusions do not transfer to pick-and-place, other robots, altered physics, or real hardware. A comparison with a few seeds can be inconclusive and must be reported as measured, including negative outcomes.
