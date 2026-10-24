# Panda controller calibration

## Scope

The first calibration component evaluates the existing Panda joint-position controller's setpoint blend gain `kp`. The controller law is

```text
q_cmd = q_current + kp * (q_target - q_current)
```

The current default (`kp=None`) sends the target directly, equivalent to a blend gain of 1. Candidate gains are bounded to `(0, 1]`. The experiment resets the bundled MuJoCo Panda scene to its named `home` keyframe with the same seed for the baseline and each candidate, then commands a bounded seven-joint target offset. Offsets are limited to 0.10 rad per joint.

This is controller command calibration. It does **not** identify mass, friction, actuator bandwidth, or other physical dynamics. The current scene exposes position-servo command ranges and joint observations, but this initial experiment is not designed to distinguish those dynamics parameters.

## Estimation and comparison

For every recorded step, the estimator maps each normalized arm action back through the known Panda joint range to reconstruct the commanded joint setpoint. It estimates the blend gain as the origin-constrained least-squares slope:

```text
kp_hat = sum((q_target - q) * (q_cmd - q)) / sum((q_target - q)^2)
```

Samples with negligible target displacement do not contribute. The estimator rejects non-finite or insufficient data and estimates outside the expected `[−0.05, 1.05]` numerical range. It checks the controller's command mapping; it is not a physical dynamics estimate.

The objective is mean joint tracking RMSE in radians, pooled over measured post-step joint states and configured episodes. Each gain also reports mean final-window RMSE, number of episodes that settled, settling time, constraint violations, truncation/termination, and the fitted command gain. Settling time is the first observed sample after which every joint remains within the configured tolerance for the rest of that rollout. A non-settling, terminated, or truncated rollout counts as a failure.

The lowest-RMSE candidate is accepted only if all of these checks pass:

1. At least two episodes are available, with candidate and baseline seed sequences matched.
2. Its RMSE improvement over baseline is at least `minimum_rmse_improvement_rad` (default `0.0005` rad).
3. It settles at least as many episodes as baseline.
4. It has zero constraint violations and zero failed episodes.

If any check fails, the result records a rejection reason. `best_candidate_kp` names the lowest-RMSE candidate even when it is rejected; `accepted_candidate_kp` is populated only after all checks pass. The controller default is never changed automatically, and a one-run result should be treated as a candidate evaluation rather than a general guarantee.

## Run and load results

From the repository root:

```bash
python examples/calibrate_panda_controller.py \
  --episodes 3 --seed 0 --max-steps 250 \
  --candidate-kp 0.25 0.5 0.75 1.0
```

By default this appends validated calibration results to `artifacts/calibrations.jsonl`, step-level observations/actions to `artifacts/calibration_transitions.jsonl`, and writes `artifacts/calibration_transitions.manifest.json`. All are ignored by Git. Existing JSONL output is appended, never truncated; use new output paths when you want a separate dataset. Each result records the calibration spec, environment timestep and substeps, baseline/candidate estimates and metrics, acceptance decision, transition count, and a hash of that run's transition rows. Its experiment and episode IDs identify the corresponding steps in the transition file.

An API caller can use `TrackingCalibrationSpec`, `PandaControllerCalibrator`, `JsonlTransitionStore`, and `JsonlCalibrationStore` from `roboforge.experiments`. Calibration data can be loaded and checked with:

```python
from roboforge.environments.mujoco_env import MujocoRobotEnvironment
from roboforge.experiments import (
    JsonlCalibrationStore,
    JsonlTransitionStore,
    PandaControllerCalibrator,
    TrackingCalibrationSpec,
)

env = MujocoRobotEnvironment(max_episode_steps=501)
try:
    spec = TrackingCalibrationSpec(episodes=3, seed=0, max_steps=250)
    result = PandaControllerCalibrator(env).run(
        spec,
        transition_recorder=JsonlTransitionStore("artifacts/calibration_transitions.jsonl"),
    )
    JsonlCalibrationStore("artifacts/calibrations.jsonl").write(result)
finally:
    env.close()
```

Read previously saved records:

```python
from roboforge.experiments import JsonlCalibrationStore, TransitionDataset

calibrations = JsonlCalibrationStore("artifacts/calibrations.jsonl").read_all()
transitions = TransitionDataset.from_jsonl("artifacts/calibration_transitions.jsonl")
print(calibrations[-1].candidate_accepted)
print(transitions.manifest()["record_count"])
```

## Limits and next steps

The current calibration objective measures only joint-position tracking for a small, fixed target perturbation around the bundled Panda home pose. It does not evaluate pick-and-place success, workspace variation, randomized object conditions, or physical hardware. The environment's reward remains zero and is recorded as provided; task-success reward shaping is not introduced. Any dynamics system-identification work should first define controlled inputs and observable outputs that can distinguish a specific physical parameter, then compare that estimator with synthetic and held-out simulation evidence.
