# Adaptive manipulation and sim-to-real readiness

This phase adds bounded variation and recovery to the existing Panda pick-and-place state machine. It remains a MuJoCo-only implementation; no physical robot adapter is installed or enabled.

## Adaptive pick-and-place behavior

`PickPlaceConfig` can specify a fixed `cube_start_position_m` (or the legacy alias `cube_target_position_m`), seeded `cube_start_bounds_m`, a fixed `destination_xy_m`, or seeded `destination_bounds_xy_m`. The cube reset position must keep it fully inside the table footprint and on the table surface; destinations are checked against the same table geometry. Robot initial joints can be specified by seven named Panda arm values or sampled around the packaged home key with `robot_joint_perturbation_rad` (limited to 0.15 rad). MuJoCo validates joint limits before reset.

The task reads the actual cube and hand state and solves IK targets at each task phase. Approach, align, and descend targets are derived from the current cube position. If the cube moves by more than `object_replan_threshold_m` during one of these phases, the task recalculates the target. `max_replans_per_phase` bounds this behavior; exhaustion returns an explicit task failure.

Grasp and placement success still require simulated evidence: bilateral finger contact, cube rise while grasped, table support after release, low placement error, and no lingering finger contact. Failure categories (`missed_grasp`, `failed_reach`, `placement_failure`, `task_failure`) are coarse labels based on explicit task phase and error text; they are not causal diagnoses.

`max_recovery_attempts` is limited to 0–3 and defaults to zero, preserving previous behavior. When enabled, only classified reach, grasp, or placement failures are retried. The task resets the episode with the next deterministic seed and tries a configured grasp-height offset (default offsets: 0, -5 mm, +5 mm). It records every attempt's sampled conditions and failure label. Each retry consumes the same bounded `max_steps` episode budget; there is no automatic loop beyond the configured attempt count.

## Simulation variation and evaluation

The current MuJoCo adapter supports optional reset variation for the Panda `target_cube`: cube position, mass scale, and friction scale. Mass and body inertia are scaled together; friction is scaled for the cube geom. Allowed mass/friction scales are [0.5, 1.5]; nominal reset restores original values before applying a seeded episode configuration. Robot joint offsets and task positions are separately bounded and recorded in `ExperienceRecord.condition_parameters`.

Run a paired evaluation using identical seeds and configured conditions for baseline (no retry) and adaptive (bounded retry) arms:

```bash
python examples/evaluate_adaptive_manipulation.py \
  --episodes 10 --seed 100 --vary-conditions --max-recovery-attempts 2 \
  --output artifacts/adaptive_report.json \
  --transitions-dir artifacts/adaptive_transitions
```

Without `--vary-conditions`, this tests the nominal scene. Variation mode samples cube starts in x=[0.40, 0.50] m, y=[-0.08, 0.08] m, destination in x=[0.62, 0.68] m, y=[-0.18, -0.12] m, joint offsets up to 0.02 rad, and cube mass/friction scales [0.9, 1.1]. These are bounded simulator scenarios, not calibrated real-world parameter distributions. The report summarizes success, grasp verification, placement error, steps, retry counts, failure classes, timing, and per-episode sampled values. It writes episode and transition records via the existing stores. Output paths must be new/empty.

## Backend and safety boundary

`environments/robot_backend.py` defines `RobotBackend`, `BackendCapabilities`, and command-result contracts. `MujocoPandaBackend` implements state reads, joint-position commands, gripper commands, and simulator reset. Commands reject unknown joints, non-finite values, joint-limit violations, invalid gripper fractions, and invalid step/time budgets; non-settling joint commands return a timeout result. The adapter advertises simulation-only capability and limitations.

`create_robot_backend("hardware")` intentionally raises `NotImplementedError`. No code auto-connects to a robot. The task state machine still consumes the MuJoCo environment/controller APIs directly; this contract is an integration boundary, not a completed hardware refactor or validated driver. There is no physical emergency stop, watchdog, force/torque sensor, camera localization, communications timeout behavior, or real collision guarantee here. MuJoCo contacts validate the simulated task only.

## Measured smoke evaluation

The current host ran a five-seed nominal comparison (seeds 0–4) and a 10-seed varied-condition comparison (seeds 100–109; cube/target/joint/mass/friction bounds above; two recovery retries allowed). Both baseline and adaptive arms achieved 5/5 nominal and 10/10 varied successes, verified every grasp, and had zero recovery attempts. Mean placement error was 0.00760 m nominal and 0.00764 m varied; mean episode length was 570 nominal and 570.8 varied steps. Because no episode needed a retry, these runs demonstrate baseline performance under those sampled conditions but provide no measured recovery benefit. The sample is a simulation smoke test, not statistical robustness evidence or sim-to-real validation. Randomization is evaluation-only in this suite; no policy training under randomized conditions was conducted.

## Readiness status

| Area | Status |
| --- | --- |
| Seeded reset variation and task-state feedback | Implemented and MuJoCo-tested within the documented bounds |
| Replanning after detected cube movement | Implemented with a finite per-phase budget; unit-tested |
| Grasp/placement verification | Implemented using MuJoCo contact and object state |
| Bounded recovery | Implemented for classified failures; retry control tested with deterministic mocks; no recovery was triggered in the 10-seed smoke run |
| MuJoCo backend command boundary | Implemented and tested against simulation |
| Physical Panda driver and operator-approved execution | Not implemented; hardware path refuses to instantiate |
| Sim-to-real transfer | Not tested and not claimed |

Before hardware work, Trajecta needs a concrete robot/cell specification, calibrated camera and object localization, robot/tool/gripper calibration, joint/Cartesian and force limits, collision scene validation, controller timing/latency measurements, communications fault handling, a physical emergency stop and independent safety review, operator-confirmed commissioning, and a staged real-robot evaluation protocol. Domain randomization alone does not establish transfer.
