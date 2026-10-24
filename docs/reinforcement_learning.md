# Panda reach PPO

Trajecta includes an optional first reinforcement-learning pipeline for a narrowly scoped task: move the Panda end effector to a sampled Cartesian target in the bundled MuJoCo scene. It is separate from the existing pick-and-place task and experiment runner. It does not establish generalization to other robots, tasks, physics, or hardware.

## Install and run

PPO uses Stable-Baselines3 and PyTorch, so training dependencies are optional and do not burden the default simulation install:

```bash
python -m pip install -e '.[dev,rl]'
python examples/train_panda_reach.py --total-timesteps 50000 --seed 0
python examples/evaluate_panda_reach.py artifacts/panda_reach_ppo/checkpoints/best_model.zip --episodes 20 --seed 1000000
```

The training output directory must be empty. Training writes the validated configuration and result, Monitor CSV, periodic evaluation data, `policy.zip`, and the best evaluation checkpoint at `checkpoints/best_model.zip`. To continue from a checkpoint, supply `--resume` and choose a new, empty output directory. Training starts with a fixed seed and CPU device by default; `PPOTrainingConfig` validates rollout/batch sizes and evaluation frequency.

Evaluation runs the IK-guided heuristic and PPO on the same held-out seed sequence, each in a fresh reset episode. It reports success rate, mean return, final target error, episode length, failure categories, and constraint violations. It appends a versioned comparison JSONL record and uses the existing episode experience and transition recorders when output paths are provided. By default the CLI saves episode experiences and per-step transitions under `artifacts/` and writes a transition manifest.

## Environment and reward

`PandaReachEnv` follows the Gymnasium `reset` and five-value `step` API. The observation combines the existing Panda joint observation with goal-relative displacement and command-to-hand displacement. Its three bounded action values command a Cartesian delta through the existing Panda end-effector position controller. Targets are sampled from the configured reachable workspace; an explicit target may be supplied on reset for tests and controlled evaluation.

The reward function in `roboforge.environments.panda_reach.calculate_reach_reward` is:

```text
10 * (previous_target_distance - current_target_distance)
- 0.01 * mean(action ** 2)
+ 2.0 if the target is reached
- 1.0 if the step violates a workspace/IK constraint
```

Success is distance within `success_tolerance_m` (default 0.04 m). An invalid workspace command or failed IK command terminates with a constraint violation; reaching the configured episode horizon truncates. This reward is task-specific and separate from the existing pick-and-place adapter, whose recorded environment reward remains zero.

## What is implemented and what is not

Implemented modules are `roboforge.environments.panda_reach` (task, reward, Gymnasium API, IK heuristic) and `roboforge.experiments.rl` (validated PPO config, training/checkpoint callback, reload, paired policy evaluation, versioned metrics, and recorder integration). Tests cover the Gymnasium contract, deterministic resets, reward and termination semantics, baseline reach success, paired evaluation and records, and optional short PPO train/save/reload.

The PPO pipeline is an initial simulation baseline. It does not learn a world model, select experiments by uncertainty, identify physical dynamics, optimize pick-and-place, validate robustness under domain randomization, interface with physical robots, or demonstrate sim-to-real transfer. Use the paired metrics as measured outcomes for the selected seeds and configuration, not as a claim of broad policy quality.
