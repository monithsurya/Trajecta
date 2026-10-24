# Robot model and control

## Bundled model

The package includes the MuJoCo Menagerie Franka Emika Panda description and mesh assets under `src/roboforge/environments/assets/panda/`. The default scene is selected by `get_default_asset_path()`. The simulation model includes the Panda arm, hand, table, and target cube. The model's own Apache-2.0 license, README, and changelog are preserved in the asset directory. See `THIRD_PARTY_NOTICES.md` before redistributing changes to these assets.

## Controls and units

`MujocoRobotEnvironment` exposes normalized float32 actions in `[-1, 1]`, one per MuJoCo actuator in model order. Each value is scaled to that actuator's `ctrlrange`. `actuator_specs` exposes actuator names, transmission targets, ranges, gain, and bias metadata; do not assume that every actuator is a joint position actuator. Observations concatenate MuJoCo `qpos` and `qvel` as float32. `observation_layout` maps scalar joint addresses.

Panda IK/controller targets are world-frame positions in meters and joint targets are radians. The gripper command is mapped using the named tendon actuator. Task logic uses named model objects and actual contact/lift state for success verification.

## Viewer and demo

```bash
python -m roboforge.visualization
mjpython examples/pick_place_panda.py --seed 0
```

The interactive viewer needs a working graphical environment. On macOS, use MuJoCo's `mjpython` launcher for the passive pick-and-place task demo.

## Hardware boundary

No hardware backend, network transport, emergency-stop channel, or physical robot support is included. Simulation success is not sim-to-real validation. Hardware work must separately validate joint/workspace limits, units, control timing, communication timeouts, sensor/state calibration, collision and contact constraints, an independent emergency stop, safe fault behavior, and supervised low-speed commissioning.
