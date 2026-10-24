# Third-party notices

## Franka Emika Panda MuJoCo model

The Panda MJCF description and mesh files in `src/roboforge/environments/assets/panda/` are from the [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) Franka Emika Panda model. The asset README identifies derivation from the publicly available [`franka_ros` description](https://github.com/frankaemika/franka_ros/tree/develop/franka_description). The bundled asset directory includes its Apache License 2.0 text, upstream README, and changelog; retain these with the assets. This notice does not change the assets' license.

## Python dependencies

Runtime dependencies are Pydantic, Gymnasium, MuJoCo, and their transitive dependencies. pytest is a development extra; OpenCV is an optional video-recording extra. Consult each upstream package's license when redistributing binary environments.
