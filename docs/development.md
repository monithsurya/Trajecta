# Development

## Environment

Requires Python 3.10 or newer. MuJoCo, Gymnasium, NumPy, and Pydantic are installed from project metadata; pytest and OpenCV are optional extras.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,video]'
```

The `venv/` directory occasionally present in a local copy is not portable and is ignored. Use a newly created `.venv`.

## Checks

```bash
python -m pytest -q
python -m roboforge.visualization --help
python examples/evaluate_pick_place.py --help
python examples/record_pick_place.py --help
```

The viewer command can require a graphical display. Automated environment/task tests run headless. The 1080p recorder requires an offscreen MuJoCo GL context and OpenCV's MP4 writer.

## Workflow and quality

Keep simulator mechanics in the environment, task transitions and success checks in the task, and evaluation/storage out of task control. Add deterministic tests for malformed inputs, failure paths, actuator mapping, and verified outcomes. Record random seeds and configuration. Do not interpret a passing unit test or one successful episode as a benchmark.

Before sharing changes, inspect `git status`, the complete diff, license notices, and generated outputs. Do not commit virtual environments, local JSONL runs, videos, caches, or credentials. Keep bundled Panda mesh assets and their license notice intact.
