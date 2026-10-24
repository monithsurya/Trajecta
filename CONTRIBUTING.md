# Contributing

Contributions should preserve the simulation-first scope and keep claims tied to code and measured evidence.

1. Create a focused change and explain the behavior it addresses.
2. Set up a clean Python environment using [the development guide](docs/development.md).
3. Add or update tests for the relevant success and failure paths.
4. Run `python -m pytest -q` and any task-specific simulation check.
5. Include reproducibility details (configuration and seeds) for reported results; distinguish smoke checks from evaluations.
6. Review the full diff for generated outputs, personal paths, secrets, large files, and third-party license changes.
7. Preserve the Panda asset notices. Do not relicense third-party model files as project code.

Keep issues and pull requests specific: state the observed behavior, expected behavior, how to reproduce it, and the environment used. Avoid claims of hardware readiness or learned performance without evidence.
