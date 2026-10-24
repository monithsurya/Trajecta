"""Trajecta experiments package.

Contains experiment specifications, validation logic, and (in future chunks)
experiment orchestration interfaces.
"""

from roboforge.experiments.spec import (
    MAX_SEED_VALUE,
    ControllerConfig,
    ControllerType,
    ExperimentSpec,
    TaskType,
)
from roboforge.experiments.evaluation import EvaluationSummary, ExperimentEvaluator
from roboforge.experiments.experience import ExperienceRecord, ExperienceRecorder, JsonlExperienceStore
from roboforge.experiments.runner import ExperimentRun, ExperimentRunner
from roboforge.experiments.failure_analysis import FailureDiagnosis, analyze, diagnose
from roboforge.experiments.dataset import DatasetManifest, ExperienceDataset

__all__ = [
    "MAX_SEED_VALUE",
    "ControllerConfig",
    "ControllerType",
    "ExperimentSpec",
    "TaskType",
    "EvaluationSummary",
    "ExperimentEvaluator",
    "ExperienceRecord",
    "ExperienceRecorder",
    "JsonlExperienceStore",
    "ExperimentRun",
    "ExperimentRunner",
    "FailureDiagnosis",
    "analyze",
    "diagnose",
    "DatasetManifest",
    "ExperienceDataset",
]
