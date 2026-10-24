from roboforge.experiments.dataset import ExperienceDataset
from roboforge.experiments.experience import ExperienceRecord
from roboforge.experiments.failure_analysis import diagnose


def make_record(*, experiment_id, success=False, reason="grasp was not verified"):
    return ExperienceRecord(task_id="pick_and_place", environment_id="test", experiment_id=experiment_id,
                            success=success, outcome="success" if success else "failure",
                            failure_reason=None if success else reason, phase_history=("DESCEND", "FAILURE"))


def test_dataset_split_keeps_experiments_whole_and_manifest_is_stable():
    a, b = make_record(experiment_id="00000000-0000-0000-0000-000000000001"), make_record(experiment_id="00000000-0000-0000-0000-000000000002", success=True)
    dataset = ExperienceDataset([a, b])
    split1, split2 = dataset.grouped_split(seed=4), dataset.grouped_split(seed=4)
    assert split1 == split2
    assert {r.experiment_id for r in split1["train"]}.isdisjoint({r.experiment_id for r in split1["test"]})
    assert dataset.manifest() == dataset.manifest()


def test_failure_analysis_reports_evidence_and_unknown_case():
    diagnosis = diagnose(make_record(experiment_id="00000000-0000-0000-0000-000000000001"))
    assert diagnosis.category == "grasp_failure"
    assert diagnosis.confidence is not None and diagnosis.evidence
    unknown = diagnose(make_record(experiment_id="00000000-0000-0000-0000-000000000002", reason=""))
    assert unknown.category == "unknown_or_insufficient_evidence"
