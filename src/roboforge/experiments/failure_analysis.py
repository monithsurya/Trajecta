"""Deterministic, evidence-linked initial failure analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from roboforge.experiments.experience import ExperienceRecord


@dataclass(frozen=True)
class FailureDiagnosis:
    episode_id: str
    category: str
    evidence: tuple[str, ...]
    likely_cause: str | None
    confidence: float | None
    recommended_investigation: str


def diagnose(record: ExperienceRecord) -> FailureDiagnosis:
    """Classify explicit failure signals without claiming causal certainty."""
    if record.success:
        return FailureDiagnosis(str(record.episode_id), "success", ("task outcome marked successful",), None, None, "No failure investigation required.")
    reason = (record.failure_reason or "").strip()
    phases = tuple(record.phase_history)
    lower = reason.lower()
    evidence = tuple(x for x in (f"failure_reason={reason}" if reason else "", f"last_phase={phases[-1]}" if phases else "", f"placement_error_m={record.placement_error_m}" if record.placement_error_m is not None else "") if x)
    if "ik" in lower and ("converg" in lower or "reach" in lower):
        category, cause, confidence, advice = "target_unreachable", "The configured target may be outside the solver's reachable workspace.", 0.8, "Check target coordinates, joint limits, and IK residuals."
    elif "timed out" in lower or "timeout" in lower:
        category, cause, confidence, advice = "positioning_timeout", "A task phase did not meet its completion condition before its timeout.", 0.75, "Inspect phase trajectory, position error, joint velocity, and timeout budget."
    elif "grasp" in lower or "finger" in lower or "contact" in lower:
        category, cause, confidence, advice = "grasp_failure", "The recorded grasp verification condition was not satisfied.", 0.75, "Inspect bilateral contacts, gripper opening, cube alignment, and contact forces."
    elif "lift" in lower or "follow" in lower or "dropped" in lower:
        category, cause, confidence, advice = "object_dropped", "The cube did not remain supported during transport or lift.", 0.7, "Inspect grasp force, lift displacement, and transport acceleration."
    elif "placement" in lower or record.placement_error_m is not None:
        category, cause, confidence, advice = "placement_outside_tolerance", "Placement verification failed or the recorded error is outside tolerance.", 0.7, "Compare placement error with task tolerance and inspect release settling."
    elif "collision" in lower or "unsafe" in lower:
        category, cause, confidence, advice = "unsafe_contact", "The task reported a collision or unsafe contact.", 0.8, "Review contact pairs, force limits, and the planned path."
    elif "non-finite" in lower or "invalid action" in lower or "numerical" in lower:
        category, cause, confidence, advice = "numerical_failure", "The task reported invalid numerical state or action data.", 0.85, "Check action bounds and state/controller values around the failure."
    elif "environment terminated" in lower or "max_steps" in lower or "episode" in lower and "truncat" in lower:
        category, cause, confidence, advice = "episode_timeout", "The episode ended before task success was verified.", 0.7, "Inspect episode horizon and the last recorded task phase."
    else:
        category, cause, confidence, advice = "unknown_or_insufficient_evidence", None, None, "Record phase-level errors, contacts, and controller diagnostics on the next run."
    return FailureDiagnosis(str(record.episode_id), category, evidence, cause, confidence, advice)


def analyze(records: Iterable[ExperienceRecord]) -> tuple[FailureDiagnosis, ...]:
    return tuple(diagnose(r) for r in records)
