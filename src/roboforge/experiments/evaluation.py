"""Aggregate episode outcomes without inventing missing measurements."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import Any, Iterable
import math


@dataclass(frozen=True)
class EvaluationSummary:
    attempted_episodes: int
    successful_episodes: int
    success_rate: float | None
    mean_episode_length: float | None
    median_episode_length: float | None
    mean_placement_error_m: float | None
    failure_counts: dict[str, int]


class ExperimentEvaluator:
    """Summarize explicit task results; no reward-based success inference."""

    def evaluate(self, results: Iterable[Any]) -> EvaluationSummary:
        episodes = list(results)
        successes = 0
        lengths: list[float] = []
        placement_errors: list[float] = []
        failure_counts: dict[str, int] = {}

        for result in episodes:
            success = getattr(result, "success", None)
            if not isinstance(success, bool):
                # Support numpy.bool_ without importing NumPy into this layer.
                if type(success).__name__ != "bool_":
                    raise ValueError("each episode result must expose an explicit boolean 'success'")
                success = bool(success)
            if success and getattr(result, "grasp_verified", None) is False:
                raise ValueError("episode reports success but task grasp verification is false")
            successes += int(success)

            length = getattr(result, "episode_length", None)
            if length is not None:
                if isinstance(length, bool) or not isinstance(length, (int, float)) or not math.isfinite(length) or length < 0:
                    raise ValueError("episode_length must be a non-negative finite number or None")
                lengths.append(float(length))

            error = getattr(result, "final_placement_error_m", None)
            phases = tuple(getattr(result, "phase_history", ()) or ())
            placement_attempted = getattr(result, "placement_attempted", None)
            if placement_attempted is None:
                placement_attempted = any(
                    phase in {"PLACE", "RELEASE", "VERIFY_PLACEMENT", "SUCCESS"}
                    for phase in phases
                )
            if placement_attempted and error is not None:
                if isinstance(error, (int, float)) and math.isfinite(error) and error >= 0:
                    placement_errors.append(float(error))

            if not success:
                reason = getattr(result, "failure_reason", None)
                if isinstance(reason, str) and reason.strip():
                    category = reason.strip()
                else:
                    nonterminal = [p for p in phases if p not in ("FAILURE", "SUCCESS")]
                    category = f"phase:{nonterminal[-1]}" if nonterminal else "unknown"
                failure_counts[category] = failure_counts.get(category, 0) + 1

        attempts = len(episodes)
        return EvaluationSummary(
            attempted_episodes=attempts,
            successful_episodes=successes,
            success_rate=(successes / attempts) if attempts else None,
            mean_episode_length=mean(lengths) if lengths else None,
            median_episode_length=median(lengths) if lengths else None,
            mean_placement_error_m=mean(placement_errors) if placement_errors else None,
            failure_counts=failure_counts,
        )
