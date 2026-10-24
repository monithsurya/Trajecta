"""Reproducible dataset manifests, filtering, export, and leakage-safe splits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable
from uuid import UUID

from roboforge.experiments.experience import ExperienceRecord, JsonlExperienceStore


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    schema_versions: tuple[int, ...]
    record_count: int
    source_path: str
    record_ids: tuple[str, ...]
    content_sha256: str
    split_group: str = "experiment_id"


class ExperienceDataset:
    """Validated view over JSONL records; transforms never fabricate fields."""

    def __init__(self, records: Iterable[ExperienceRecord], *, source_path: str = "") -> None:
        self.records = tuple(ExperienceRecord.model_validate(r) for r in records)
        self.source_path = source_path

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "ExperienceDataset":
        store = JsonlExperienceStore(path)
        return cls(store.read_all(), source_path=str(Path(path).resolve()))

    def filter(self, *, task_id: str | None = None, success: bool | None = None,
               experiment_id: UUID | str | None = None) -> "ExperienceDataset":
        experiment = str(experiment_id) if experiment_id is not None else None
        selected = (r for r in self.records if (task_id is None or r.task_id == task_id)
                    and (success is None or r.success == success)
                    and (experiment is None or str(r.experiment_id) == experiment))
        return ExperienceDataset(selected, source_path=self.source_path)

    def manifest(self) -> DatasetManifest:
        lines = [r.model_dump_json(exclude_none=False) for r in self.records]
        digest = hashlib.sha256(("\n".join(lines) + ("\n" if lines else "")).encode()).hexdigest()
        return DatasetManifest(digest[:16], tuple(sorted({r.schema_version for r in self.records})),
                               len(self.records), self.source_path,
                               tuple(str(r.episode_id) for r in self.records), digest)

    def export_jsonl(self, path: str | Path) -> None:
        JsonlExperienceStore(path).write_many(self.records)

    def grouped_split(self, *, train_fraction: float = 0.8, seed: int = 0) -> dict[str, tuple[ExperienceRecord, ...]]:
        """Assign whole experiments to train/test using a stable seeded hash."""
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be strictly between 0 and 1")
        groups: dict[str, list[ExperienceRecord]] = {}
        for record in self.records:
            groups.setdefault(str(record.experiment_id), []).append(record)
        train, test = [], []
        for group, rows in groups.items():
            value = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:16], 16) / 2**64
            (train if value < train_fraction else test).extend(rows)
        # A single experiment cannot be split without leakage; make this explicit.
        return {"train": tuple(train), "test": tuple(test)}

    def write_manifest(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self.manifest()), indent=2) + "\n", encoding="utf-8")
