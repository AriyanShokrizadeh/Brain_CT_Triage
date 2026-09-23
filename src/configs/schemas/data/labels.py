"""Metadata and sample schema configuration."""

from __future__ import annotations

from dataclasses import dataclass, fields

from src.configs.schemas.validation import (
    check_non_empty_string,
    check_string_sequence,
)


@dataclass(slots=True)
class MetadataConfig:
    """Normalized metadata column names."""

    series_id: str
    slice_id: str
    relative_path: str
    triage_class: str

    spacing: list[str]
    hemorrhage_columns: list[str]

    fracture_prob: str
    MLS_mm: str

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, str):
                check_non_empty_string(f"metadata.{field.name}", value)

        check_string_sequence("metadata.spacing", self.spacing)
        check_string_sequence("metadata.hemorrhage_columns", self.hemorrhage_columns)


@dataclass(slots=True)
class KeysConfig:
    """Keys exchanged between datasets, transforms, and models."""

    image: str
    annotations: str

    hemorrhage_mask: str
    hemorrhage_labels: str
    keypoint_heatmap: str

    box: str
    label: str
    target: str

    fracture_prob: str
    MLS_mm: str

    def __post_init__(self) -> None:
        values: list[str] = []

        for field in fields(self):
            value = getattr(self, field.name)
            check_non_empty_string(f"keys.{field.name}", value)
            values.append(value)

        check_string_sequence("keys", values)


@dataclass(slots=True)
class LabelsConfig:
    """Metadata, submission, and in-memory sample schema."""

    metadata: MetadataConfig
    keys: KeysConfig
