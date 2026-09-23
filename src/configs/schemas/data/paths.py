"""Project path configuration schemas."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

from src.configs.schemas.validation import check_non_empty_string


@dataclass(slots=True)
class TrainPathsConfig:
    """Training dataset paths."""

    dicoms: Path
    annotations: Path
    annotated: Path
    unannotated: Path


@dataclass(slots=True)
class PredictPathsConfig:
    """Prediction dataset paths."""

    dicoms: Path
    annotations: Path
    annotated: Path


@dataclass(slots=True)
class DataPathsConfig:
    """Project dataset paths."""

    source: Path
    train: TrainPathsConfig
    predict: PredictPathsConfig


@dataclass(slots=True)
class ArtifactPathsConfig:
    """Generated artifact directories."""

    cache: Path
    figures: Path
    logs: Path
    checkpoints: Path
    tensorboard: Path
    weights: Path


@dataclass(slots=True)
class ExperimentNamesConfig:
    """Experiment names used for artifact directories."""

    dino: str
    probe: str
    multitask: str
    metadata: str

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            check_non_empty_string(f"experiments.{field.name}", value)


@dataclass(slots=True)
class PathsConfig:
    """Project data, artifact, and experiment paths."""

    data: DataPathsConfig
    artifacts: ArtifactPathsConfig
    experiments: ExperimentNamesConfig
