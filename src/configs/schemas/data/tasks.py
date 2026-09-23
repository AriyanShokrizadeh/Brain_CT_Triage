"""Supervised task target configuration schemas."""

from __future__ import annotations

from dataclasses import dataclass

from src.configs.schemas.validation import (
    check_positive,
    check_string_sequence,
)


@dataclass(slots=True)
class HemorrhageConfig:
    """Hemorrhage segmentation class configuration."""

    class_names: list[str]

    def __post_init__(self) -> None:
        if self.class_names[0] != "BG":
            raise ValueError("hemorrhage.class_names[0] must be 'BG'.")

        check_string_sequence("hemorrhage.class_names", self.class_names)


@dataclass(slots=True)
class KeypointsConfig:
    """Midline keypoint target configuration."""

    class_names: list[str]
    sigma: float

    def __post_init__(self) -> None:
        check_string_sequence("keypoints.class_names", self.class_names)
        check_positive("keypoints.sigma", self.sigma)


@dataclass(slots=True)
class FractureConfig:
    """Fracture detection target configuration."""

    class_index: int

    def __post_init__(self) -> None:
        if self.class_index != 0:
            raise ValueError("fracture.class_index must be 0 for a one-class detector.")


@dataclass(slots=True)
class TasksConfig:
    """Supervised task target configuration."""

    hemorrhage: HemorrhageConfig
    keypoints: KeypointsConfig
    fracture: FractureConfig
