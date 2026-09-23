"""Dataset preprocessing, splitting, and preparation configuration schemas."""

from __future__ import annotations

from dataclasses import dataclass

from src.configs.schemas.validation import check_non_empty_string, check_positive


@dataclass(slots=True)
class CTWindowConfig:
    """CT intensity window bounds in Hounsfield units."""

    lower: int
    upper: int

    def __post_init__(self) -> None:
        if self.lower >= self.upper:
            raise ValueError(
                "CT window must satisfy lower < upper, "
                f"got lower={self.lower} and upper={self.upper}."
            )


@dataclass(slots=True)
class CTWindowsConfig:
    """Named CT intensity windows."""

    brain: CTWindowConfig
    blood: CTWindowConfig
    bone: CTWindowConfig


@dataclass(slots=True)
class DicomConfig:
    """DICOM image preprocessing configuration."""

    image_size: list[int]
    ct_windows: CTWindowsConfig

    def __post_init__(self) -> None:
        if len(self.image_size) != 2:
            raise ValueError(
                "dicom.image_size must contain exactly two values "
                f"[height, width], got {self.image_size}."
            )

        height, width = self.image_size
        check_positive("dicom.image_size[0]", height)
        check_positive("dicom.image_size[1]", width)


@dataclass(slots=True)
class CrossValidationConfig:
    """Cross-validation split configuration."""

    group_column: str
    stratify_column: str
    n_splits: int

    def __post_init__(self) -> None:
        check_non_empty_string("cross_validation.group_column", self.group_column)
        check_non_empty_string("cross_validation.stratify_column", self.stratify_column)

        if self.n_splits < 2:
            raise ValueError(
                "cross_validation.n_splits must be >= 2, " f"got {self.n_splits}."
            )


@dataclass(slots=True)
class ProcessingConfig:
    """Parallel dataset preparation configuration."""

    ingestion_workers: int

    def __post_init__(self) -> None:
        check_positive("processing.ingestion_workers", self.ingestion_workers)


@dataclass(slots=True)
class DatasetConfig:
    """Shared dataset definition and preparation configuration."""

    dicom: DicomConfig
    cross_validation: CrossValidationConfig
    processing: ProcessingConfig
