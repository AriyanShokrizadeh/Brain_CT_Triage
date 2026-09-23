"""Metadata validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.configs.schemas import DatasetConfig, LabelsConfig


class MetadataValidator:
    """Validate prepared metadata."""

    def __init__(
        self,
        dataset_config: DatasetConfig,
        labels_config: LabelsConfig,
        metadata: pd.DataFrame,
        labeled: bool,
    ) -> None:
        self.columns = labels_config.metadata
        self.cross_validation = dataset_config.cross_validation
        self.metadata = metadata
        self.labeled = labeled

    @property
    def kind(self) -> str:
        """Return metadata type."""
        return "labeled" if self.labeled else "unlabeled"

    @property
    def hemorrhage_area_columns(self) -> tuple[str, ...]:
        """Return hemorrhage area columns."""
        return tuple(f"{col}_Area" for col in self.columns.hemorrhage_columns)

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Return required metadata columns."""
        columns = [
            self.columns.series_id,
            self.cross_validation.group_column,
            self.columns.slice_id,
            *self.columns.spacing,
            self.columns.relative_path,
        ]

        if self.labeled:
            columns.extend(
                [
                    self.cross_validation.stratify_column,
                    *self.columns.hemorrhage_columns,
                    *self.hemorrhage_area_columns,
                    self.columns.fracture_prob,
                    self.columns.MLS_mm,
                    self.columns.triage_class,
                ]
            )

        return tuple(dict.fromkeys(columns))

    def _check(self, invalid: pd.Series, message: str) -> None:
        """Raise when invalid rows are found."""
        invalid = invalid.fillna(True).astype(bool)
        if not invalid.any():
            return

        rows = self.metadata.index[invalid].tolist()
        sample = rows[:10]

        logger.error("{} | invalid_rows={:,} | sample={}", message, len(rows), sample)
        raise ValueError(f"{message} Invalid rows={len(rows)}, sample={sample}.")

    def _numeric(self, column: str) -> pd.Series:
        """Return a column converted to numeric values."""
        return pd.to_numeric(self.metadata[column], errors="coerce")

    @staticmethod
    def _not_finite(values: pd.Series) -> pd.Series:
        """Return non-finite numeric values."""
        return values.isna() | ~np.isfinite(values)

    def _validate_columns(self) -> None:
        """Validate required columns."""
        if missing := sorted(set(self.required_columns) - set(self.metadata.columns)):
            raise ValueError(f"Missing required metadata columns: {missing}.")

    def _validate_identifiers(self) -> None:
        """Validate identifiers, paths, and slice uniqueness."""
        columns = dict.fromkeys(
            (
                self.columns.series_id,
                self.cross_validation.group_column,
                self.columns.slice_id,
                self.columns.relative_path,
            )
        )

        for col in columns:
            values = self.metadata[col].astype("string").str.strip()
            self._check(
                values.isna() | values.eq(""), f"{col} must be present and non-empty."
            )

        duplicates = self.metadata.duplicated(
            subset=[self.columns.series_id, self.columns.slice_id],
            keep=False,
        )
        self._check(duplicates, "series_id/slice_id pairs must be unique.")

    def _validate_spacing(self) -> None:
        """Validate physical spacing."""
        for col in self.columns.spacing:
            values = self._numeric(col)
            self._check(
                self._not_finite(values) | values.le(0),
                f"{col} must be finite and greater than 0.",
            )

    def _validate_hemorrhages(self) -> None:
        """Validate hemorrhage labels and areas."""
        for label_col, area_col in zip(
            self.columns.hemorrhage_columns,
            self.hemorrhage_area_columns,
            strict=True,
        ):
            labels = self._numeric(label_col)
            self._check(
                self._not_finite(labels) | ~labels.isin((0, 1)),
                f"{label_col} must be 0 or 1.",
            )

            areas = self._numeric(area_col)
            self._check(
                self._not_finite(areas) | areas.lt(0),
                f"{area_col} must be finite and non-negative.",
            )

            self._check(
                labels.eq(1) != areas.gt(0),
                f"{label_col} and {area_col} are inconsistent.",
            )

    def _validate_targets(self) -> None:
        """Validate supervised scalar targets."""
        stratify_col = self.cross_validation.stratify_column
        stratify = self.metadata[stratify_col].astype("string").str.strip()
        self._check(
            stratify.isna() | stratify.eq(""),
            f"{stratify_col} must be present for cross-validation.",
        )

        fracture_col = self.columns.fracture_prob
        fracture = self._numeric(fracture_col)
        self._check(
            self._not_finite(fracture) | ~fracture.between(0.0, 1.0),
            f"{fracture_col} must be between 0 and 1.",
        )

        mls_col = self.columns.MLS_mm
        mls = self._numeric(mls_col)
        self._check(
            self._not_finite(mls) | mls.lt(0),
            f"{mls_col} must be finite and non-negative.",
        )

        triage_col = self.columns.triage_class
        triage = self._numeric(triage_col)
        self._check(
            self._not_finite(triage) | ~triage.isin((0, 1, 2)),
            f"{triage_col} must be 0, 1, or 2.",
        )

    def __call__(self) -> pd.DataFrame:
        """Validate and return metadata."""
        logger.info("Validating {} metadata | rows={:,}", self.kind, len(self.metadata))

        self._validate_columns()
        self._validate_identifiers()
        self._validate_spacing()

        if self.labeled:
            self._validate_hemorrhages()
            self._validate_targets()

        logger.success(
            "{} metadata validated | rows={:,}",
            self.kind.capitalize(),
            len(self.metadata),
        )

        return self.metadata
