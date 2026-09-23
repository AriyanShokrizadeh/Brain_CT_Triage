"""Frozen-backbone linear-probe data module."""

import numpy as np
import pandas as pd
from loguru import logger

from src.configs.schemas import (
    AugmentationsConfig,
    DataModuleConfig,
    DatasetConfig,
    LabelsConfig,
    PathsConfig,
    RuntimeConfig,
)
from src.data.datamodules.cross_validation import CrossValidationDataModule
from src.data.datasets.probe import LinearProbeDataset
from src.data.transforms.pipelines import build_probe_pipeline


class LinearProbeDataModule(CrossValidationDataModule):
    """Data module for frozen-backbone linear probing."""

    def __init__(
        self,
        augmentations_config: AugmentationsConfig,
        dataset_config: DatasetConfig,
        labels_config: LabelsConfig,
        paths_config: PathsConfig,
        datamodule_config: DataModuleConfig,
        runtime_config: RuntimeConfig,
        *,
        fold: int = 0,
    ) -> None:
        super().__init__(dataset_config, datamodule_config, runtime_config, fold=fold)

        self.augmentations = augmentations_config
        self.dataset = dataset_config
        self.labels = labels_config
        self.paths = paths_config
        self.datamodule = datamodule_config
        self.mls_stats: tuple[float, float] | None = None

    def build_dataset(
        self, frame: pd.DataFrame, *, training: bool
    ) -> LinearProbeDataset:
        """Build a probe dataset with appropriate transforms."""
        transform = build_probe_pipeline(
            self.augmentations,
            self.dataset,
            self.labels,
            training=training,
        )

        return LinearProbeDataset(
            self.labels,
            self.datamodule,
            self.paths,
            dataframe=frame,
            transform=transform,
        )

    def setup(self, stage: str | None = None) -> None:
        """Create datasets and compute MLS statistics."""
        if stage not in {None, "fit", "validate"}:
            return

        need_train = stage in {None, "fit"} and self.train_dataset is None
        need_val = self.val_dataset is None
        need_stats = self.mls_stats is None

        if not any((need_train, need_val, need_stats)):
            return

        train_frame, val_frame = self.read_split(self.paths.data.train.annotated)

        if need_stats:
            self.mls_stats = self._compute_mls_stats(train_frame)
            logger.info(
                "MLS stats | fold={} | mean={:.4f} | std={:.4f}",
                self.fold,
                self.mls_stats[0],
                self.mls_stats[1],
            )

        if need_train:
            self.train_dataset = self.build_dataset(train_frame, training=True)

        if need_val:
            self.val_dataset = self.build_dataset(val_frame, training=False)

    def get_mls_stats(self) -> tuple[float, float]:
        """Return training-fold MLS statistics."""
        if self.mls_stats is None:
            raise RuntimeError(
                "MLS statistics are not initialized. Call setup() first."
            )

        return self.mls_stats

    def _compute_mls_stats(self, frame: pd.DataFrame) -> tuple[float, float]:
        """Compute training-fold MLS statistics."""
        col = self.labels.metadata.MLS_mm

        if col not in frame.columns:
            raise KeyError(f"Missing column: {col!r}")

        values = frame[col].to_numpy(dtype=np.float32)

        if values.size == 0:
            raise ValueError(f"{col!r} is empty.")

        if not np.isfinite(values).all():
            raise ValueError(f"{col!r} must contain finite values.")

        mean, std = float(values.mean()), float(values.std())

        if std <= 0:
            raise ValueError(f"{col!r} must have positive variance.")

        return mean, std
