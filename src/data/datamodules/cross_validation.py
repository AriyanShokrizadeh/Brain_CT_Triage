"""Fold-based supervised data-module infrastructure."""

from pathlib import Path

import pandas as pd
from loguru import logger
from monai.data.dataloader import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset

from src.configs.schemas import DataModuleConfig, DatasetConfig, RuntimeConfig
from src.data.datamodules.base import BaseDataModule
from src.utils.io import read_csv
from src.utils.types import Sample


class CrossValidationDataModule(BaseDataModule):
    """Base class for stratified group cross-validation."""

    def __init__(
        self,
        dataset_config: DatasetConfig,
        datamodule_config: DataModuleConfig,
        runtime_config: RuntimeConfig,
        *,
        fold: int = 0,
    ) -> None:
        super().__init__(
            datamodule_config,
            runtime_config,
        )

        self.cross_validation = dataset_config.cross_validation
        self.n_splits = self.cross_validation.n_splits

        self.fold = fold

        self.val_dataset: Dataset[Sample] | None = None

        if not 0 <= self.fold < self.n_splits:
            raise ValueError(f"fold must be between 0 and {self.n_splits - 1}.")

    def _split_columns(
        self,
        frame: pd.DataFrame,
    ) -> tuple[pd.Series, pd.Series]:
        """Validate and return split columns."""
        group_col = self.cross_validation.group_column
        target_col = self.cross_validation.stratify_column

        required = [group_col, target_col]
        missing = [col for col in required if col not in frame.columns]

        if missing:
            raise KeyError(f"Missing split columns: {missing}")

        split_data = frame[required]

        if split_data.isna().any().any():
            raise ValueError("Split columns cannot contain missing values.")

        groups = split_data[group_col]
        targets = split_data[target_col]

        if groups.nunique() < self.n_splits:
            raise ValueError(f"At least {self.n_splits} groups are required.")

        if targets.nunique() < 2:
            raise ValueError("Stratification requires at least two classes.")

        return groups, targets

    def split(
        self,
        frame: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create the selected cross-validation fold."""
        groups, targets = self._split_columns(frame)

        splitter = StratifiedGroupKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.runtime.seed,
        )

        splits = splitter.split(frame, y=targets, groups=groups)

        train_idx, val_idx = list(splits)[self.fold]
        train = frame.iloc[train_idx].reset_index(drop=True)
        val = frame.iloc[val_idx].reset_index(drop=True)

        logger.info(
            "Data split | fold={} | train={:,} | validation={:,}",
            self.fold,
            len(train),
            len(val),
        )

        return train, val

    def read_split(
        self,
        path: str | Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Read metadata and create the selected fold."""
        frame = read_csv(path)
        return self.split(frame)

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        if self.val_dataset is None:
            raise RuntimeError(
                "Validation dataset is not initialized. Call setup() first."
            )

        return self.dataloader(
            self.val_dataset,
            training=False,
        )
