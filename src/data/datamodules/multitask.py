"""Multitask supervised data module."""

from functools import partial

import pandas as pd
from monai.transforms.compose import Compose

from src.configs.schemas import (
    AugmentationsConfig,
    DataModuleConfig,
    DatasetConfig,
    LabelsConfig,
    PathsConfig,
    RuntimeConfig,
    TasksConfig,
)
from src.data.collate import multitask_collate
from src.data.datamodules.cross_validation import CrossValidationDataModule
from src.data.datasets.multitask import MultiTaskDataset
from src.data.transforms.pipelines import build_multitask_pipeline
from src.utils.types import CollateFn


class MultiTaskDataModule(CrossValidationDataModule):
    """Data module for supervised multitask training."""

    def __init__(
        self,
        augmentations_config: AugmentationsConfig,
        dataset_config: DatasetConfig,
        labels_config: LabelsConfig,
        paths_config: PathsConfig,
        tasks_config: TasksConfig,
        datamodule_config: DataModuleConfig,
        runtime_config: RuntimeConfig,
        fold: int = 0,
    ) -> None:
        super().__init__(dataset_config, datamodule_config, runtime_config, fold=fold)

        self.augmentations = augmentations_config
        self.dataset = dataset_config
        self.labels = labels_config
        self.paths = paths_config
        self.tasks = tasks_config
        self.datamodule = datamodule_config

    def collate_fn(self) -> CollateFn:
        keys = self.labels.keys

        return partial(multitask_collate, keys=keys)

    def build_transforms(
        self,
        *,
        training: bool,
    ) -> Compose:
        """Build training or validation transforms."""
        return build_multitask_pipeline(
            self.augmentations,
            self.dataset,
            self.labels,
            self.tasks,
            training=training,
        )

    def build_dataset(
        self,
        frame: pd.DataFrame,
        *,
        training: bool,
    ) -> MultiTaskDataset:
        """Build a multitask dataset."""
        transform = self.build_transforms(
            training=training,
        )

        return MultiTaskDataset(
            self.datamodule,
            self.labels,
            self.paths,
            dataframe=frame,
            transform=transform,
        )

    def setup(self, stage: str | None = None) -> None:
        """Create datasets for the requested Lightning stage."""
        if stage not in {None, "fit", "validate"}:
            return

        need_train = stage in {None, "fit"} and self.train_dataset is None
        need_val = self.val_dataset is None

        if not need_train and not need_val:
            return

        train_frame, val_frame = self.read_split(self.paths.data.train.annotated)

        if need_train:
            self.train_dataset = self.build_dataset(train_frame, training=True)

        if need_val:
            self.val_dataset = self.build_dataset(val_frame, training=False)
