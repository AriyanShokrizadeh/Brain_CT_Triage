"""DINO self-supervised data module."""

from monai.transforms.compose import Compose

from src.configs.schemas import (
    AugmentationsConfig,
    DataModuleConfig,
    DatasetConfig,
    LabelsConfig,
    PathsConfig,
    RuntimeConfig,
)
from src.data.datamodules.base import BaseDataModule
from src.data.datasets.dino import DinoDataset
from src.data.transforms.pipelines import build_dino_pipeline
from src.utils.io import read_csv


class DinoDataModule(BaseDataModule):
    """Data module for DINO pretraining."""

    def __init__(
        self,
        augmentations_config: AugmentationsConfig,
        dataset_config: DatasetConfig,
        labels_config: LabelsConfig,
        paths_config: PathsConfig,
        datamodule_config: DataModuleConfig,
        runtime_config: RuntimeConfig,
    ) -> None:
        super().__init__(
            datamodule_config,
            runtime_config,
        )

        self.augmentations = augmentations_config
        self.dataset = dataset_config
        self.labels = labels_config
        self.paths = paths_config
        self.datamodule = datamodule_config

    def build_transforms(self) -> Compose:
        """Build DINO training transforms."""
        return build_dino_pipeline(
            self.augmentations,
            self.dataset,
            self.labels,
        )

    def setup(self, stage: str | None = None) -> None:
        """Create the DINO training dataset."""
        if stage not in {None, "fit"}:
            return

        if self.train_dataset is not None:
            return

        dataframe = read_csv(self.paths.data.train.unannotated)

        self.train_dataset = DinoDataset(
            self.datamodule,
            self.labels,
            self.paths,
            dataframe=dataframe,
            transform=self.build_transforms(),
        )
