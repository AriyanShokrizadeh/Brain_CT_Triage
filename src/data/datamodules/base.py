"""Shared Lightning data-module infrastructure."""

from typing import Any

from lightning.pytorch import LightningDataModule
from loguru import logger
from monai.data.dataloader import DataLoader
from torch.cuda import is_available as cuda_is_available
from torch.utils.data import Dataset

from src.configs.schemas import DataModuleConfig, RuntimeConfig
from src.utils.types import CollateFn, Sample


class BaseDataModule(LightningDataModule):
    """Base class for shared DataLoader behavior."""

    def __init__(
        self,
        datamodule_config: DataModuleConfig,
        runtime_config: RuntimeConfig,
    ) -> None:
        super().__init__()

        self.loader = datamodule_config.dataloader
        self.batch_size = self.loader.batch_size
        self.runtime = runtime_config

        self.train_dataset: Dataset[Sample] | None = None

    def collate_fn(self) -> CollateFn | None:
        """Return a task-specific collate function."""
        return None

    def dataloader(
        self,
        dataset: Dataset[Sample],
        *,
        training: bool,
    ) -> DataLoader:
        """Build a DataLoader."""
        pin_memory = self.loader.pin_memory and cuda_is_available()
        persistent_workers = (
            self.loader.persistent_workers and self.loader.num_workers > 0
        )

        kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "shuffle": training,
            "num_workers": self.loader.num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers,
            "drop_last": training and self.loader.drop_last,
        }

        collate_fn = self.collate_fn()
        if collate_fn is not None:
            kwargs["collate_fn"] = collate_fn

        logger.debug(
            "DataLoader | training={} | initial batch_size={} | workers={} | pin_memory={}",
            training,
            self.batch_size,
            self.loader.num_workers,
            pin_memory,
        )

        return DataLoader(dataset, **kwargs)

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader."""
        if self.train_dataset is None:
            raise RuntimeError(
                "Training dataset is not initialized. Call setup() first."
            )

        return self.dataloader(
            self.train_dataset,
            training=True,
        )
