"""Base MONAI dataset."""

from typing import cast

from monai.data.dataset import CacheDataset, Dataset, PersistentDataset
from monai.data.utils import pickle_hashing
from monai.transforms.transform import Transform
from torch.utils.data import Dataset as TorchDataset

from src.configs.schemas import DataModuleConfig, PathsConfig
from src.utils.types import Sample


class BaseDataset(TorchDataset[Sample]):
    """Dataset with configurable MONAI caching."""

    def __init__(
        self,
        datamodule_config: DataModuleConfig,
        paths_config: PathsConfig,
        *,
        data: list[Sample],
        transform: Transform,
        cache_namespace: str,
    ) -> None:
        if not data:
            raise ValueError("Dataset is empty.")

        config = datamodule_config.cache

        if config.backend == "persistent":
            cache_dir = paths_config.artifacts.cache / cache_namespace
            cache_dir.mkdir(parents=True, exist_ok=True)

            self.dataset = PersistentDataset(
                data=data,
                transform=transform,
                cache_dir=cache_dir,
                hash_transform=pickle_hashing,
            )

        elif config.backend == "memory":
            self.dataset = CacheDataset(
                data=data,
                transform=transform,
                cache_rate=config.memory_cache_rate,
                num_workers=config.num_workers,
            )

        else:
            self.dataset = Dataset(
                data=data,
                transform=transform,
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Sample:
        return cast(Sample, self.dataset[index])
