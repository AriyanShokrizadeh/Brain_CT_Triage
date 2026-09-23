"""Task-specific cache and DataLoader configuration schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import get_args

from src.configs.schemas.validation import (
    check_choice,
    check_non_negative,
    check_positive,
    check_probability,
)
from src.utils.types import CacheBackend


@dataclass(slots=True)
class CacheConfig:
    """MONAI dataset caching configuration."""

    backend: str
    memory_cache_rate: float
    num_workers: int

    def __post_init__(self) -> None:
        check_choice("cache.backend", self.backend, get_args(CacheBackend))
        check_probability("cache.memory_cache_rate", self.memory_cache_rate)
        check_non_negative("cache.num_workers", self.num_workers)


@dataclass(slots=True)
class DataLoaderConfig:
    """PyTorch DataLoader configuration."""

    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    drop_last: bool

    def __post_init__(self) -> None:
        if self.persistent_workers and self.num_workers == 0:
            raise ValueError(
                "dataloader.persistent_workers requires " "dataloader.num_workers > 0."
            )

        check_positive("dataloader.batch_size", self.batch_size)
        check_non_negative("dataloader.num_workers", self.num_workers)


@dataclass(slots=True)
class DataModuleConfig:
    """Task-specific data loading and caching configuration."""

    cache: CacheConfig
    dataloader: DataLoaderConfig
