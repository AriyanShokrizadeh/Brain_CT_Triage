"""Shared pytest fixtures."""

from dataclasses import replace
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from src.configs import (
    load_dino_datamodule_config,
    load_multitask_datamodule_config,
    load_probe_datamodule_config,
)
from src.configs.schemas import DataModuleConfig
from src.data.datamodules.dino import DinoDataModule
from src.data.datamodules.multitask import MultiTaskDataModule
from src.data.datamodules.probe import LinearProbeDataModule
from src.utils.experiments import TrainingContext, load_training_context


def _test_config(config: DataModuleConfig) -> DataModuleConfig:
    """Return lightweight datamodule settings for tests."""
    return replace(
        config,
        cache=replace(
            config.cache,
            backend="none",
            memory_cache_rate=0.0,
            num_workers=0,
        ),
        dataloader=replace(
            config.dataloader,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            drop_last=False,
        ),
    )


@pytest.fixture(scope="session")
def training_context() -> TrainingContext:
    """Load shared project configuration."""
    return load_training_context()


@pytest.fixture(scope="session")
def figures_dir(
    training_context: TrainingContext,
) -> Path:
    """Return the visualization output directory."""
    path = training_context.paths.artifacts.figures
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="module")
def dino_datamodule(
    training_context: TrainingContext,
) -> DinoDataModule:
    """Return an initialized DINO datamodule."""
    context = training_context

    datamodule = DinoDataModule(
        augmentations_config=context.augmentations,
        dataset_config=context.dataset,
        labels_config=context.labels,
        datamodule_config=_test_config(load_dino_datamodule_config()),
        paths_config=context.paths,
        runtime_config=context.runtime,
    )

    datamodule.setup("fit")
    return datamodule


@pytest.fixture(scope="module")
def probe_datamodule(
    training_context: TrainingContext,
) -> LinearProbeDataModule:
    """Return an initialized linear-probe datamodule."""
    context = training_context

    datamodule = LinearProbeDataModule(
        augmentations_config=context.augmentations,
        dataset_config=context.dataset,
        labels_config=context.labels,
        datamodule_config=_test_config(load_probe_datamodule_config()),
        paths_config=context.paths,
        runtime_config=context.runtime,
    )

    datamodule.setup("fit")
    return datamodule


@pytest.fixture(scope="module")
def multitask_datamodule(
    training_context: TrainingContext,
) -> MultiTaskDataModule:
    """Return an initialized multitask datamodule."""
    context = training_context

    datamodule = MultiTaskDataModule(
        augmentations_config=context.augmentations,
        dataset_config=context.dataset,
        labels_config=context.labels,
        datamodule_config=_test_config(load_multitask_datamodule_config()),
        paths_config=context.paths,
        tasks_config=context.tasks,
        runtime_config=context.runtime,
    )

    datamodule.setup("fit")
    return datamodule
