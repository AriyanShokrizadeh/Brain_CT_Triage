"""Lightning training utilities."""

import gc
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    BatchSizeFinder,
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger
from loguru import logger

from src.configs.schemas import ExperimentConfig


def _is_explicit_multi_device(devices: object) -> bool:
    """Return whether multiple training devices are explicitly configured."""
    if isinstance(devices, int):
        return devices > 1

    if isinstance(devices, (list, tuple)):
        return len(devices) > 1

    return False


def _callbacks(
    config: ExperimentConfig,
    checkpoint_dir: Path,
) -> list[Callback]:
    """Build training callbacks."""
    callbacks: list[Callback] = [
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=10),
        RichModelSummary(max_depth=1),
    ]

    if config.batch_size_finder is not None:
        if _is_explicit_multi_device(config.trainer.devices):
            raise ValueError(
                "BatchSizeFinder does not support multi-device training. "
                "Use trainer.devices=1 while tuning the batch size."
            )

        callbacks.append(
            BatchSizeFinder(
                **asdict(config.batch_size_finder),
            )
        )

    if config.early_stopping is not None:
        callbacks.append(
            EarlyStopping(
                **asdict(config.early_stopping),
            )
        )

    if config.checkpoint is not None:
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="epoch-{epoch:03d}",
                auto_insert_metric_name=False,
                **asdict(config.checkpoint),
            )
        )

    return callbacks


def get_trainer(
    *,
    experiment_name: str,
    experiment_config: ExperimentConfig,
    tensorboard_dir: str | Path,
    checkpoints_dir: str | Path,
) -> Trainer:
    """Create a configured Lightning trainer."""
    checkpoint_dir = Path(checkpoints_dir) / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_dir = Path(tensorboard_dir)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    logger_backend = TensorBoardLogger(
        save_dir=str(tensorboard_dir),
        name=experiment_name,
    )

    trainer_config = asdict(experiment_config.trainer)

    logger.info(
        "Creating Trainer | accelerator={} | devices={} | precision={}",
        trainer_config["accelerator"],
        trainer_config["devices"],
        trainer_config["precision"],
    )

    return Trainer(
        **trainer_config,
        callbacks=_callbacks(
            experiment_config,
            checkpoint_dir,
        ),
        logger=logger_backend,
        default_root_dir=checkpoint_dir,
        enable_model_summary=False,
        enable_checkpointing=experiment_config.checkpoint is not None,
    )


def get_fold_trainer(
    *,
    experiment_name: str,
    fold: int,
    experiment_config: ExperimentConfig,
    tensorboard_dir: str | Path,
    checkpoints_dir: str | Path,
) -> Trainer:
    """Create a trainer for a cross-validation fold."""
    return get_trainer(
        experiment_name=f"{experiment_name}/fold_{fold}",
        experiment_config=experiment_config,
        tensorboard_dir=tensorboard_dir,
        checkpoints_dir=checkpoints_dir,
    )


def cleanup_training_resources(
    *,
    cache_backend: str,
    cache_dir: str | Path,
    experiment_name: str,
) -> None:
    """Release training memory and persistent cache resources."""
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if cache_backend != "persistent":
        return

    cache_path = Path(cache_dir) / experiment_name

    if cache_path.exists():
        shutil.rmtree(cache_path)
