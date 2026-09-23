"""Training, optimizer, checkpoint, and callback schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import get_args

from src.configs.schemas.validation import (
    check_choice,
    check_non_empty_string,
    check_non_negative,
    check_positive,
)
from src.utils.types import BatchSizeFinderMode, CheckpointMode


@dataclass(slots=True)
class TrainerConfig:
    """PyTorch Lightning trainer configuration."""

    max_epochs: int
    accelerator: str
    devices: str | int
    precision: str
    gradient_clip_val: float
    log_every_n_steps: int
    accumulate_grad_batches: int

    def __post_init__(self) -> None:
        check_positive("trainer.max_epochs", self.max_epochs)
        check_non_empty_string("trainer.accelerator", self.accelerator)
        check_non_empty_string("trainer.precision", self.precision)
        check_non_negative("trainer.gradient_clip_val", self.gradient_clip_val)
        check_positive("trainer.log_every_n_steps", self.log_every_n_steps)
        check_positive("trainer.accumulate_grad_batches", self.accumulate_grad_batches)

        if isinstance(self.devices, int):
            check_positive("trainer.devices", self.devices)
        else:
            check_non_empty_string("trainer.devices", self.devices)


@dataclass(slots=True)
class OptimizerConfig:
    """Optimizer and learning-rate schedule configuration."""

    learning_rate: float
    min_learning_rate: float
    weight_decay: float
    warmup_epochs: int

    def __post_init__(self) -> None:
        check_positive("optimizer.learning_rate", self.learning_rate)
        check_non_negative(
            "optimizer.min_learning_rate",
            self.min_learning_rate,
        )
        check_non_negative("optimizer.weight_decay", self.weight_decay)
        check_non_negative("optimizer.warmup_epochs", self.warmup_epochs)

        if self.min_learning_rate > self.learning_rate:
            raise ValueError(
                "optimizer.min_learning_rate must be <= "
                "optimizer.learning_rate, "
                f"got {self.min_learning_rate} > "
                f"{self.learning_rate}."
            )


@dataclass(slots=True)
class BatchSizeFinderConfig:
    """Automatic batch-size search configuration."""

    mode: str
    steps_per_trial: int
    init_val: int
    max_trials: int
    batch_arg_name: str

    def __post_init__(self) -> None:
        check_choice("batch_size_finder.mode", self.mode, get_args(BatchSizeFinderMode))
        check_positive("batch_size_finder.steps_per_trial", self.steps_per_trial)
        check_positive("batch_size_finder.init_val", self.init_val)
        check_positive("batch_size_finder.max_trials", self.max_trials)
        check_non_empty_string("batch_size_finder.batch_arg_name", self.batch_arg_name)


@dataclass(slots=True)
class CheckpointConfig:
    """Model checkpoint configuration."""

    monitor: str
    mode: str
    save_top_k: int
    save_last: bool
    every_n_epochs: int

    def __post_init__(self) -> None:
        check_non_empty_string("checkpoint.monitor", self.monitor)
        check_choice("checkpoint.mode", self.mode, get_args(CheckpointMode))
        check_positive("checkpoint.every_n_epochs", self.every_n_epochs)

        if self.save_top_k < -1:
            raise ValueError(
                "checkpoint.save_top_k must be -1, 0, "
                "or a positive integer, "
                f"got {self.save_top_k}."
            )


@dataclass(slots=True)
class EarlyStoppingConfig:
    """Early-stopping configuration."""

    monitor: str
    mode: str
    patience: int
    min_delta: float
    verbose: bool

    def __post_init__(self) -> None:
        check_non_empty_string("early_stopping.monitor", self.monitor)
        check_choice("early_stopping.mode", self.mode, get_args(CheckpointMode))
        check_non_negative("early_stopping.patience", self.patience)
        check_non_negative("early_stopping.min_delta", self.min_delta)


@dataclass(slots=True)
class ExperimentConfig:
    """Complete training experiment configuration."""

    trainer: TrainerConfig
    optimizer: OptimizerConfig
    batch_size_finder: BatchSizeFinderConfig | None = None
    checkpoint: CheckpointConfig | None = None
    early_stopping: EarlyStoppingConfig | None = None

    def __post_init__(self) -> None:
        if self.optimizer.warmup_epochs > self.trainer.max_epochs:
            raise ValueError(
                "optimizer.warmup_epochs must be <= trainer.max_epochs, "
                f"got {self.optimizer.warmup_epochs} > "
                f"{self.trainer.max_epochs}."
            )
