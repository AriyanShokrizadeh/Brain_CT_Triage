"""Checkpoint and model-weight utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import Tensor, nn

from src.configs.schemas import PathsConfig
from src.utils.io import prepare_output_path, require_file
from src.utils.types import CheckpointMode


@dataclass(frozen=True, slots=True)
class BestCheckpoint:
    """Best checkpoint information."""

    path: Path
    score: float
    mode: CheckpointMode


def experiment_weights_path(
    paths: PathsConfig,
    experiment_name: str,
    filename: str,
) -> Path:
    """Return the weight path for an experiment."""
    return paths.artifacts.weights / experiment_name / filename


def save_weights(module: nn.Module, path: str | Path) -> Path:
    """Save model weights."""
    path = prepare_output_path(path)
    torch.save(module.state_dict(), path)
    return path


def load_weights[T: nn.Module](module: T, path: str | Path) -> T:
    """Load model weights."""
    path = require_file(path, description="Weight file")
    state = torch.load(path, map_location="cpu", weights_only=True)
    module.load_state_dict(state)
    return module


def best_checkpoint(trainer: Trainer) -> BestCheckpoint:
    """Return the trainer's best checkpoint."""
    callback = trainer.checkpoint_callback

    if not isinstance(callback, ModelCheckpoint):
        raise RuntimeError("Trainer does not use ModelCheckpoint.")

    if not callback.best_model_path or callback.best_model_score is None:
        raise RuntimeError("No best checkpoint is available.")

    if callback.mode not in ("min", "max"):
        raise RuntimeError(f"Unsupported checkpoint mode: {callback.mode!r}.")

    return BestCheckpoint(
        path=Path(callback.best_model_path),
        score=float(callback.best_model_score),
        mode=cast(CheckpointMode, callback.mode),
    )


def export_checkpoint_weights(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    prefix: str,
) -> Path:
    """Export module weights from a trusted Lightning checkpoint."""
    checkpoint_path = require_file(
        checkpoint_path,
        description="Checkpoint",
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state = checkpoint.get("state_dict")

    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint has no valid state_dict: {checkpoint_path}")

    weights: dict[str, Tensor] = {
        name.removeprefix(prefix): value
        for name, value in state.items()
        if isinstance(name, str)
        and isinstance(value, Tensor)
        and name.startswith(prefix)
    }

    if not weights:
        raise RuntimeError(f"No weights found with prefix {prefix!r}.")

    output_path = prepare_output_path(output_path)
    torch.save(weights, output_path)

    return output_path


def is_better_checkpoint(
    current: BestCheckpoint,
    best: BestCheckpoint | None,
) -> bool:
    """Return whether a checkpoint improves on the current best."""
    if best is None:
        return True

    if current.mode != best.mode:
        raise RuntimeError(
            f"Checkpoint mode changed: {best.mode!r} -> {current.mode!r}."
        )

    if current.mode == "min":
        return current.score < best.score

    return current.score > best.score
