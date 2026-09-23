"""Reproducibility and deterministic-execution helpers."""

from __future__ import annotations

import lightning.pytorch as pl
import torch
from monai.utils.misc import set_determinism

from src.configs.schemas import RuntimeConfig


def configure_reproducibility(
    runtime_config: RuntimeConfig,
) -> None:
    """Configure random seeds and deterministic execution."""
    torch.set_float32_matmul_precision(precision="medium")

    pl.seed_everything(
        runtime_config.seed,
        workers=True,
    )

    if runtime_config.deterministic:
        set_determinism(seed=runtime_config.seed)

    torch.backends.cudnn.deterministic = runtime_config.deterministic
    torch.backends.cudnn.benchmark = runtime_config.benchmark
