"""Loguru configuration helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

from loguru import logger

from src.configs.schemas import RuntimeConfig

LOG_FORMAT: Final[str] = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)


def _log_file_stem(experiment_name: str) -> str:
    """Return a filename-safe experiment name while preserving directory nesting."""
    normalized = experiment_name.replace("\\", "/").rstrip("/")
    stem = normalized.rsplit("/", maxsplit=1)[-1]
    return stem or "experiment"


def configure_logging(
    log_dir: str | Path,
    *,
    experiment_name: str,
    runtime_config: RuntimeConfig,
) -> Path:
    """Configure console and file logging."""
    experiment_name = experiment_name.strip()
    if not experiment_name:
        raise ValueError("experiment_name cannot be empty.")

    config = runtime_config.logging
    experiment_dir = Path(log_dir) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()

    logger.add(
        sys.stderr,
        level=config.level,
        format=LOG_FORMAT,
        colorize=True,
        enqueue=True,
        backtrace=config.backtrace,
        diagnose=config.diagnose,
    )

    log_path = experiment_dir / (
        f"{_log_file_stem(experiment_name)}_{{time:YYYY-MM-DD_HH}}.log"
    )

    logger.add(
        log_path,
        level=config.level,
        format=LOG_FORMAT,
        enqueue=True,
        rotation=config.rotation,
        backtrace=config.backtrace,
        diagnose=config.diagnose,
    )

    logger.info(
        "Logging initialized | experiment={} | directory={} | level={}",
        experiment_name,
        experiment_dir,
        config.level,
    )

    return log_path


def log_runtime_summary(
    runtime_config: RuntimeConfig,
) -> None:
    """Log runtime and reproducibility settings."""
    logger.info(
        "Runtime | seed={} | deterministic={} | benchmark={}",
        runtime_config.seed,
        runtime_config.deterministic,
        runtime_config.benchmark,
    )
