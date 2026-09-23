"""Runtime and logging configuration schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import get_args

from src.configs.schemas.validation import (
    check_choice,
    check_non_empty_string,
    check_non_negative,
)
from src.utils.types import LogLevel


@dataclass(slots=True)
class LoggingConfig:
    """Application logging configuration."""

    level: str
    rotation: str
    backtrace: bool
    diagnose: bool

    def __post_init__(self) -> None:
        check_choice("logging.level", self.level, get_args(LogLevel))
        check_non_empty_string("logging.rotation", self.rotation)


@dataclass(slots=True)
class RuntimeConfig:
    """Global runtime configuration."""

    seed: int
    deterministic: bool
    benchmark: bool
    logging: LoggingConfig

    def __post_init__(self) -> None:
        if self.deterministic and self.benchmark:
            raise ValueError(
                "runtime.benchmark must be false when " "runtime.deterministic is true."
            )

        check_non_negative("runtime.seed", self.seed)
