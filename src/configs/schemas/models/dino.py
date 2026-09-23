"""DINO model and loss configuration schemas."""

from __future__ import annotations

from dataclasses import dataclass

from src.configs.schemas.validation import (
    check_fraction_open_upper,
    check_non_negative,
    check_positive,
)


@dataclass(slots=True)
class DinoLossConfig:
    """DINO teacher-student loss configuration."""

    teacher_momentum: float
    center_momentum: float
    student_temperature: float
    teacher_temperature_start: float
    teacher_temperature_end: float
    temperature_warmup_epochs: int

    def __post_init__(self) -> None:
        check_fraction_open_upper("loss.teacher_momentum", self.teacher_momentum)
        check_fraction_open_upper("loss.center_momentum", self.center_momentum)
        check_positive("loss.student_temperature", self.student_temperature)
        check_positive("loss.teacher_temperature_start", self.teacher_temperature_start)
        check_positive("loss.teacher_temperature_end", self.teacher_temperature_end)
        check_non_negative(
            "loss.temperature_warmup_epochs",
            self.temperature_warmup_epochs,
        )

        if self.teacher_temperature_start > self.teacher_temperature_end:
            raise ValueError(
                "loss.teacher_temperature_start must be <= "
                "loss.teacher_temperature_end, "
                f"got {self.teacher_temperature_start} > "
                f"{self.teacher_temperature_end}."
            )


@dataclass(slots=True)
class DinoHeadConfig:
    """DINO projection-head configuration."""

    output_dim: int
    hidden_dim: int
    bottleneck_dim: int
    head_layers: int
    norm_last_layer: bool
    freeze_last_layer_epochs: int

    def __post_init__(self) -> None:
        check_positive("head.output_dim", self.output_dim)
        check_positive("head.hidden_dim", self.hidden_dim)
        check_positive("head.bottleneck_dim", self.bottleneck_dim)
        check_positive("head.head_layers", self.head_layers)
        check_non_negative(
            "head.freeze_last_layer_epochs",
            self.freeze_last_layer_epochs,
        )


@dataclass(slots=True)
class DinoConfig:
    """DINO model configuration."""

    loss: DinoLossConfig
    head: DinoHeadConfig
