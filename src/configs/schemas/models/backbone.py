"""Backbone architecture configuration schemas."""

from __future__ import annotations

from dataclasses import dataclass

from src.configs.schemas.validation import (
    check_fraction_open_upper,
    check_non_empty_string,
    check_positive,
)


@dataclass(slots=True)
class CNNConfig:
    """CNN encoder configuration."""

    architecture: str
    input_channels: int
    pretrained: bool

    def __post_init__(self) -> None:
        check_non_empty_string("cnn.architecture", self.architecture)
        check_positive("cnn.input_channels", self.input_channels)


@dataclass(slots=True)
class TransformerConfig:
    """Transformer encoder configuration."""

    hidden_size: int
    num_depths: int
    num_heads: int
    mlp_ratio: int
    dropout: float
    nominal_feature_size: int

    def __post_init__(self) -> None:
        check_positive("transformer.hidden_size", self.hidden_size)
        check_positive("transformer.num_depths", self.num_depths)
        check_positive("transformer.num_heads", self.num_heads)
        check_positive("transformer.mlp_ratio", self.mlp_ratio)
        check_fraction_open_upper("transformer.dropout", self.dropout)
        check_positive("transformer.nominal_feature_size", self.nominal_feature_size)

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                "transformer.hidden_size must be divisible by "
                "transformer.num_heads, "
                f"got {self.hidden_size} and {self.num_heads}."
            )


@dataclass(slots=True)
class BackboneConfig:
    """TransUNet backbone configuration."""

    cnn: CNNConfig
    transformer: TransformerConfig
