"""Multitask decoder and fracture-detection configuration schemas."""

from __future__ import annotations

from dataclasses import dataclass

from src.configs.schemas.validation import (
    check_fraction_open_upper,
    check_non_empty_string,
    check_positive,
    check_probability,
)


@dataclass(slots=True)
class DecoderConfig:
    """Segmentation or keypoint decoder configuration."""

    decoder_channels: list[int]
    normalization: str
    activation: str
    dropout: float

    def __post_init__(self) -> None:
        if not self.decoder_channels:
            raise ValueError("decoder.decoder_channels cannot be empty.")

        for index, channel in enumerate(self.decoder_channels):
            check_positive(f"decoder.decoder_channels[{index}]", channel)

        check_non_empty_string("decoder.normalization", self.normalization)
        check_non_empty_string("decoder.activation", self.activation)
        check_fraction_open_upper("decoder.dropout", self.dropout)


@dataclass(slots=True)
class HardNegativeConfig:
    """Hard-negative sampling configuration."""

    batch_size: int
    positive_fraction: float
    minimum_negative: int
    pool_size: float

    def __post_init__(self) -> None:
        check_positive("hard_negative.batch_size", self.batch_size)
        check_probability("hard_negative.positive_fraction", self.positive_fraction)
        check_positive("hard_negative.minimum_negative", self.minimum_negative)
        check_positive("hard_negative.pool_size", self.pool_size)

        if self.minimum_negative > self.batch_size:
            raise ValueError(
                "hard_negative.minimum_negative must be <= "
                "hard_negative.batch_size, "
                f"got {self.minimum_negative} > {self.batch_size}."
            )


@dataclass(slots=True)
class InferenceConfig:
    """Detection inference configuration."""

    score_threshold: float
    nms_threshold: float
    max_detections: int

    def __post_init__(self) -> None:
        check_probability("inference.score_threshold", self.score_threshold)
        check_probability("inference.nms_threshold", self.nms_threshold)
        check_positive("inference.max_detections", self.max_detections)


@dataclass(slots=True)
class AnchorBoxesConfig:
    """Detection anchor-box configuration."""

    shapes: list[list[int]]

    def __post_init__(self) -> None:
        if not self.shapes:
            raise ValueError("anchor_boxes.shapes cannot be empty.")

        for index, shape in enumerate(self.shapes):
            if len(shape) != 2:
                raise ValueError(
                    "anchor_boxes.shapes entries must contain "
                    f"[height, width], got {shape}."
                )

            check_positive(f"anchor_boxes.shapes[{index}][0]", shape[0])
            check_positive(f"anchor_boxes.shapes[{index}][1]", shape[1])


@dataclass(slots=True)
class DetectionConfig:
    """Fracture detection configuration."""

    fpn_channels: int
    atss_candidates: int
    hard_negative: HardNegativeConfig
    inference: InferenceConfig
    anchor_boxes: AnchorBoxesConfig

    def __post_init__(self) -> None:
        check_positive("detection.fpn_channels", self.fpn_channels)
        check_positive("detection.atss_candidates", self.atss_candidates)

        if self.fpn_channels % 8 != 0:
            raise ValueError(
                "detection.fpn_channels must be divisible by 8 "
                "for GroupNorm, "
                f"got {self.fpn_channels}."
            )


@dataclass(slots=True)
class MultitaskConfig:
    """Multitask model configuration."""

    hemorrhage_decoder: DecoderConfig
    keypoint_decoder: DecoderConfig
    detection: DetectionConfig
