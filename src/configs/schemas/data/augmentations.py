"""Augmentation configuration schemas."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.configs.schemas.validation import (
    check_finite,
    check_fraction_open_upper,
    check_non_empty_string,
    check_non_negative,
    check_positive,
    check_probability,
    check_range,
)


def _to_pair(values: Sequence[float]) -> tuple[float, float]:
    """Convert two values to a float pair."""
    if len(values) != 2:
        raise ValueError("Expected exactly two values.")

    return float(values[0]), float(values[1])


@dataclass(slots=True)
class FlipConfig:
    """Random spatial flip configuration."""

    spatial_axis: int
    prob: float

    def __post_init__(self) -> None:
        if self.spatial_axis not in {0, 1}:
            raise ValueError("flip.spatial_axis must be 0 or 1.")

        check_probability("flip.prob", self.prob)


@dataclass(slots=True)
class RotateConfig:
    """Random in-plane rotation configuration."""

    prob: float
    range_x: float
    mode: str
    padding_mode: str
    keep_size: bool

    def __post_init__(self) -> None:
        check_probability("rotate.prob", self.prob)
        check_non_negative("rotate.range_x", self.range_x)
        check_non_empty_string("rotate.mode", self.mode)
        check_non_empty_string("rotate.padding_mode", self.padding_mode)


@dataclass(slots=True)
class SmoothConfig:
    """Random Gaussian smoothing configuration."""

    sigma_x: tuple[float, float]
    sigma_y: tuple[float, float]
    prob: float

    def __post_init__(self) -> None:
        self.sigma_x = _to_pair(self.sigma_x)
        self.sigma_y = _to_pair(self.sigma_y)

        check_range("smooth.sigma_x", self.sigma_x, strictly_positive=True)
        check_range("smooth.sigma_y", self.sigma_y, strictly_positive=True)
        check_probability("smooth.prob", self.prob)


@dataclass(slots=True)
class ContrastConfig:
    """Random contrast adjustment configuration."""

    prob: float
    gamma: tuple[float, float]

    def __post_init__(self) -> None:
        self.gamma = _to_pair(self.gamma)

        check_range("contrast.gamma", self.gamma, strictly_positive=True)
        check_probability("contrast.prob", self.prob)


@dataclass(slots=True)
class NoiseConfig:
    """Random Gaussian noise configuration."""

    prob: float
    mean: float
    std: float

    def __post_init__(self) -> None:
        check_probability("noise.prob", self.prob)
        check_finite("noise.mean", self.mean)
        check_non_negative("noise.std", self.std)


@dataclass(slots=True)
class GeometryPrimitivesConfig:
    """Shared geometric augmentations."""

    flip: FlipConfig
    rotate: RotateConfig


@dataclass(slots=True)
class IntensityPrimitivesConfig:
    """Shared intensity augmentations."""

    smooth: SmoothConfig
    contrast: ContrastConfig
    noise: NoiseConfig


@dataclass(slots=True)
class PrimitivesConfig:
    """Shared augmentation primitives."""

    geometry: GeometryPrimitivesConfig
    intensity: IntensityPrimitivesConfig


@dataclass(slots=True)
class DetectionZoomConfig:
    """Detection-safe random zoom configuration."""

    prob: float
    min_zoom: float
    max_zoom: float
    mode: str
    padding_mode: str
    keep_size: bool

    def __post_init__(self) -> None:
        check_probability("detection.zoom.prob", self.prob)
        check_range(
            "detection.zoom.range",
            (self.min_zoom, self.max_zoom),
            strictly_positive=True,
        )
        check_non_empty_string("detection.zoom.mode", self.mode)
        check_non_empty_string(
            "detection.zoom.padding_mode",
            self.padding_mode,
        )


@dataclass(slots=True)
class DetectionAugmentationConfig:
    """Detection augmentation configuration."""

    flip: FlipConfig
    zoom: DetectionZoomConfig


@dataclass(slots=True)
class DinoCropConfig:
    """DINO crop-specific augmentation configuration."""

    size: int

    smooth_sigma: tuple[float, float]
    smooth_prob: float

    contrast_gamma: tuple[float, float]
    contrast_prob: float

    noise_std: float
    noise_prob: float

    def __post_init__(self) -> None:
        self.smooth_sigma = _to_pair(self.smooth_sigma)
        self.contrast_gamma = _to_pair(self.contrast_gamma)

        check_positive("dino.crop.size", self.size)
        check_range(
            "dino.crop.smooth_sigma",
            self.smooth_sigma,
            strictly_positive=True,
        )
        check_range(
            "dino.crop.contrast_gamma",
            self.contrast_gamma,
            strictly_positive=True,
        )
        check_non_negative("dino.crop.noise_std", self.noise_std)
        check_probability("dino.crop.smooth_prob", self.smooth_prob)
        check_probability("dino.crop.contrast_prob", self.contrast_prob)
        check_probability("dino.crop.noise_prob", self.noise_prob)


@dataclass(slots=True)
class DinoAugmentationConfig:
    """DINO multi-crop augmentation configuration."""

    num_global_crops: int
    num_local_crops: int

    flip_prob: float
    rotate_range_x: float
    rotate_prob: float

    shift_offset: float
    shift_prob: float

    minimum_foreground_fraction: float
    foreground_channels: list[int]
    foreground_threshold: float
    maximum_crop_attempts: int

    global_crop: DinoCropConfig
    local_crop: DinoCropConfig

    def __post_init__(self) -> None:
        if self.num_global_crops < 2:
            raise ValueError("dino.num_global_crops must be >= 2.")

        check_positive("dino.num_local_crops", self.num_local_crops)
        check_positive("dino.maximum_crop_attempts", self.maximum_crop_attempts)

        if not self.foreground_channels:
            raise ValueError("dino.foreground_channels cannot be empty.")

        if len(self.foreground_channels) != len(set(self.foreground_channels)):
            raise ValueError("dino.foreground_channels must be unique.")

        if min(self.foreground_channels) < 0 or max(self.foreground_channels) >= 3:
            raise ValueError(
                "dino.foreground_channels must contain indices from 0 to 2."
            )

        if self.local_crop.size > self.global_crop.size:
            raise ValueError("dino.local_crop.size must be <= dino.global_crop.size.")

        check_non_negative("dino.rotate_range_x", self.rotate_range_x)
        check_non_negative("dino.shift_offset", self.shift_offset)

        check_probability("dino.flip_prob", self.flip_prob)
        check_probability("dino.rotate_prob", self.rotate_prob)
        check_probability("dino.shift_prob", self.shift_prob)
        check_probability(
            "dino.minimum_foreground_fraction",
            self.minimum_foreground_fraction,
        )

        check_fraction_open_upper(
            "dino.foreground_threshold",
            self.foreground_threshold,
        )

    @property
    def number_of_views(self) -> int:
        """Return the total number of DINO views."""
        return self.num_global_crops + self.num_local_crops


@dataclass(slots=True)
class AugmentationsConfig:
    """Top-level augmentation configuration."""

    primitives: PrimitivesConfig
    detection: DetectionAugmentationConfig
    dino: DinoAugmentationConfig
