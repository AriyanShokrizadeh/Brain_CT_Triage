"""Reusable MONAI augmentation builders."""

from __future__ import annotations

from collections.abc import Sequence

from monai.apps.detection.transforms.dictionary import (
    ClipBoxToImaged,
    RandFlipBoxd,
    RandZoomBoxd,
)
from monai.transforms.intensity.dictionary import (
    RandAdjustContrastd,
    RandGaussianNoised,
    RandGaussianSmoothd,
)
from monai.transforms.spatial.dictionary import RandFlipd, RandRotated
from monai.transforms.transform import Transform

from src.configs.schemas import AugmentationsConfig, LabelsConfig


def build_detection_geometric_transforms(
    augmentation_config: AugmentationsConfig,
    labels_config: LabelsConfig,
    *,
    image_keys: Sequence[str],
    reference_image_key: str,
    zoom_modes: Sequence[str],
) -> list[Transform]:
    """Build synchronized augmentations with fracture boxes."""
    if len(image_keys) != len(zoom_modes):
        raise ValueError("image_keys and zoom_modes must have the same length.")

    config = augmentation_config.detection
    keys = labels_config.keys

    return [
        RandFlipBoxd(
            image_keys=image_keys,
            box_keys=keys.box,
            box_ref_image_keys=reference_image_key,
            prob=config.flip.prob,
            spatial_axis=config.flip.spatial_axis,
        ),
        RandZoomBoxd(
            image_keys=image_keys,
            box_keys=keys.box,
            box_ref_image_keys=reference_image_key,
            prob=config.zoom.prob,
            min_zoom=config.zoom.min_zoom,
            max_zoom=config.zoom.max_zoom,
            mode=zoom_modes,
            padding_mode=config.zoom.padding_mode,
            keep_size=config.zoom.keep_size,
        ),
        ClipBoxToImaged(
            box_keys=keys.box,
            label_keys=keys.label,
            box_ref_image_keys=reference_image_key,
            remove_empty=True,
        ),
    ]


def build_geometric_transforms(
    augmentation_config: AugmentationsConfig,
    labels_config: LabelsConfig,
) -> list[Transform]:
    """Build image-only geometric augmentations for supervised probing."""
    config = augmentation_config.primitives.geometry
    image_key = labels_config.keys.image
    return [
        RandFlipd(
            keys=image_key,
            spatial_axis=config.flip.spatial_axis,
            prob=config.flip.prob,
        ),
        RandRotated(
            keys=image_key,
            prob=config.rotate.prob,
            range_x=config.rotate.range_x,
            mode=config.rotate.mode,
            padding_mode=config.rotate.padding_mode,
            keep_size=config.rotate.keep_size,
        ),
    ]


def build_intensity_transforms(
    augmentation_config: AugmentationsConfig,
    labels_config: LabelsConfig,
) -> list[Transform]:
    """Build image-only intensity augmentations shared by supervised tasks."""
    config = augmentation_config.primitives.intensity
    image_key = labels_config.keys.image
    return [
        RandGaussianSmoothd(
            keys=image_key,
            sigma_x=config.smooth.sigma_x,
            sigma_y=config.smooth.sigma_y,
            prob=config.smooth.prob,
        ),
        RandAdjustContrastd(
            keys=image_key,
            gamma=config.contrast.gamma,
            prob=config.contrast.prob,
        ),
        RandGaussianNoised(
            keys=image_key,
            mean=config.noise.mean,
            std=config.noise.std,
            prob=config.noise.prob,
        ),
    ]
