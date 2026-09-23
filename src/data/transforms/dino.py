"""Foreground-aware DINO multi-crop augmentation."""

from __future__ import annotations

import numpy as np
from monai.transforms.compose import Compose
from monai.transforms.croppad.array import RandSpatialCrop
from monai.transforms.intensity.array import (
    RandAdjustContrast,
    RandGaussianNoise,
    RandGaussianSmooth,
    RandShiftIntensity,
)
from monai.transforms.spatial.array import RandFlip, RandRotate
from monai.transforms.transform import MapTransform, RandomizableTransform
from numpy.random import RandomState
from torch import Tensor

from src.configs.schemas.data.augmentations import (
    DinoAugmentationConfig,
    DinoCropConfig,
)
from src.utils.types import Sample


class DinoMultiCropd(RandomizableTransform, MapTransform):
    """Create foreground-aware global and local DINO views from a CT tensor."""

    def __init__(
        self,
        config: DinoAugmentationConfig,
        *,
        image_key: str,
        allow_missing_keys: bool = False,
    ) -> None:
        RandomizableTransform.__init__(self, prob=1.0)
        MapTransform.__init__(
            self,
            keys=image_key,
            allow_missing_keys=allow_missing_keys,
        )

        self.config = config

        self.global_crop = RandSpatialCrop(
            roi_size=(
                config.global_crop.size,
                config.global_crop.size,
            ),
            random_size=False,
        )
        self.local_crop = RandSpatialCrop(
            roi_size=(
                config.local_crop.size,
                config.local_crop.size,
            ),
            random_size=False,
        )

        self.global_augment = self._build_augmentations(config.global_crop)
        self.local_augment = self._build_augmentations(config.local_crop)

    def __call__(self, data: Sample) -> Sample:
        result = dict(data)

        for key in self.key_iterator(result):
            image = result[key]

            if not isinstance(image, Tensor):
                raise TypeError(f"Expected {key!r} to contain a Tensor.")

            self._validate_image(image)

            result[key] = [
                *self._make_views(
                    image,
                    crop_transform=self.global_crop,
                    augmentation=self.global_augment,
                    number_of_views=self.config.num_global_crops,
                ),
                *self._make_views(
                    image,
                    crop_transform=self.local_crop,
                    augmentation=self.local_augment,
                    number_of_views=self.config.num_local_crops,
                ),
            ]

        return result

    def set_random_state(
        self,
        seed: int | None = None,
        state: RandomState | None = None,
    ) -> DinoMultiCropd:
        """Seed every nested MONAI random transform reproducibly."""
        super().set_random_state(seed=seed, state=state)
        max_seed = np.iinfo(np.int32).max
        for transform in (
            self.global_crop,
            self.local_crop,
            self.global_augment,
            self.local_augment,
        ):
            seed = int(self.R.randint(0, max_seed))
            transform.set_random_state(seed=seed)
        return self

    def _validate_image(self, image: Tensor) -> None:
        channels, height, width = image.shape[-3:]

        if max(self.config.foreground_channels) >= channels:
            raise ValueError(
                f"Invalid foreground channel for {channels}-channel image."
            )

        crop_size = max(
            self.config.global_crop.size,
            self.config.local_crop.size,
        )

        if crop_size > min(height, width):
            raise ValueError(
                f"Crop size {crop_size} exceeds image size " f"{height}x{width}."
            )

    def _make_views(
        self,
        image: Tensor,
        *,
        crop_transform: RandSpatialCrop,
        augmentation: Compose,
        number_of_views: int,
    ) -> list[Tensor]:
        views: list[Tensor] = []
        for _ in range(number_of_views):
            crop = self._foreground_crop(image, crop_transform)
            view = augmentation(crop)
            if not isinstance(view, Tensor):
                raise TypeError("DINO augmentation must return a Tensor.")
            views.append(view)
        return views

    def _foreground_crop(
        self, image: Tensor, crop_transform: RandSpatialCrop
    ) -> Tensor:
        best_crop: Tensor | None = None
        best_ratio = -1.0

        for _ in range(self.config.maximum_crop_attempts):
            crop = crop_transform(image)
            if not isinstance(crop, Tensor):
                raise TypeError("DINO crop transform must return a Tensor.")

            ratio = self._foreground_ratio(crop)
            if ratio >= self.config.minimum_foreground_fraction:
                return crop
            if ratio > best_ratio:
                best_crop = crop
                best_ratio = ratio

        if best_crop is None:
            raise RuntimeError("Failed to generate a DINO crop.")
        return best_crop

    def _foreground_ratio(self, image: Tensor) -> float:
        """Return the fraction of pixels classified as foreground."""
        channels = image[self.config.foreground_channels]
        foreground = channels.amax(dim=0)

        foreground_mask = foreground > self.config.foreground_threshold

        return foreground_mask.float().mean().item()

    def _build_augmentations(self, crop_config: DinoCropConfig) -> Compose:
        config = self.config
        return Compose(
            [
                RandFlip(
                    spatial_axis=1,
                    prob=config.flip_prob,
                ),
                RandRotate(
                    range_x=config.rotate_range_x,
                    prob=config.rotate_prob,
                    keep_size=True,
                ),
                RandGaussianSmooth(
                    sigma_x=crop_config.smooth_sigma,
                    sigma_y=crop_config.smooth_sigma,
                    prob=crop_config.smooth_prob,
                ),
                RandAdjustContrast(
                    gamma=crop_config.contrast_gamma,
                    prob=crop_config.contrast_prob,
                ),
                RandShiftIntensity(
                    offsets=config.shift_offset,
                    prob=config.shift_prob,
                ),
                RandGaussianNoise(
                    std=crop_config.noise_std,
                    prob=crop_config.noise_prob,
                ),
            ]
        )
