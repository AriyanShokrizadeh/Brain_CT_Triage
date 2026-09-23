"""Spatial resizing transforms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from monai.apps.detection.transforms.array import ResizeBox
from monai.transforms.transform import Transform

from src.utils.types import Sample


class ResizeBoxd(Transform):
    """Resize boxes to match a resized reference image."""

    def __init__(
        self,
        *,
        box_key: str,
        image_key: str,
        spatial_size: Sequence[int],
    ) -> None:
        self.box_key = box_key
        self.image_key = image_key
        self.resizer = ResizeBox(spatial_size=spatial_size)

    def __call__(self, data: Sample) -> Sample:
        result = dict(data)

        source_size = result[self.image_key].shape[-2:]

        result[self.box_key] = self.resizer(
            result[self.box_key],
            src_spatial_size=source_size,
        )

        return cast(Sample, result)
