"""Keypoint heatmap target construction."""

from collections.abc import Mapping

import numpy as np
from scipy.ndimage import gaussian_filter

from src.configs.schemas import TasksConfig
from src.utils.types import Point


def build_keypoint_heatmap(
    keypoints: Mapping[str, Point | None],
    shape: tuple[int, int],
    *,
    tasks_config: TasksConfig,
) -> np.ndarray:
    """Build normalized Gaussian keypoint heatmaps."""
    height, width = shape
    config = tasks_config.keypoints

    if unknown := set(keypoints) - set(config.class_names):
        raise ValueError(f"Unknown keypoints: {sorted(unknown)}")

    heatmaps = np.zeros((len(config.class_names), height, width), dtype=np.float32)

    for channel, name in enumerate(config.class_names):
        if (point := keypoints.get(name)) is None:
            continue

        x, y = point
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(
                f"Keypoint {name!r} is outside the image: ({x}, {y}) for shape {shape}."
            )

        heatmaps[channel, min(round(y), height - 1), min(round(x), width - 1)] = 1.0

    heatmaps = gaussian_filter(heatmaps, sigma=(0, config.sigma, config.sigma))

    maxima = heatmaps.max(axis=(1, 2), keepdims=True)
    np.divide(heatmaps, maxima, out=heatmaps, where=maxima > 0)

    return heatmaps
