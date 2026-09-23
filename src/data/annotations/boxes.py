"""Fracture-detection target construction."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from src.configs.schemas import TasksConfig
from src.utils.types import Box


def build_fracture_targets(
    boxes_xywh: Sequence[Box],
    *,
    tasks_config: TasksConfig,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Convert annotation XYWH boxes to MONAI StandardMode in image-axis order.

    The DICOM loader uses swap_ij=False, so image spatial axes are [H, W].
    Therefore MONAI's ij-indexed box coordinates are [y1, x1, y2, x2].
    """
    boxes = np.asarray(boxes_xywh, dtype=np.float32).reshape(-1, 4)

    x, y, w, h = boxes.T
    boxes = np.column_stack((y, x, y + h, x + w))

    labels = np.full(
        len(boxes),
        tasks_config.fracture.class_index,
        dtype=np.int64,
    )

    return boxes, labels
