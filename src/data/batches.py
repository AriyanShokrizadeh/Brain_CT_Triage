"""Typed batches produced by task-specific collate functions."""

from typing import NamedTuple

from torch import Tensor

from src.utils.types import DetBatch


class MultiTaskBatch(NamedTuple):
    """Ready-to-train batch for supervised multitask learning."""

    images: Tensor
    masks: Tensor
    heatmaps: Tensor
    detection_targets: DetBatch
    keypoint_active: bool
