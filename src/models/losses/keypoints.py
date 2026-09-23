"""Keypoint heatmap losses."""

from __future__ import annotations

from math import isfinite

import torch.nn.functional as F
from torch import Tensor, nn


class KeypointHeatmapLoss(nn.Module):
    """Peak-weighted Smooth L1 loss for valid keypoint heatmaps."""

    def __init__(self, peak_weight: float) -> None:
        super().__init__()

        if not isfinite(peak_weight) or peak_weight < 0:
            raise ValueError(
                f"peak_weight must be finite and non-negative, got {peak_weight}."
            )

        self.peak_weight = peak_weight

    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """Compute loss over keypoints present in the target."""
        if predictions.shape != targets.shape:
            raise ValueError(
                "Predictions and targets must have the same shape, "
                f"got {tuple(predictions.shape)} and {tuple(targets.shape)}."
            )

        valid = targets.amax(dim=(-2, -1)) > 0

        if not valid.any():
            return predictions.sum() * 0.0

        predictions = predictions[valid]
        targets = targets[valid]

        loss = F.smooth_l1_loss(predictions, targets, reduction="none")

        weights = 1.0 + self.peak_weight * targets

        return (loss * weights).mean()
