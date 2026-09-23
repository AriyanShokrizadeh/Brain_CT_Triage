from __future__ import annotations

from collections.abc import Sequence
from itertools import groupby

import torch
from torch import Tensor, nn

from src.models.backbones.base import Backbone2D


class DinoMultiCropWrapper(nn.Module):
    """Batch views of equal resolution before running the backbone."""

    def __init__(
        self,
        backbone: Backbone2D,
        head: nn.Module,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.head = head

    def forward(self, views: Sequence[Tensor]) -> Tensor:
        outputs: list[Tensor] = []

        for _, crops in groupby(
            views,
            key=lambda view: view.shape[-2:],
        ):
            images = torch.cat(tuple(crops), dim=0)
            features = self.backbone(images).global_feature
            outputs.append(self.head(features))

        return torch.cat(outputs, dim=0)
