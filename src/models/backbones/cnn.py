from itertools import pairwise
from typing import Any

import timm
from torch import Tensor, nn

from src.configs.schemas import BackboneConfig


class CNNEncoder2D(nn.Module):
    """Expose hierarchical feature maps from a timm backbone."""

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()

        cnn = config.cnn

        self.encoder: Any = timm.create_model(
            cnn.architecture,
            in_chans=cnn.input_channels,
            pretrained=cnn.pretrained,
            features_only=True,
        )

        self.channels = tuple(self.encoder.feature_info.channels())
        self.reductions = tuple(self.encoder.feature_info.reduction())

        if len(self.channels) < 2:
            raise ValueError(
                f"{cnn.architecture!r} must expose at least two feature levels."
            )

        if self.reductions[0] != 2:
            raise ValueError(
                f"First feature reduction must be 2, got {self.reductions[0]}."
            )

        if any(
            current != previous * 2 for previous, current in pairwise(self.reductions)
        ):
            raise ValueError(f"Feature reductions must double, got {self.reductions}.")

    def forward(self, images: Tensor) -> tuple[Tensor, ...]:
        return tuple(self.encoder(images))
