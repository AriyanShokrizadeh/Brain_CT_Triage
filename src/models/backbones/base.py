from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn

from src.models.backbones.outputs import BackboneOutput


class Backbone2D(nn.Module, ABC):
    """Base interface for all 2D feature-extraction backbones."""

    input_channels: int
    output_channels: int
    skip_channels: tuple[int, ...]
    reductions: tuple[int, ...]

    @abstractmethod
    def forward(
        self,
        images: Tensor,
    ) -> BackboneOutput: ...
