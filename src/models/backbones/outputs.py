from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, slots=True)
class BackboneOutput:
    """Shared output of the ResNet -> ViT backbone."""

    global_feature: Tensor
    skip_features: tuple[Tensor, ...]
    latent_features: Tensor


@dataclass(frozen=True, slots=True)
class TransformerOutput:
    """Global and spatial features returned by the transformer bottleneck."""

    global_feature: Tensor
    spatial_feature: Tensor
