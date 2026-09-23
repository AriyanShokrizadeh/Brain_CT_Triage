from itertools import pairwise
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.parametrizations import weight_norm

from src.configs.schemas import DinoConfig


class DinoHead(nn.Module):
    """DINO projection head."""

    def __init__(
        self,
        dino_config: DinoConfig,
        *,
        latent_channels: int,
    ) -> None:
        super().__init__()

        config = dino_config.head

        self.mlp = self._build_mlp(
            input_dim=latent_channels,
            hidden_dim=config.hidden_dim,
            bottleneck_dim=config.bottleneck_dim,
            num_layers=config.head_layers,
        )

        self.mlp.apply(self.initialize_linear)

        self.last_layer = weight_norm(
            nn.Linear(
                config.bottleneck_dim,
                config.output_dim,
                bias=False,
            ),
            dim=0,
        )

        self.configure_last_layer(
            freeze_scale=config.norm_last_layer,
        )

    @staticmethod
    def _build_mlp(
        *,
        input_dim: int,
        hidden_dim: int,
        bottleneck_dim: int,
        num_layers: int,
    ) -> nn.Sequential:
        """Build the projection MLP."""
        dimensions = [
            input_dim,
            *[hidden_dim] * (num_layers - 1),
            bottleneck_dim,
        ]

        layers: list[nn.Module] = []

        for in_dim, out_dim in pairwise(dimensions[:-1]):
            layers.extend(
                (
                    nn.Linear(in_dim, out_dim),
                    nn.GELU(),
                )
            )

        layers.append(
            nn.Linear(
                dimensions[-2],
                dimensions[-1],
            )
        )

        return nn.Sequential(*layers)

    @staticmethod
    def initialize_linear(module: nn.Module) -> None:
        """Initialize linear layers."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def configure_last_layer(
        self,
        *,
        freeze_scale: bool,
    ) -> None:
        """Initialize and optionally freeze weight-norm scale."""
        layer = cast(Any, self.last_layer)
        scale = layer.parametrizations.weight.original0

        with torch.no_grad():
            scale.fill_(1.0)

        scale.requires_grad_(not freeze_scale)

    def forward(
        self,
        features: Tensor,
    ) -> Tensor:
        projected = self.mlp(features)
        normalized = F.normalize(projected, dim=-1)

        return self.last_layer(normalized)
