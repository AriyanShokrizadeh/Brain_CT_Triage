from __future__ import annotations

from einops import rearrange
from torch import Tensor, nn
from transformers import ViTConfig, ViTModel

from src.configs.schemas import BackboneConfig
from src.models.backbones.outputs import TransformerOutput


class ViTBottleneck2D(nn.Module):
    """Map CNN spatial features to a ViT sequence and back to a feature map."""

    def __init__(
        self,
        backbone_config: BackboneConfig,
        *,
        num_channels: int,
    ) -> None:
        super().__init__()

        config = backbone_config.transformer
        self.out_channels = config.hidden_size

        vit_config = ViTConfig(
            patch_size=1,
            qkv_bias=True,
            num_channels=num_channels,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_depths,
            num_attention_heads=config.num_heads,
            intermediate_size=config.hidden_size * config.mlp_ratio,
            hidden_dropout_prob=config.dropout,
            attention_probs_dropout_prob=config.dropout,
            image_size=config.nominal_feature_size,
        )

        self.model = ViTModel(
            vit_config,
            add_pooling_layer=False,
        )

    def forward(self, features: Tensor) -> TransformerOutput:
        height, width = features.shape[-2:]

        output = self.model(
            pixel_values=features,
            interpolate_pos_encoding=True,
            return_dict=True,
        )

        tokens = output.last_hidden_state

        return TransformerOutput(
            global_feature=tokens[:, 0],
            spatial_feature=rearrange(
                tokens[:, 1:],
                "b (h w) c -> b c h w",
                h=height,
                w=width,
            ),
        )
