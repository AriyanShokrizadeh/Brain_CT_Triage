from __future__ import annotations

from torch import Tensor

from src.configs.schemas import BackboneConfig
from src.models.backbones.base import Backbone2D
from src.models.backbones.cnn import CNNEncoder2D
from src.models.backbones.outputs import BackboneOutput
from src.models.backbones.transformer import ViTBottleneck2D


class TransUNetBackbone2D(Backbone2D):
    """CNN encoder followed by a ViT bottleneck."""

    def __init__(self, backbone_config: BackboneConfig) -> None:
        super().__init__()

        self.cnn = CNNEncoder2D(backbone_config)
        self.transformer = ViTBottleneck2D(
            backbone_config,
            num_channels=self.cnn.channels[-1],
        )

        self.input_channels = backbone_config.cnn.input_channels
        self.output_channels = self.transformer.out_channels
        self.skip_channels = self.cnn.channels[:-1]
        self.reductions = self.cnn.reductions

    def forward(self, images: Tensor) -> BackboneOutput:
        cnn_features = self.cnn(images)
        transformer = self.transformer(cnn_features[-1])

        return BackboneOutput(
            global_feature=transformer.global_feature,
            skip_features=cnn_features[:-1],
            latent_features=transformer.spatial_feature,
        )
