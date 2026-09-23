"""RetinaNet fracture-detection head."""

from collections import OrderedDict
from collections.abc import Sequence

from monai.apps.detection.networks.retinanet_network import (
    RetinaNetClassificationHead,
    RetinaNetRegressionHead,
)
from monai.networks.blocks.feature_pyramid_network import (
    FeaturePyramidNetwork,
    LastLevelP6P7,
)
from torch import Tensor, nn

from src.configs.schemas import DetectionConfig
from src.models.backbones.outputs import BackboneOutput
from src.models.detection.runtime import RetinaNetRuntime2D
from src.utils.types import DetBatch, DetOutput


class BackboneFPN2D(nn.Module):
    """Build an FPN from shared backbone features."""

    def __init__(
        self,
        *,
        skip_channels: Sequence[int],
        latent_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        in_channels = (*skip_channels[1:], latent_channels)
        self.num_levels = len(in_channels) + 2

        self.fpn = FeaturePyramidNetwork(
            spatial_dims=2,
            in_channels_list=list(in_channels),
            out_channels=out_channels,
            extra_blocks=LastLevelP6P7(
                spatial_dims=2,
                in_channels=latent_channels,
                out_channels=out_channels,
            ),
        )

    def forward(self, features: BackboneOutput) -> list[Tensor]:
        maps = (*features.skip_features[1:], features.latent_features)
        inputs = OrderedDict(
            (f"c{index}", feature) for index, feature in enumerate(maps, start=2)
        )
        return list(self.fpn(inputs).values())


class FractureRetinaHead2D(nn.Module):
    """Predict fracture boxes from shared backbone features."""

    def __init__(
        self,
        detection_config: DetectionConfig,
        *,
        skip_channels: Sequence[int],
        latent_channels: int,
        size_divisible: int,
        box_key: str,
        label_key: str,
    ) -> None:
        super().__init__()

        anchors = detection_config.anchor_boxes.shapes
        channels = detection_config.fpn_channels

        self.fpn = BackboneFPN2D(
            skip_channels=skip_channels,
            latent_channels=latent_channels,
            out_channels=channels,
        )
        self.classifier = RetinaNetClassificationHead(
            channels,
            len(anchors),
            num_classes=1,
            spatial_dims=2,
        )
        self.regressor = RetinaNetRegressionHead(
            channels,
            len(anchors),
            spatial_dims=2,
        )
        self.runtime = RetinaNetRuntime2D(
            detection_config,
            anchor_shapes=anchors,
            feature_map_scales=tuple(2**i for i in range(self.fpn.num_levels)),
            size_divisible=size_divisible,
            box_key=box_key,
            label_key=label_key,
        )

    @property
    def score_key(self) -> str:
        """Return the prediction score key."""
        return self.runtime.score_key

    @property
    def max_detections(self) -> int:
        """Return the maximum number of detections per image."""
        return self.runtime.max_detections

    def forward(self, features: BackboneOutput) -> DetOutput:
        maps = self.fpn(features)
        return {
            "classification": self.classifier(maps),
            "box_regression": self.regressor(maps),
        }

    def loss(
        self,
        images: Tensor,
        outputs: DetOutput,
        targets: DetBatch,
    ) -> dict[str, Tensor]:
        """Compute fracture-detection losses."""
        return self.runtime.loss(images, outputs, targets)

    def predict(
        self,
        images: Tensor,
        outputs: DetOutput,
    ) -> DetBatch:
        """Decode fracture predictions."""
        return self.runtime.predict(images, outputs)
