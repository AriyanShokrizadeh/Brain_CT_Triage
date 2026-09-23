from __future__ import annotations

from torch import Tensor, nn

from src.configs.schemas import LabelsConfig, MultitaskConfig, TasksConfig
from src.models.backbones.base import Backbone2D
from src.models.decoders.unet import DenseDecoder2D
from src.models.detection.fracture import FractureRetinaHead2D
from src.models.multitask.outputs import MultiTaskOutput


class MultiTaskNet(nn.Module):
    """Shared backbone with task-specific dense and detection heads."""

    def __init__(
        self,
        multitask_config: MultitaskConfig,
        tasks_config: TasksConfig,
        labels_config: LabelsConfig,
        *,
        backbone: Backbone2D,
    ) -> None:
        super().__init__()

        self.backbone = backbone

        self.hemorrhage_decoder = DenseDecoder2D(
            multitask_config.hemorrhage_decoder,
            input_channels=backbone.input_channels,
            skip_channels=backbone.skip_channels,
            bottleneck_channels=backbone.output_channels,
            out_channels=len(tasks_config.hemorrhage.class_names),
        )

        self.keypoint_decoder = DenseDecoder2D(
            multitask_config.keypoint_decoder,
            input_channels=backbone.input_channels,
            skip_channels=backbone.skip_channels,
            bottleneck_channels=backbone.output_channels,
            out_channels=len(tasks_config.keypoints.class_names),
        )

        self.fracture_detector = FractureRetinaHead2D(
            multitask_config.detection,
            skip_channels=backbone.skip_channels,
            latent_channels=backbone.output_channels,
            size_divisible=backbone.reductions[-1],
            box_key=labels_config.keys.box,
            label_key=labels_config.keys.label,
        )

    def forward(self, images: Tensor) -> MultiTaskOutput:
        features = self.backbone(images)

        hemorrhage = self.hemorrhage_decoder(images, features)
        heatmaps = self.keypoint_decoder(images, features)
        fracture = self.fracture_detector(features)

        return MultiTaskOutput(
            hemorrhage=hemorrhage,
            heatmaps=heatmaps,
            fracture=fracture,
        )
