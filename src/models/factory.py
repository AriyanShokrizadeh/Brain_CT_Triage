"""Backbone factory utilities."""

from dataclasses import replace
from pathlib import Path

from src.configs.schemas import BackboneConfig
from src.configs.schemas.data.labels import LabelsConfig
from src.configs.schemas.data.tasks import TasksConfig
from src.configs.schemas.models.multitask import MultitaskConfig
from src.models.backbones.transunet import TransUNetBackbone2D
from src.models.multitask.network import MultiTaskNet
from src.utils.checkpoints import load_weights


def load_dino_backbone(
    config: BackboneConfig,
    weights_path: str | Path,
) -> TransUNetBackbone2D:
    """Create a backbone initialized from DINO weights."""
    config = replace(config, cnn=replace(config.cnn, pretrained=False))

    backbone = TransUNetBackbone2D(config)
    return load_weights(backbone, weights_path)


def load_multitask_model(
    backbone_config: BackboneConfig,
    multitask_config: MultitaskConfig,
    tasks_config: TasksConfig,
    labels_config: LabelsConfig,
    weights_path: str | Path,
) -> MultiTaskNet:
    """Create a multitask model and load trained weights."""
    backbone_config = replace(
        backbone_config,
        cnn=replace(backbone_config.cnn, pretrained=False),
    )
    backbone = TransUNetBackbone2D(backbone_config)
    model = MultiTaskNet(
        multitask_config,
        tasks_config,
        labels_config,
        backbone=backbone,
    )

    return load_weights(model, weights_path)
