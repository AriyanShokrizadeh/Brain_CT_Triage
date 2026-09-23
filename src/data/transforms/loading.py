"""Deterministic DICOM and annotation loading transforms."""

from __future__ import annotations

import logging
from typing import cast

from monai.data.image_reader import PydicomReader
from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.transform import Transform
from pydicom.config import IGNORE, settings

from src.configs.schemas import DatasetConfig, LabelsConfig, TasksConfig
from src.data.annotations.boxes import build_fracture_targets
from src.data.annotations.keypoints import build_keypoint_heatmap
from src.data.annotations.masks import decode_rle_mask
from src.data.annotations.models import load_annotation
from src.data.transforms.windowing import CTWindowd
from src.utils.types import Sample

settings.reading_validation_mode = IGNORE
logging.getLogger("pydicom").setLevel(logging.ERROR)


class LoadDicomd(Transform):
    """Load a DICOM slice and create CT-window channels."""

    def __init__(
        self,
        dataset_config: DatasetConfig,
        labels_config: LabelsConfig,
    ) -> None:
        image_key = labels_config.keys.image

        self.transform = Compose(
            [
                LoadImaged(
                    keys=image_key,
                    reader=PydicomReader,
                    swap_ij=False,
                    image_only=True,
                    prune_metadata=True,
                    ensure_channel_first=True,
                ),
                CTWindowd(
                    labels_config=labels_config,
                    dataset_config=dataset_config,
                ),
            ]
        )

    def __call__(self, data: Sample) -> Sample:
        return cast(Sample, self.transform(data))


class LoadAnnotationTargetsd(Transform):
    """Load annotation and create multitask targets."""

    def __init__(
        self,
        labels_config: LabelsConfig,
        tasks_config: TasksConfig,
        *,
        annotation_key: str,
    ) -> None:
        self.keys = labels_config.keys
        self.tasks = tasks_config
        self.annotation_key = annotation_key

    def __call__(self, data: Sample) -> Sample:
        result = dict(data)
        image = result[self.keys.image]
        annotation = load_annotation(result[self.annotation_key])
        shape = annotation.segmentation_rle.shape

        if tuple(image.shape[-2:]) != shape:
            raise ValueError(
                f"Image shape {tuple(image.shape[-2:])} "
                f"does not match annotation shape {shape}."
            )

        mask = decode_rle_mask(
            annotation.segmentation_rle,
            class_map=annotation.class_map,
            class_names=self.tasks.hemorrhage.class_names,
        )

        heatmap = build_keypoint_heatmap(
            annotation.keypoints,
            shape,
            tasks_config=self.tasks,
        )

        boxes, labels = build_fracture_targets(
            annotation.boxes_xywh,
            tasks_config=self.tasks,
        )

        result.update(
            {
                self.keys.hemorrhage_mask: mask[None],
                self.keys.keypoint_heatmap: heatmap,
                self.keys.box: boxes,
                self.keys.label: labels,
            }
        )

        return cast(Sample, result)
