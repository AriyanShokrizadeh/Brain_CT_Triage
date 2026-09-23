"""High-level MONAI transform pipelines for each training task."""

from __future__ import annotations

from monai.transforms.compose import Compose
from monai.transforms.spatial.dictionary import Resized
from monai.transforms.utility.dictionary import DeleteItemsd, EnsureTyped

from src.configs.schemas import (
    AugmentationsConfig,
    DatasetConfig,
    LabelsConfig,
    TasksConfig,
)
from src.data.transforms.augmentations import (
    build_detection_geometric_transforms,
    build_geometric_transforms,
    build_intensity_transforms,
)
from src.data.transforms.dino import DinoMultiCropd
from src.data.transforms.loading import LoadAnnotationTargetsd, LoadDicomd
from src.data.transforms.resizing import ResizeBoxd


def build_dino_pipeline(
    augmentation_config: AugmentationsConfig,
    dataset_config: DatasetConfig,
    labels_config: LabelsConfig,
) -> Compose:
    """Build deterministic CT preprocessing followed by random DINO multi-crop."""
    image_key = labels_config.keys.image
    image_size = dataset_config.dicom.image_size

    return Compose(
        [
            LoadDicomd(
                dataset_config,
                labels_config,
            ),
            EnsureTyped(
                keys=image_key,
                track_meta=False,
            ),
            Resized(
                keys=image_key,
                spatial_size=image_size,
                mode="bilinear",
            ),
            DinoMultiCropd(
                augmentation_config.dino,
                image_key=image_key,
            ),
        ]
    )


def build_probe_pipeline(
    augmentation_config: AugmentationsConfig,
    dataset_config: DatasetConfig,
    labels_config: LabelsConfig,
    *,
    training: bool,
) -> Compose:
    """Build linear-probe preprocessing and optional training augmentation."""
    keys = labels_config.keys
    image_size = dataset_config.dicom.image_size

    transforms = [
        LoadDicomd(
            dataset_config,
            labels_config,
        ),
        EnsureTyped(
            keys=(
                keys.image,
                keys.hemorrhage_labels,
                keys.fracture_prob,
                keys.MLS_mm,
            ),
            track_meta=False,
        ),
        Resized(
            keys=keys.image,
            spatial_size=image_size,
            mode="bilinear",
        ),
    ]

    if training:
        transforms.extend(
            [
                *build_geometric_transforms(
                    augmentation_config,
                    labels_config,
                ),
                *build_intensity_transforms(
                    augmentation_config,
                    labels_config,
                ),
            ]
        )

    return Compose(transforms)


def build_multitask_pipeline(
    augmentation_config: AugmentationsConfig,
    dataset_config: DatasetConfig,
    labels_config: LabelsConfig,
    tasks_config: TasksConfig,
    *,
    training: bool,
) -> Compose:
    """Build deterministic targets and synchronized augmentation."""
    keys = labels_config.keys
    image_size = dataset_config.dicom.image_size
    zoom_mode = augmentation_config.detection.zoom.mode

    transforms = [
        LoadDicomd(
            dataset_config,
            labels_config,
        ),
        LoadAnnotationTargetsd(
            labels_config,
            tasks_config,
            annotation_key=keys.annotations,
        ),
        DeleteItemsd(
            keys=keys.annotations,
        ),
        EnsureTyped(
            keys=(
                keys.image,
                keys.hemorrhage_mask,
                keys.keypoint_heatmap,
                keys.box,
                keys.label,
            ),
            track_meta=False,
        ),
        ResizeBoxd(
            box_key=keys.box,
            image_key=keys.image,
            spatial_size=image_size,
        ),
        Resized(
            keys=(
                keys.image,
                keys.hemorrhage_mask,
                keys.keypoint_heatmap,
            ),
            spatial_size=image_size,
            mode=(zoom_mode, "nearest", zoom_mode),
        ),
    ]

    if training:
        transforms.extend(
            [
                *build_detection_geometric_transforms(
                    augmentation_config,
                    labels_config,
                    image_keys=(
                        keys.image,
                        keys.hemorrhage_mask,
                        keys.keypoint_heatmap,
                    ),
                    reference_image_key=keys.image,
                    zoom_modes=(zoom_mode, "nearest", zoom_mode),
                ),
                *build_intensity_transforms(
                    augmentation_config,
                    labels_config,
                ),
                DeleteItemsd(
                    keys=f"{keys.box}_transforms",
                ),
            ]
        )

    return Compose(transforms)


def build_inference_pipeline(
    dataset_config: DatasetConfig,
    labels_config: LabelsConfig,
) -> Compose:
    """Match the deterministic image branch of multitask preprocessing."""
    image_key = labels_config.keys.image

    return Compose(
        [
            LoadDicomd(
                dataset_config,
                labels_config,
            ),
            EnsureTyped(
                keys=image_key,
                track_meta=False,
            ),
            Resized(
                keys=image_key,
                spatial_size=dataset_config.dicom.image_size,
                mode="bilinear",
            ),
        ]
    )
