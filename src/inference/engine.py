"""Batched multitask inference engine."""

import torch
from monai.transforms.transform import Transform
from torch import Tensor
from torch.utils.data import DataLoader

from src.configs.schemas import (
    DataModuleConfig,
    EvaluationConfig,
    LabelsConfig,
    PathsConfig,
    TasksConfig,
)
from src.data.datasets.inference import InferenceDataset
from src.data.series import SliceGeometry
from src.inference.postprocess import SeriesPostprocessor
from src.inference.results import DecodedBatch, SeriesPrediction
from src.models.multitask.network import MultiTaskNet
from src.models.multitask.outputs import MultiTaskOutput


def _decode_keypoints(
    heatmaps: Tensor,
) -> tuple[Tensor, Tensor]:
    """Decode heatmap peaks into coordinates and scores."""
    batch_size, num_keypoints, _, width = heatmaps.shape

    heatmaps = heatmaps.reshape(
        batch_size,
        num_keypoints,
        -1,
    )

    scores, indices = heatmaps.max(dim=-1)

    points = torch.stack(
        (
            indices % width,
            indices // width,
        ),
        dim=-1,
    ).float()

    return points, scores


def _decode_fractures(
    model: MultiTaskNet,
    images: Tensor,
    output: MultiTaskOutput,
) -> Tensor:
    """Return the maximum fracture score for each slice."""
    fracture_output = {
        key: [feature.float() for feature in features]
        for key, features in output.fracture.items()
    }

    detections = model.fracture_detector.predict(
        images.float(),
        fracture_output,
    )

    score_key = model.fracture_detector.score_key
    scores = []

    for detection in detections:
        values = detection[score_key]

        if values.numel():
            score = values.max()
        else:
            score = values.new_zeros(())

        scores.append(score)

    return torch.stack(scores)


def _decode_output(
    model: MultiTaskNet,
    images: Tensor,
    output: MultiTaskOutput,
) -> DecodedBatch:
    """Decode raw multitask predictions."""
    keypoints, keypoint_scores = _decode_keypoints(output.heatmaps)

    fracture_scores = _decode_fractures(
        model,
        images,
        output,
    )

    return DecodedBatch(
        hemorrhage_labels=output.hemorrhage.argmax(dim=1).cpu(),
        keypoints_xy=keypoints.cpu(),
        keypoint_scores=keypoint_scores.cpu(),
        fracture_scores=fracture_scores.float().cpu(),
    )


class InferenceEngine:
    """Run multitask inference over one CT series."""

    def __init__(
        self,
        model: MultiTaskNet,
        transform: Transform,
        device: torch.device,
        *,
        labels_config: LabelsConfig,
        tasks_config: TasksConfig,
        datamodule_config: DataModuleConfig,
        evaluation_config: EvaluationConfig,
        paths_config: PathsConfig,
    ) -> None:
        self.model = model.to(device).eval()
        self.transform = transform
        self.device = device

        self.labels = labels_config
        self.tasks = tasks_config
        self.datamodule = datamodule_config
        self.evaluation = evaluation_config
        self.paths = paths_config

        self.use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    def _build_loader(
        self,
        slices: tuple[SliceGeometry, ...],
    ) -> DataLoader:
        """Build the DataLoader for one CT series."""
        dataset = InferenceDataset(
            self.datamodule,
            self.labels,
            self.paths,
            slices=slices,
            transform=self.transform,
        )

        config = self.datamodule.dataloader

        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=(config.persistent_workers and config.num_workers > 0),
            shuffle=False,
            drop_last=False,
        )

    @torch.inference_mode()
    def predict(
        self,
        slices: tuple[SliceGeometry, ...],
    ) -> SeriesPrediction:
        """Predict one complete CT series."""
        loader = self._build_loader(slices)

        postprocessor = SeriesPostprocessor(
            slices,
            self.tasks,
            self.evaluation,
        )

        image_key = self.labels.keys.image
        non_blocking = self.datamodule.dataloader.pin_memory

        for batch in loader:
            images = batch[image_key].to(
                self.device,
                dtype=torch.float32,
                non_blocking=non_blocking,
            )

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.use_bf16,
            ):
                output = self.model(images)

            predictions = _decode_output(
                self.model,
                images,
                output,
            )

            postprocessor.update(
                predictions,
                batch["slice_index"],
            )

        return postprocessor.finalize()
