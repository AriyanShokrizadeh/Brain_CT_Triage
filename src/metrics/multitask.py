"""Validation metrics for supervised multitask learning."""

from typing import cast

import torch
from einops import rearrange
from monai.metrics.meandice import DiceMetric
from torch import Tensor, nn
from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryF1Score
from torchmetrics.detection import MeanAveragePrecision

from src.utils.types import DetBatch, MetricValue


class MultiTaskValidationMetrics(nn.Module):
    """Accumulate validation metrics for supervised multitask learning."""

    def __init__(
        self,
        *,
        num_hemorrhage_classes: int,
        max_detections: int,
        box_key: str,
        label_key: str,
        score_key: str,
    ) -> None:
        super().__init__()

        if max_detections <= 10:
            raise ValueError("max_detections must be greater than 10.")

        self.box_key = box_key
        self.label_key = label_key
        self.score_key = score_key
        self.max_detections = max_detections

        self.hemorrhage_dice = DiceMetric(
            num_classes=num_hemorrhage_classes,
            include_background=False,
            reduction="mean",
            ignore_empty=True,
        )

        self.keypoint_error = MeanMetric()
        self.fracture_presence_f1 = BinaryF1Score()

        self.fracture_map = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            max_detection_thresholds=[1, 10, max_detections],
            class_metrics=False,
        )

    def update_hemorrhage(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> None:
        """Update hemorrhage Dice."""
        predictions = logits.argmax(
            dim=1,
            keepdim=True,
        )

        self.hemorrhage_dice(y_pred=predictions, y=targets)

    def update_keypoints(
        self,
        predictions: Tensor,
        targets: Tensor,
    ) -> None:
        """Update mean keypoint localization error."""
        if predictions.shape != targets.shape:
            raise ValueError(
                "Keypoint predictions and targets must have the same shape, "
                f"got {tuple(predictions.shape)} and {tuple(targets.shape)}."
            )

        prediction_flat = rearrange(
            predictions,
            "b k h w -> b k (h w)",
        )
        target_flat = rearrange(
            targets,
            "b k h w -> b k (h w)",
        )

        valid = target_flat.amax(dim=-1).gt(0)

        if not valid.any():
            return

        prediction_index = prediction_flat.argmax(dim=-1)
        target_index = target_flat.argmax(dim=-1)

        height, width = targets.shape[-2:]

        pred_y, pred_x = torch.unravel_index(prediction_index, (height, width))
        target_y, target_x = torch.unravel_index(target_index, (height, width))

        prediction_points = torch.stack((pred_x, pred_y), dim=-1).float()
        target_points = torch.stack((target_x, target_y), dim=-1).float()

        errors = torch.linalg.vector_norm(prediction_points - target_points, dim=-1)

        self.keypoint_error.update(errors[valid])

    def update_fractures(
        self,
        predictions: DetBatch,
        targets: DetBatch,
    ) -> None:
        """Update fracture classification and detection metrics."""
        if len(predictions) != len(targets):
            raise ValueError("Prediction and target batch sizes must match.")

        if not predictions:
            return

        device = predictions[0][self.box_key].device

        prediction_presence = torch.tensor(
            [sample[self.box_key].shape[0] > 0 for sample in predictions],
            device=device,
            dtype=torch.long,
        )

        target_presence = torch.tensor(
            [sample[self.box_key].shape[0] > 0 for sample in targets],
            device=device,
            dtype=torch.long,
        )

        self.fracture_presence_f1.update(
            prediction_presence,
            target_presence,
        )

        map_predictions = []

        for sample in predictions:
            boxes = self._boxes_to_xyxy(sample[self.box_key])

            map_predictions.append(
                {
                    "boxes": boxes.float(),
                    "scores": sample[self.score_key].float(),
                    "labels": sample[self.label_key].long(),
                }
            )

        map_targets = []

        for sample in targets:
            boxes = self._boxes_to_xyxy(sample[self.box_key])

            map_targets.append(
                {
                    "boxes": boxes.float(),
                    "labels": sample[self.label_key].long(),
                }
            )

        self.fracture_map.update(
            map_predictions,
            map_targets,
        )

    def compute(self) -> dict[str, MetricValue]:
        """Compute accumulated validation metrics."""
        metrics: dict[str, MetricValue] = {
            "val/hemorrhage_dice": cast(
                Tensor,
                self.hemorrhage_dice.aggregate(),
            ),
            "val/fracture_presence_f1": (self.fracture_presence_f1.compute()),
        }

        if self.keypoint_error.update_called:
            keypoint_error = self.keypoint_error.compute()

            if torch.isfinite(keypoint_error):
                metrics["val/keypoint_error_px"] = keypoint_error

        detection = self.fracture_map.compute()
        mar_key = f"mar_{self.max_detections}"

        metrics.update(
            {
                "val/fracture/map": detection["map"],
                "val/fracture/map_50": detection["map_50"],
                "val/fracture/map_75": detection["map_75"],
                f"val/fracture/{mar_key}": detection[mar_key],
            }
        )

        return metrics

    def reset(self) -> None:
        """Reset accumulated validation state."""
        self.hemorrhage_dice.reset()
        self.keypoint_error.reset()
        self.fracture_presence_f1.reset()
        self.fracture_map.reset()

    @staticmethod
    def _boxes_to_xyxy(boxes: Tensor) -> Tensor:
        """Convert MONAI spatial-axis boxes to conventional XYXY."""
        corners = rearrange(
            boxes,
            "n (corner coord) -> n corner coord",
            corner=2,
            coord=2,
        )
        corners = corners.flip(dims=(-1,))

        return rearrange(corners, "n corner coord -> n (corner coord)")
