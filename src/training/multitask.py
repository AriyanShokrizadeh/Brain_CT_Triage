"""Multitask supervised training."""

from typing import NamedTuple

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from monai.losses.dice import GeneralizedDiceFocalLoss
from monai.optimizers.lr_scheduler import WarmupCosineSchedule
from torch import Tensor
from torch.optim import AdamW

from src.configs.schemas import ExperimentConfig, LabelsConfig, TasksConfig
from src.data.batches import MultiTaskBatch
from src.metrics.multitask import MultiTaskValidationMetrics
from src.models.losses.keypoints import KeypointHeatmapLoss
from src.models.losses.uncertainty import KendallTaskWeighting
from src.models.multitask.network import MultiTaskNet
from src.models.multitask.outputs import MultiTaskOutput
from src.utils.types import Stage


class TaskLosses(NamedTuple):
    """Raw losses for the three supervised tasks."""

    hemorrhage: Tensor
    keypoint: Tensor
    fracture: Tensor

    @property
    def total(self) -> Tensor:
        return torch.stack(self).sum()


class MultiTaskLightningModule(LightningModule):
    """Train and validate the supervised multitask model."""

    def __init__(
        self,
        labels_config: LabelsConfig,
        task_config: TasksConfig,
        experiment_config: ExperimentConfig,
        *,
        model: MultiTaskNet,
    ) -> None:
        super().__init__()

        self.model = model
        self.experiment = experiment_config
        self.keys = labels_config.keys

        self.hemorrhage_loss = GeneralizedDiceFocalLoss(
            include_background=False,
            softmax=True,
            to_onehot_y=True,
        )
        self.keypoint_loss = KeypointHeatmapLoss(peak_weight=20.0)
        self.task_weighting = KendallTaskWeighting(num_tasks=3)

        self.val_metrics = MultiTaskValidationMetrics(
            num_hemorrhage_classes=len(task_config.hemorrhage.class_names),
            max_detections=self.model.fracture_detector.max_detections,
            box_key=self.keys.box,
            label_key=self.keys.label,
            score_key=self.model.fracture_detector.score_key,
        )

    def forward(self, images: Tensor) -> MultiTaskOutput:
        """Run the multitask network."""
        return self.model(images)

    def training_step(self, batch: MultiTaskBatch, _batch_idx: int) -> Tensor:
        """Run one training step."""
        return self._step(batch, stage="train")

    def validation_step(self, batch: MultiTaskBatch, _batch_idx: int) -> Tensor:
        """Run one validation step."""
        return self._step(batch, stage="val")

    def _step(self, batch: MultiTaskBatch, *, stage: Stage) -> Tensor:
        output = self(batch.images)
        total_loss, losses, fracture_losses = self._compute_losses(batch, output)

        self._log_losses(
            stage,
            total_loss,
            losses,
            fracture_losses,
            batch_size=batch.images.shape[0],
        )

        if stage == "train":
            self._log_fracture_batch(batch)
        else:
            self.val_metrics.update_hemorrhage(output.hemorrhage, batch.masks)
            self.val_metrics.update_keypoints(output.heatmaps, batch.heatmaps)
            self.val_metrics.update_fractures(
                self.model.fracture_detector.predict(batch.images, output.fracture),
                batch.detection_targets,
            )

        return total_loss

    def _compute_losses(
        self,
        batch: MultiTaskBatch,
        output: MultiTaskOutput,
    ) -> tuple[Tensor, TaskLosses, dict[str, Tensor]]:
        """Compute task losses and their uncertainty-weighted total."""
        fracture_losses = self.model.fracture_detector.loss(
            batch.images,
            output.fracture,
            batch.detection_targets,
        )
        if not fracture_losses:
            raise RuntimeError("Fracture detector returned no loss terms.")

        losses = TaskLosses(
            hemorrhage=self.hemorrhage_loss(output.hemorrhage, batch.masks),
            keypoint=self.keypoint_loss(output.heatmaps, batch.heatmaps),
            fracture=torch.stack(tuple(fracture_losses.values())).sum(),
        )

        total_loss = self.task_weighting(
            losses,
            active=(True, batch.keypoint_active, True),
        )

        return total_loss, losses, fracture_losses

    def _log_losses(
        self,
        stage: Stage,
        total_loss: Tensor,
        losses: TaskLosses,
        fracture_losses: dict[str, Tensor],
        *,
        batch_size: int,
    ) -> None:
        """Log aggregate and per-task losses."""
        self.log_dict(
            {
                f"{stage}/loss": total_loss,
                f"{stage}/raw_loss": losses.total,
                f"{stage}/hemorrhage_loss": losses.hemorrhage,
                f"{stage}/keypoint_loss": losses.keypoint,
                f"{stage}/fracture_loss": losses.fracture,
                **{
                    f"{stage}/fracture_{name}": value
                    for name, value in fracture_losses.items()
                },
            },
            on_step=stage == "train",
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )

    def _log_fracture_batch(self, batch: MultiTaskBatch) -> None:
        """Log fracture-target statistics for the current training batch."""
        box_counts = torch.tensor(
            [target[self.keys.box].shape[0] for target in batch.detection_targets],
            dtype=torch.float32,
            device=batch.images.device,
        )
        positive = box_counts.gt(0).float()

        self.log_dict(
            {
                "train/fracture_positive_fraction": positive.mean(),
                "train/fracture_boxes_per_slice": box_counts.mean(),
            },
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            batch_size=batch.images.shape[0],
        )
        self.log_dict(
            {
                "train/fracture_positive_slices": positive.sum(),
                "train/fracture_gt_boxes": box_counts.sum(),
            },
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            reduce_fx="sum",
        )

    def on_train_epoch_end(self) -> None:
        """Log learned uncertainty task weights."""
        weights = self.task_weighting.weights.detach()
        self.log_dict(
            {
                "train/task_weight_hemorrhage": weights[0],
                "train/task_weight_keypoint": weights[1],
                "train/task_weight_fracture": weights[2],
            },
            sync_dist=True,
        )

    def on_validation_epoch_end(self) -> None:
        """Log and reset validation metrics."""
        try:
            if not self.trainer.sanity_checking:
                self.log_dict(
                    self.val_metrics.compute(),
                    prog_bar=True,
                    sync_dist=True,
                )
        finally:
            self.val_metrics.reset()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure optimizer and epoch-wise warmup cosine decay."""
        config = self.experiment.optimizer

        optimizer = AdamW(
            [
                {
                    "params": self.model.parameters(),
                    "weight_decay": config.weight_decay,
                },
                {
                    "params": self.task_weighting.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            lr=config.learning_rate,
        )

        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=config.warmup_epochs,
            t_total=self.experiment.trainer.max_epochs,
            end_lr=config.min_learning_rate,
            warmup_multiplier=config.min_learning_rate / config.learning_rate,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
