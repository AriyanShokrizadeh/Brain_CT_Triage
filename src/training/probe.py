"""Frozen-backbone linear-probe training."""

from __future__ import annotations

from math import isfinite
from typing import cast

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from monai.optimizers.lr_scheduler import WarmupCosineSchedule
from torch import Tensor, nn
from torch.optim import AdamW

from src.configs.schemas import EvaluationConfig, ExperimentConfig, LabelsConfig
from src.metrics.probe import ProbeValidationMetrics
from src.models.backbones.transunet import TransUNetBackbone2D
from src.models.probe.heads import LinearProbeHeads
from src.models.probe.outputs import LinearProbeOutput
from src.utils.types import Batch, Stage


def _validate_mls_stats(mean: float, std: float) -> None:
    """Validate MLS normalization statistics."""
    if not isfinite(mean):
        raise ValueError(f"mls_mean must be finite, got {mean}.")
    if not isfinite(std) or std <= 0:
        raise ValueError(f"mls_std must be finite and > 0, got {std}.")


class LinearProbeLightningModule(LightningModule):
    """Evaluate frozen backbone features with linear probes."""

    mls_mean: Tensor
    mls_std: Tensor

    def __init__(
        self,
        backbone: TransUNetBackbone2D,
        mls_mean: float,
        mls_std: float,
        *,
        labels_config: LabelsConfig,
        evaluation_config: EvaluationConfig,
        experiment_config: ExperimentConfig,
    ) -> None:
        super().__init__()
        _validate_mls_stats(mls_mean, mls_std)

        self.experiment = experiment_config
        self.keys = labels_config.keys

        self.register_buffer("mls_mean", torch.tensor(mls_mean, dtype=torch.float32))
        self.register_buffer("mls_std", torch.tensor(mls_std, dtype=torch.float32))

        self.backbone = backbone.requires_grad_(False)
        self.backbone.eval()

        self.heads = LinearProbeHeads(
            labels_config,
            latent_channels=backbone.output_channels,
        )

        self.hemorrhage_loss = nn.BCEWithLogitsLoss()
        self.fracture_loss = nn.BCEWithLogitsLoss()
        self.mls_loss = nn.MSELoss()

        self.val_metrics = ProbeValidationMetrics(
            evaluation_config,
            num_hemorrhage_labels=len(labels_config.metadata.hemorrhage_columns),
        )

    def forward(self, images: Tensor) -> LinearProbeOutput:
        """Run frozen backbone features through the probe heads."""
        with torch.no_grad():
            features = self.backbone(images).global_feature
        return self.heads(features)

    def training_step(self, batch: Batch, _batch_idx: int) -> Tensor:
        """Run one training step."""
        return self._step(batch, stage="train")

    def validation_step(self, batch: Batch, _batch_idx: int) -> Tensor:
        """Run one validation step."""
        return self._step(batch, stage="val")

    def _step(self, batch: Batch, *, stage: Stage) -> Tensor:
        """Compute probe losses and validation metrics."""
        images = cast(Tensor, batch[self.keys.image])
        hemorrhage_target = cast(Tensor, batch[self.keys.hemorrhage_labels]).float()
        fracture_target = cast(Tensor, batch[self.keys.fracture_prob]).float().flatten()
        mls_target = cast(Tensor, batch[self.keys.MLS_mm]).float().flatten()

        output = self(images)

        hemorrhage_loss = self.hemorrhage_loss(
            output.hemorrhage_logits, hemorrhage_target
        )
        fracture_loss = self.fracture_loss(output.fracture_logit, fracture_target)

        normalized_mls = (mls_target - self.mls_mean) / self.mls_std
        mls_loss = self.mls_loss(output.midline_shift, normalized_mls)

        loss = hemorrhage_loss + fracture_loss + mls_loss

        self._log_losses(
            stage=stage,
            loss=loss,
            hemorrhage_loss=hemorrhage_loss,
            fracture_loss=fracture_loss,
            mls_loss=mls_loss,
            batch_size=images.shape[0],
        )

        if stage == "val":
            self.val_metrics.update(
                output,
                hemorrhage_target=hemorrhage_target,
                fracture_target=fracture_target,
                mls_target=mls_target,
                mls_mean=self.mls_mean,
                mls_std=self.mls_std,
            )

        return loss

    def _log_losses(
        self,
        *,
        stage: Stage,
        loss: Tensor,
        hemorrhage_loss: Tensor,
        fracture_loss: Tensor,
        mls_loss: Tensor,
        batch_size: int,
    ) -> None:
        """Log probe losses."""
        self.log_dict(
            {
                f"{stage}/loss": loss,
                f"{stage}/hemorrhage_loss": hemorrhage_loss,
                f"{stage}/fracture_loss": fracture_loss,
                f"{stage}/mls_loss": mls_loss,
            },
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=stage == "val",
            sync_dist=True,
            batch_size=batch_size,
        )

    def on_train_epoch_start(self) -> None:
        """Keep the frozen backbone in evaluation mode."""
        self.backbone.eval()

    def on_validation_epoch_end(self) -> None:
        """Log and reset validation metrics."""
        try:
            if not self.trainer.sanity_checking:
                self.log_dict(self.val_metrics.compute(), prog_bar=True, sync_dist=True)
        finally:
            self.val_metrics.reset()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure linear-probe optimization."""
        config = self.experiment.optimizer

        optimizer = AdamW(
            self.heads.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=config.warmup_epochs,
            t_total=self.experiment.trainer.max_epochs,
            end_lr=config.min_learning_rate,
            warmup_multiplier=(config.min_learning_rate / config.learning_rate),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
