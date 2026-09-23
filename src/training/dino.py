"""DINO self-supervised training."""

from __future__ import annotations

from copy import deepcopy
from math import cos, pi
from typing import Any, cast

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from loguru import logger
from monai.optimizers.lr_scheduler import WarmupCosineSchedule
from timm.optim._optim_factory import create_optimizer_v2
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.optim import Optimizer

from src.configs.schemas import (
    AugmentationsConfig,
    DinoConfig,
    ExperimentConfig,
    LabelsConfig,
)
from src.models.backbones.transunet import TransUNetBackbone2D
from src.models.dino.head import DinoHead
from src.models.dino.loss import DinoLoss
from src.models.dino.network import DinoMultiCropWrapper
from src.utils.types import Batch

BATCH_NORM_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
)


class DinoLightningModule(LightningModule):
    """Train a DINO student with an EMA teacher."""

    def __init__(
        self,
        augmentation_config: AugmentationsConfig,
        labels_config: LabelsConfig,
        dino_config: DinoConfig,
        experiment_config: ExperimentConfig,
        *,
        backbone: TransUNetBackbone2D,
    ) -> None:
        super().__init__()
        self.config = dino_config
        self.experiment = experiment_config
        self.image_key = labels_config.keys.image

        dino_augmentation = augmentation_config.dino
        self.num_global_views = dino_augmentation.num_global_crops
        self.num_student_views = dino_augmentation.number_of_views

        self.student = DinoMultiCropWrapper(
            backbone,
            DinoHead(dino_config, latent_channels=backbone.output_channels),
        )

        self.teacher = deepcopy(self.student).requires_grad_(False)
        self._configure_teacher_mode()

        self.criterion = DinoLoss(
            dino_config,
            output_dim=dino_config.head.output_dim,
            num_global_views=self.num_global_views,
        )

        self._training_steps: int | None = None
        self._last_diagnostic_step = -1

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        """Compute one DINO training step."""
        views = cast(list[Tensor], batch[self.image_key])

        if len(views) != self.num_student_views:
            raise ValueError(
                f"Expected {self.num_student_views} DINO views, got {len(views)}."
            )

        batch_size = views[0].shape[0]
        if any(view.shape[0] != batch_size for view in views):
            raise ValueError("All DINO views must have the same batch size.")

        student_output = self.student(views)

        with torch.no_grad():
            teacher_output = self.teacher(views[: self.num_global_views])

        loss = self.criterion(
            student_output,
            teacher_output,
            num_student_views=self.num_student_views,
            epoch=self.current_epoch,
        )

        self._log_diagnostics(student_output, teacher_output, batch_size=batch_size)

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure AdamW and step-wise warmup cosine decay."""
        opt_cfg = self.experiment.optimizer
        optimizer = create_optimizer_v2(
            self.student,
            opt="adamw",
            lr=opt_cfg.learning_rate,
            weight_decay=opt_cfg.weight_decay,
        )

        total_steps = self._estimated_training_steps()
        warmup_steps = self._warmup_steps(total_steps)

        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=warmup_steps,
            t_total=total_steps,
            end_lr=opt_cfg.min_learning_rate,
            warmup_multiplier=(opt_cfg.min_learning_rate / opt_cfg.learning_rate),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_train_start(self) -> None:
        """Finalize step-dependent schedules after batch-size tuning."""
        self._last_diagnostic_step = -1
        self._configure_training_schedule()

    def on_train_epoch_start(self) -> None:
        """Restore the intended teacher training mode."""
        self._configure_teacher_mode()

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        """Freeze the final student projection during early epochs."""
        if self.current_epoch >= self.config.head.freeze_last_layer_epochs:
            return

        for parameter in cast(DinoHead, self.student.head).last_layer.parameters():
            parameter.grad = None

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Optimizer | LightningOptimizer,
        optimizer_closure: Any | None = None,
    ) -> None:
        """Perform one student optimizer step and update the EMA teacher."""
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        self._update_teacher()

    @torch.no_grad()
    def _update_teacher(self) -> None:
        """Update teacher parameters from the student using EMA."""
        momentum = self._teacher_momentum()
        for student_param, teacher_param in zip(
            self.student.parameters(),
            self.teacher.parameters(),
            strict=True,
        ):
            teacher_param.lerp_(student_param, 1.0 - momentum)

    def _configure_teacher_mode(self) -> None:
        """Disable teacher stochastic layers while adapting BatchNorm."""
        self.teacher.eval()
        for module in self.teacher.modules():
            if isinstance(module, BATCH_NORM_TYPES):
                module.train()

    def _configure_training_schedule(self) -> None:
        """Synchronize schedules with the final training geometry."""
        total_steps = self._estimated_training_steps()
        warmup_steps = self._warmup_steps(total_steps)
        scheduler = self.lr_schedulers()

        if not isinstance(scheduler, WarmupCosineSchedule):
            raise RuntimeError(
                f"Expected WarmupCosineSchedule, got {type(scheduler).__name__}."
            )

        scheduler.t_total = total_steps
        scheduler.warmup_steps = warmup_steps
        self._training_steps = total_steps

        logger.info(
            "DINO schedule | total_steps={} | warmup_steps={}",
            total_steps,
            warmup_steps,
        )

    def _estimated_training_steps(self) -> int:
        """Return Lightning's estimated number of optimizer steps."""
        if (total_steps := int(self.trainer.estimated_stepping_batches)) <= 0:
            raise RuntimeError("Trainer estimated zero optimization steps.")
        return total_steps

    def _warmup_steps(self, total_steps: int) -> int:
        """Return the number of optimizer steps used for LR warmup."""
        opt_cfg = self.experiment.optimizer
        trn_cfg = self.experiment.trainer
        return min(
            round(total_steps * opt_cfg.warmup_epochs / trn_cfg.max_epochs), total_steps
        )

    @torch.no_grad()
    def _log_diagnostics(
        self,
        student_output: Tensor,
        teacher_output: Tensor,
        *,
        batch_size: int,
    ) -> None:
        """Log statistics useful for detecting DINO collapse."""
        step = self.global_step
        if (
            step == self._last_diagnostic_step
            or step % self.experiment.trainer.log_every_n_steps != 0
        ):
            return

        self._last_diagnostic_step = step

        student = student_output.float().unflatten(
            0, (self.num_student_views, batch_size)
        )
        teacher = teacher_output.float().unflatten(
            0, (self.num_global_views, batch_size)
        )
        temperature = self.criterion.teacher_temperature(self.current_epoch)

        distribution = Categorical(
            logits=(teacher - self.criterion.center) / temperature
        )
        mean_distribution = Categorical(probs=distribution.probs.mean(dim=(0, 1)))

        self.log_dict(
            {
                "train/center_norm": self.criterion.center.norm(),
                "train/teacher_momentum": self._teacher_momentum(),
                "train/student_logit_std": student.std(dim=1, correction=0).mean(),
                "train/teacher_logit_std": teacher.std(dim=1, correction=0).mean(),
                "train/teacher_entropy": distribution.entropy().mean(),
                "train/teacher_mean_entropy": mean_distribution.entropy(),
            },
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            batch_size=batch_size,
        )

    def _teacher_momentum(self) -> float:
        """Return the cosine-scheduled EMA teacher momentum."""
        total_steps = (
            self._training_steps
            if self._training_steps is not None
            else self._estimated_training_steps()
        )
        progress = min(self.global_step / total_steps, 1.0)

        base_momentum = self.config.loss.teacher_momentum
        cosine = 0.5 * (1.0 + cos(pi * progress))

        return 1.0 - (1.0 - base_momentum) * cosine
