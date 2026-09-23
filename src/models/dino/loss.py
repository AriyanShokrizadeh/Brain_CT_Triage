"""DINO self-distillation loss."""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from src.configs.schemas import DinoConfig


class DinoLoss(nn.Module):
    """Cross-view DINO self-distillation loss."""

    _center: Tensor

    def __init__(
        self,
        dino_config: DinoConfig,
        *,
        output_dim: int,
        num_global_views: int,
    ) -> None:
        super().__init__()

        config = dino_config.loss

        self.num_global_views = num_global_views
        self.student_temperature = config.student_temperature
        self.teacher_temperature_start = config.teacher_temperature_start
        self.teacher_temperature_end = config.teacher_temperature_end
        self.temperature_warmup_epochs = config.temperature_warmup_epochs
        self.center_momentum = config.center_momentum

        self.register_buffer(
            "_center",
            torch.zeros(1, output_dim, dtype=torch.float32),
        )

    @property
    def center(self) -> Tensor:
        """Return the FP32 teacher center."""
        return self._center

    def forward(
        self,
        student_output: Tensor,
        teacher_output: Tensor,
        *,
        num_student_views: int,
        epoch: float,
    ) -> Tensor:
        """Compute cross-view DINO self-distillation loss."""
        batch_size = teacher_output.shape[0] // self.num_global_views

        student = student_output.float().reshape(num_student_views, batch_size, -1)
        teacher = teacher_output.float().reshape(self.num_global_views, batch_size, -1)

        student_log_probs = F.log_softmax(student / self.student_temperature, dim=-1)

        teacher_probs = F.softmax(
            (teacher - self.center) / self.teacher_temperature(epoch),
            dim=-1,
        ).detach()

        losses = [
            -(teacher_probs[teacher_view] * student_log_probs[student_view])
            .sum(dim=-1)
            .mean()
            for teacher_view in range(self.num_global_views)
            for student_view in range(num_student_views)
            if teacher_view != student_view
        ]

        loss = torch.stack(losses).mean()
        self._update_center(teacher_output)

        return loss

    def teacher_temperature(self, epoch: float) -> float:
        """Return the scheduled teacher temperature."""
        warmup = self.temperature_warmup_epochs

        if warmup <= 0 or epoch >= warmup:
            return self.teacher_temperature_end

        progress = epoch / warmup

        return self.teacher_temperature_start + progress * (
            self.teacher_temperature_end - self.teacher_temperature_start
        )

    @torch.inference_mode()
    def _update_center(self, teacher_logits: Tensor) -> None:
        teacher_logits = teacher_logits.float()

        batch_sum = teacher_logits.sum(dim=0, keepdim=True)
        sample_count = torch.tensor(
            teacher_logits.shape[0],
            device=teacher_logits.device,
            dtype=torch.float32,
        )

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_sum)
            dist.all_reduce(sample_count)

        batch_center = batch_sum / sample_count

        self._center.mul_(self.center_momentum).add_(
            batch_center,
            alpha=1.0 - self.center_momentum,
        )
