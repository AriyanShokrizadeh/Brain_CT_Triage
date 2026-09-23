"""Learnable uncertainty weighting for multitask losses."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class KendallTaskWeighting(nn.Module):
    """Combine active task losses using learnable uncertainty."""

    def __init__(self, num_tasks: int) -> None:
        super().__init__()

        if num_tasks < 1:
            raise ValueError("num_tasks must be positive.")

        self.log_variances = nn.Parameter(torch.zeros(num_tasks))

    @property
    def weights(self) -> Tensor:
        return torch.exp(-self.log_variances)

    def forward(
        self,
        losses: Sequence[Tensor],
        active: Sequence[bool],
    ) -> Tensor:
        """Return the uncertainty-weighted sum of active losses."""
        if len(losses) != len(active) or len(losses) != len(self.log_variances):
            raise ValueError("losses and active must match the number of tasks.")

        values = torch.stack([loss.mean() for loss in losses])
        mask = torch.as_tensor(active, device=values.device, dtype=torch.bool)

        log_variances = self.log_variances[mask]
        values = values[mask]

        return (torch.exp(-log_variances) * values + log_variances).sum()
