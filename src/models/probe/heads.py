from __future__ import annotations

from torch import Tensor, nn

from src.configs.schemas import LabelsConfig
from src.models.probe.outputs import LinearProbeOutput


class LinearProbeHeads(nn.Module):
    """Task-specific linear heads over a frozen backbone representation."""

    def __init__(
        self,
        labels_config: LabelsConfig,
        *,
        latent_channels: int,
    ) -> None:
        super().__init__()

        num_hemorrhage_classes = len(labels_config.metadata.hemorrhage_columns)

        self.hemorrhage = nn.Linear(latent_channels, num_hemorrhage_classes)
        self.fracture = nn.Linear(latent_channels, 1)
        self.midline_shift = nn.Linear(latent_channels, 1)

    def forward(self, features: Tensor) -> LinearProbeOutput:
        return LinearProbeOutput(
            hemorrhage_logits=self.hemorrhage(features),
            fracture_logit=self.fracture(features).squeeze(-1),
            midline_shift=self.midline_shift(features).squeeze(-1),
        )
