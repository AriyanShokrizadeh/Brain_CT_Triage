"""Validation metrics for frozen-backbone linear probes."""

from torch import Tensor, nn
from torchmetrics import MeanAbsoluteError, MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryF1Score,
    MultilabelAUROC,
    MultilabelF1Score,
)

from src.configs.schemas import EvaluationConfig
from src.models.probe.outputs import LinearProbeOutput


class ProbeValidationMetrics(nn.Module):
    """Accumulate validation metrics for the linear-probe tasks."""

    def __init__(
        self,
        evaluation_config: EvaluationConfig,
        *,
        num_hemorrhage_labels: int,
    ) -> None:
        super().__init__()
        auroc_thresholds = evaluation_config.auroc_thresholds
        f1_threshold = evaluation_config.f1_threshold

        self.hemorrhage = MetricCollection(
            {
                "auroc": MultilabelAUROC(
                    num_labels=num_hemorrhage_labels,
                    average="macro",
                    thresholds=auroc_thresholds,
                ),
                "f1": MultilabelF1Score(
                    num_labels=num_hemorrhage_labels,
                    average="macro",
                    threshold=f1_threshold,
                ),
            },
            prefix="val/hemorrhage_",
        )
        self.fracture = MetricCollection(
            {
                "auroc": BinaryAUROC(thresholds=auroc_thresholds),
                "f1": BinaryF1Score(threshold=f1_threshold),
            },
            prefix="val/fracture_",
        )
        self.mls_mae = MeanAbsoluteError()

    def update(
        self,
        output: LinearProbeOutput,
        *,
        hemorrhage_target: Tensor,
        fracture_target: Tensor,
        mls_target: Tensor,
        mls_mean: Tensor,
        mls_std: Tensor,
    ) -> None:
        """Update probe validation metrics."""
        self.hemorrhage.update(output.hemorrhage_logits, hemorrhage_target.int())
        self.fracture.update(output.fracture_logit, fracture_target.ge(0.5).int())
        self.mls_mae.update(
            output.midline_shift * mls_std + mls_mean,
            mls_target,
        )

    def compute(self) -> dict[str, Tensor]:
        """Compute accumulated probe metrics."""
        return {
            **self.hemorrhage.compute(),
            **self.fracture.compute(),
            "val/mls_mae_mm": self.mls_mae.compute(),
        }

    def reset(self) -> None:
        """Reset accumulated probe state."""
        self.hemorrhage.reset()
        self.fracture.reset()
        self.mls_mae.reset()
