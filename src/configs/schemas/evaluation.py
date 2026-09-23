"""Post-processing and evaluation configuration schema."""

from dataclasses import dataclass

from src.configs.schemas.validation import (
    check_positive,
    check_probability,
)


@dataclass(slots=True)
class EvaluationConfig:
    """Post-processing and evaluation configuration."""

    mls_top_k: int
    f1_threshold: float
    auroc_thresholds: int

    def __post_init__(self) -> None:
        check_positive("evaluation.mls_top_k", self.mls_top_k)
        check_probability("evaluation.f1_threshold", self.f1_threshold)
        check_positive("evaluation.auroc_thresholds", self.auroc_thresholds)
