"""Prediction results used by the inference pipeline."""

from dataclasses import asdict, dataclass
from math import isfinite

from torch import Tensor


@dataclass(frozen=True, slots=True)
class DecodedBatch:
    """Decoded predictions for one model batch."""

    hemorrhage_labels: Tensor
    keypoints_xy: Tensor
    keypoint_scores: Tensor
    fracture_scores: Tensor


@dataclass(frozen=True, slots=True)
class SeriesPrediction:
    """Series-level intermediate predictions."""

    V_EDH: float
    V_SDH: float
    V_IPH: float
    V_SAH: float
    V_IVH: float
    fracture_prob: float
    MLS_mm: float

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if not isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}.")

            if name == "fracture_prob":
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{name} must be in [0, 1], got {value}.")

            elif value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}.")

    def as_dict(self) -> dict[str, float]:
        """Return prediction values as a dictionary."""
        return asdict(self)
