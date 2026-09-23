from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from src.utils.types import DetOutput


@dataclass(frozen=True, slots=True)
class MultiTaskOutput:
    hemorrhage: Tensor
    heatmaps: Tensor
    fracture: DetOutput
