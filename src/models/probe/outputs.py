from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, slots=True)
class LinearProbeOutput:
    hemorrhage_logits: Tensor
    fracture_logit: Tensor
    midline_shift: Tensor
