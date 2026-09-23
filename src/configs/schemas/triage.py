"""Rule-based triage threshold configuration."""

from __future__ import annotations

from dataclasses import dataclass

from src.configs.schemas.validation import (
    check_non_negative,
    check_probability,
)


@dataclass(slots=True)
class TriageConfig:
    """Thresholds used to convert model outputs into triage classes."""

    minimum_ich_volume_ml: float
    fracture_probability_threshold: float

    urgent_mls_mm: float
    emergency_mls_mm: float

    edh_emergency_volume_ml: float
    sdh_emergency_volume_ml: float
    iph_emergency_volume_ml: float
    total_emergency_volume_ml: float

    combined_mls_mm: float
    combined_ich_volume_ml: float
    fracture_ich_volume_ml: float

    def __post_init__(self) -> None:
        check_non_negative("triage.minimum_ich_volume_ml", self.minimum_ich_volume_ml)
        check_probability(
            "triage.fracture_probability_threshold",
            self.fracture_probability_threshold,
        )
        check_non_negative("triage.urgent_mls_mm", self.urgent_mls_mm)
        check_non_negative("triage.emergency_mls_mm", self.emergency_mls_mm)
        check_non_negative(
            "triage.edh_emergency_volume_ml",
            self.edh_emergency_volume_ml,
        )
        check_non_negative(
            "triage.sdh_emergency_volume_ml",
            self.sdh_emergency_volume_ml,
        )
        check_non_negative(
            "triage.iph_emergency_volume_ml",
            self.iph_emergency_volume_ml,
        )
        check_non_negative(
            "triage.total_emergency_volume_ml",
            self.total_emergency_volume_ml,
        )
        check_non_negative("triage.combined_mls_mm", self.combined_mls_mm)
        check_non_negative("triage.combined_ich_volume_ml", self.combined_ich_volume_ml)
        check_non_negative("triage.fracture_ich_volume_ml", self.fracture_ich_volume_ml)

        if self.urgent_mls_mm > self.emergency_mls_mm:
            raise ValueError(
                "triage.urgent_mls_mm must be <= "
                "triage.emergency_mls_mm, "
                f"got {self.urgent_mls_mm} > "
                f"{self.emergency_mls_mm}."
            )

        if self.combined_mls_mm > self.emergency_mls_mm:
            raise ValueError(
                "triage.combined_mls_mm must be <= "
                "triage.emergency_mls_mm, "
                f"got {self.combined_mls_mm} > "
                f"{self.emergency_mls_mm}."
            )
