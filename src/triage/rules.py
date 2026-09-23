from __future__ import annotations

from math import isfinite
from typing import Any, Mapping

from src.configs.schemas.triage import TriageConfig

TRIAGE_REQUIRED_KEYS = frozenset(
    {
        "V_EDH",
        "V_SDH",
        "V_IPH",
        "V_SAH",
        "V_IVH",
        "fracture_prob",
        "MLS_mm",
    }
)


def validate_intermediates(
    intermediates: Mapping[str, Any],
) -> dict[str, float]:
    """Validate intermediate predictions for one CT series."""
    keys = set(intermediates)

    missing = TRIAGE_REQUIRED_KEYS - keys
    extra = keys - TRIAGE_REQUIRED_KEYS

    if missing:
        raise ValueError(
            f"Missing keys: {sorted(missing)}. "
            f"Expected: {sorted(TRIAGE_REQUIRED_KEYS)}."
        )

    if extra:
        raise ValueError(
            f"Unexpected keys: {sorted(extra)}. "
            f"Expected: {sorted(TRIAGE_REQUIRED_KEYS)}."
        )

    values: dict[str, float] = {}

    for key in TRIAGE_REQUIRED_KEYS:
        raw_value = intermediates[key]

        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{key!r} must be convertible to float, "
                f"got {type(raw_value).__name__}."
            ) from exc

        if not isfinite(value):
            raise ValueError(f"{key!r} must be finite, got {value}.")

        values[key] = value

    fracture_prob = values["fracture_prob"]

    if not 0.0 <= fracture_prob <= 1.0:
        raise ValueError("'fracture_prob' must be in [0, 1], " f"got {fracture_prob}.")

    return values


def triage_from_intermediates(
    intermediates: Mapping[str, Any],
    config: TriageConfig,
) -> int:
    """
    Convert intermediate imaging predictions to triage class.

    0 = Non-urgent
    1 = Urgent
    2 = Critical
    """
    values = validate_intermediates(intermediates)

    # Preserve competition behavior: negative physical quantities
    # are interpreted as zero.
    v_edh = max(0.0, values["V_EDH"])
    v_sdh = max(0.0, values["V_SDH"])
    v_iph = max(0.0, values["V_IPH"])
    v_sah = max(0.0, values["V_SAH"])
    v_ivh = max(0.0, values["V_IVH"])
    mls_mm = max(0.0, values["MLS_mm"])

    fracture_prob = values["fracture_prob"]

    total_volume = v_edh + v_sdh + v_iph + v_sah + v_ivh

    has_ich = total_volume >= config.minimum_ich_volume_ml

    has_fracture = fracture_prob >= config.fracture_probability_threshold

    # ------------------------------------------------------------
    # Critical
    # ------------------------------------------------------------

    if mls_mm >= config.emergency_mls_mm and (has_ich or has_fracture):
        return 2

    if v_edh >= config.edh_emergency_volume_ml:
        return 2

    if v_sdh >= config.sdh_emergency_volume_ml:
        return 2

    if v_iph >= config.iph_emergency_volume_ml:
        return 2

    if total_volume >= config.total_emergency_volume_ml:
        return 2

    if (
        has_ich
        and mls_mm >= config.combined_mls_mm
        and total_volume >= config.combined_ich_volume_ml
    ):
        return 2

    if has_fracture and total_volume >= config.fracture_ich_volume_ml:
        return 2

    # ------------------------------------------------------------
    # Urgent
    # ------------------------------------------------------------

    if mls_mm >= config.emergency_mls_mm:
        return 1

    if has_ich:
        return 1

    if config.urgent_mls_mm <= mls_mm < config.emergency_mls_mm:
        return 1

    if has_fracture:
        return 1

    # ------------------------------------------------------------
    # Non-urgent
    # ------------------------------------------------------------

    return 0
