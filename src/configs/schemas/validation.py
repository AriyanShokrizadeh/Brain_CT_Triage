"""Reusable configuration validators."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

type Number = int | float


def check_finite(name: str, value: Number) -> None:
    """Validate a finite numeric value."""
    if not isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}.")


def check_non_empty_string(name: str, value: str) -> None:
    """Validate a non-empty string."""
    if not value.strip():
        raise ValueError(f"{name} cannot be empty.")


def check_choice(
    name: str,
    value: str,
    choices: Sequence[str],
) -> None:
    """Validate that a string belongs to a fixed set of choices."""
    if value not in choices:
        raise ValueError(f"{name} must be one of {tuple(choices)}, got {value!r}.")


def check_probability(name: str, value: Number) -> None:
    """Validate a value in the closed interval [0, 1]."""
    check_finite(name, value)

    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1], got {value}.")


def check_fraction_open_upper(name: str, value: Number) -> None:
    """Validate a value in the half-open interval [0, 1)."""
    check_finite(name, value)

    if not 0 <= value < 1:
        raise ValueError(f"{name} must be in [0, 1), got {value}.")


def check_positive(name: str, value: Number) -> None:
    """Validate a strictly positive finite number."""
    check_finite(name, value)

    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}.")


def check_non_negative(name: str, value: Number) -> None:
    """Validate a non-negative finite number."""
    check_finite(name, value)

    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}.")


def check_range(
    name: str,
    value: Sequence[Number],
    *,
    strictly_positive: bool = False,
) -> None:
    """Validate an ordered two-value numeric range."""
    if len(value) != 2:
        raise ValueError(f"{name} must contain two values, got {value}.")

    low, high = value

    check_finite(f"{name}[0]", low)
    check_finite(f"{name}[1]", high)

    if low > high:
        raise ValueError(f"{name} must satisfy low <= high, got {value}.")

    if strictly_positive:
        if low <= 0:
            raise ValueError(f"{name} must contain positive values, got {value}.")
    elif low < 0:
        raise ValueError(f"{name} must contain non-negative values, got {value}.")


def check_string_sequence(
    name: str,
    values: Sequence[str],
    *,
    unique: bool = True,
) -> None:
    """Validate a non-empty sequence of non-empty strings."""
    if not values:
        raise ValueError(f"{name} cannot be empty.")

    normalized = [value.strip() for value in values]

    if not all(normalized):
        raise ValueError(f"{name} cannot contain empty values.")

    if unique and len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must contain unique values.")
