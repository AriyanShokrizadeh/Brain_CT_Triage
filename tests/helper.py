"""Shared test helpers."""

from __future__ import annotations

from collections.abc import Callable
from random import SystemRandom
from typing import Any

import pytest

from src.utils.types import Batch


def random_batch(
    loader: Any,
    condition: Callable[[Batch], bool] | None = None,
) -> Batch:
    """Return a random real batch, optionally matching a condition."""
    dataset = loader.dataset
    indices = SystemRandom().sample(range(len(dataset)), len(dataset))

    for index in indices:
        batch = loader.collate_fn([dataset[index]])
        if condition is None or condition(batch):
            return batch

    pytest.fail("No matching sample found.")
