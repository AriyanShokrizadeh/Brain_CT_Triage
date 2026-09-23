"""Shared type aliases."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Hashable, Literal

import numpy as np
from torch import Tensor

# Core

Sample = dict[Hashable, Any]
Batch = Mapping[str, Any]
CollateFn = Callable[[list[Sample]], Any]

# Configs

Stage = Literal["train", "val"]
Task = Literal["hemorrhage", "fracture", "keypoint"]

CheckpointMode = Literal["min", "max"]
BatchSizeFinderMode = Literal["power", "binsearch"]
CacheBackend = Literal["persistent", "memory", "none"]
LogLevel = Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]

# Detection

DetTarget = dict[str, Tensor]
DetBatch = list[DetTarget]
DetOutput = dict[str, list[Tensor]]

# Geometry

Point = tuple[float, float]
Box = tuple[float, float, float, float]


# Metrics

MetricValue = Tensor | float
IoUFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
CocoMatch = dict[int, dict[str, np.ndarray]]
