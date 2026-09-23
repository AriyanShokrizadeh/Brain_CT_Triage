"""Segmentation-mask decoding."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from src.data.annotations.models import ClassMapEntry, SegmentationRLE


def _build_class_mapping(
    class_map: Sequence[ClassMapEntry],
    class_names: Sequence[str],
) -> dict[int, int]:
    """Map annotation class values to model class indices."""
    name_to_index = {name: index for index, name in enumerate(class_names)}
    annotation_names = {entry.name for entry in class_map}
    configured_names = set(class_names)

    if annotation_names != configured_names:
        missing = configured_names - annotation_names
        unexpected = annotation_names - configured_names

        raise ValueError(
            "Annotation classes do not match configured classes. "
            f"Missing: {sorted(missing)}. "
            f"Unexpected: {sorted(unexpected)}."
        )

    return {entry.value: name_to_index[entry.name] for entry in class_map}


def decode_rle_mask(
    rle: SegmentationRLE,
    *,
    class_map: Sequence[ClassMapEntry],
    class_names: Sequence[str],
) -> NDArray[np.int64]:
    """Decode an RLE mask into integer class indices."""
    height, width = rle.shape
    mapping = _build_class_mapping(class_map, class_names)

    if not rle.counts:
        return np.zeros((height, width), dtype=np.int64)

    runs = np.asarray(rle.counts, dtype=np.int64).reshape(-1, 2)

    values = runs[:, 0]
    lengths = runs[:, 1]

    unknown = set(values) - set(mapping)
    if unknown:
        raise ValueError(f"RLE contains unmapped class values: {sorted(unknown)}.")

    mapped = np.asarray(
        [mapping[int(value)] for value in values],
        dtype=np.int64,
    )

    return np.repeat(mapped, lengths).reshape(height, width)
