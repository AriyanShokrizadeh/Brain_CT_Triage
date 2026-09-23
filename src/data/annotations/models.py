"""Schemas for one slice annotation JSON file."""

from pathlib import Path
from typing import Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)

from src.utils.types import Box, Point


class AnnotationSchema(BaseModel):
    """Base schema for annotation data."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        str_strip_whitespace=True,
    )


class SegmentationRLE(AnnotationSchema):
    """Run-length encoded segmentation mask."""

    counts: list[NonNegativeInt] = Field(default_factory=list)
    shape: tuple[PositiveInt, PositiveInt]

    @field_validator("counts")
    @classmethod
    def validate_counts(cls, counts: list[int]) -> list[int]:
        """Validate RLE value-length pairs."""
        if len(counts) % 2:
            raise ValueError("RLE counts must contain value-length pairs.")

        lengths = counts[1::2]

        if any(length == 0 for length in lengths):
            raise ValueError("RLE run lengths must be positive.")

        return counts

    @model_validator(mode="after")
    def validate_size(self) -> Self:
        """Ensure the RLE matches the declared mask shape."""
        if not self.counts:
            return self

        height, width = self.shape

        encoded_pixels = sum(self.counts[1::2])
        expected_pixels = height * width

        if encoded_pixels != expected_pixels:
            raise ValueError("RLE size does not match mask shape.")

        return self


class ClassMapEntry(AnnotationSchema):
    """Segmentation class definition."""

    value: NonNegativeInt
    name: str = Field(min_length=1)


class Annotation(AnnotationSchema):
    """Validated annotations for one CT slice."""

    segmentation_rle: SegmentationRLE
    class_map: list[ClassMapEntry] = Field(min_length=1)
    keypoints: dict[str, Point | None] = Field(default_factory=dict)
    boxes_xywh: list[Box] = Field(default_factory=list)

    @field_validator("boxes_xywh", mode="before")
    @classmethod
    def normalize_boxes(cls, boxes: Any) -> Any:
        """Normalize a single box to a list of boxes."""
        if boxes is None:
            return []

        if (
            isinstance(boxes, (list, tuple))
            and boxes
            and not isinstance(boxes[0], (list, tuple))
        ):

            return [boxes]

        return boxes

    @field_validator("class_map")
    @classmethod
    def validate_class_map(
        cls,
        entries: list[ClassMapEntry],
    ) -> list[ClassMapEntry]:
        """Ensure class values and names are unique."""
        values = [entry.value for entry in entries]
        names = [entry.name for entry in entries]

        if len(values) != len(set(values)):
            raise ValueError("Class-map values must be unique.")

        if len(names) != len(set(names)):
            raise ValueError("Class-map names must be unique.")

        return entries

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        """Ensure boxes and keypoints lie inside the image."""
        height, width = self.segmentation_rle.shape

        for x, y, box_width, box_height in self.boxes_xywh:
            if x < 0 or y < 0:
                raise ValueError("Bounding box coordinates must be non-negative.")

            if box_width <= 0 or box_height <= 0:
                raise ValueError("Bounding box width and height must be positive.")

            right = x + box_width
            bottom = y + box_height

            if right > width or bottom > height:
                raise ValueError("Bounding box exceeds image bounds.")

        for name, point in self.keypoints.items():
            if point is None:
                continue

            x, y = point
            inside_image = 0 <= x < width and 0 <= y < height

            if not inside_image:
                raise ValueError(f"Keypoint {name!r} exceeds image bounds.")

        return self


def load_annotation(path: str | Path) -> Annotation:
    """Load and validate an annotation JSON file."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    return Annotation.model_validate_json(path.read_bytes())
