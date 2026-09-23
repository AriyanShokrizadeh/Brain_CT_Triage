"""DICOM series discovery, ordering, and geometry extraction."""

from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import cast

import pydicom

HEADER_TAGS = [
    "Rows",
    "Columns",
    "PixelSpacing",
    "SliceThickness",
    "ImagePositionPatient",
    "InstanceNumber",
]


@dataclass(frozen=True, slots=True)
class SliceGeometry:
    """Geometry information for one DICOM slice."""

    path: Path
    rows: int
    columns: int
    spacing_y: float
    spacing_x: float
    spacing_z: float


@dataclass(frozen=True, slots=True)
class SliceHeader:
    """DICOM metadata required to process one slice."""

    path: Path
    rows: int
    columns: int
    spacing_y: float
    spacing_x: float
    slice_thickness: float
    position_z: float | None
    instance_number: int | None


def discover_series(data_dir: str | Path) -> list[Path]:
    """Return CT series directories."""
    root = Path(data_dir)

    if not root.is_dir():
        raise NotADirectoryError(f"Data directory not found: {root}")

    series_dirs = sorted(path for path in root.iterdir() if path.is_dir())

    if not series_dirs:
        raise ValueError(f"No CT series found in {root}.")

    return series_dirs


def discover_slices(series_dir: str | Path) -> list[Path]:
    """Return DICOM files from one CT series."""
    root = Path(series_dir)

    if not root.is_dir():
        raise NotADirectoryError(f"Series directory not found: {root}")

    slice_paths = sorted(root.glob("*.dcm"))

    if not slice_paths:
        raise ValueError(f"No DICOM files found in {root}.")

    return slice_paths


def _read_header(path: Path) -> SliceHeader:
    """Read DICOM geometry without loading pixel data."""
    dataset = pydicom.dcmread(
        path,
        stop_before_pixels=True,
        specific_tags=HEADER_TAGS,
    )

    try:
        rows = int(dataset.Rows)
        columns = int(dataset.Columns)

        spacing_y = float(dataset.PixelSpacing[0])
        spacing_x = float(dataset.PixelSpacing[1])

        slice_thickness = float(dataset.SliceThickness)

    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid DICOM geometry: {path}") from exc

    geometry_values = (
        rows,
        columns,
        spacing_y,
        spacing_x,
        slice_thickness,
    )

    if any(value <= 0 for value in geometry_values):
        raise ValueError(f"Invalid DICOM geometry: {path}")

    try:
        position_z = float(dataset.ImagePositionPatient[2])
    except (AttributeError, IndexError, TypeError, ValueError):
        position_z = None

    try:
        instance_number = int(dataset.InstanceNumber)
    except (AttributeError, TypeError, ValueError):
        instance_number = None

    return SliceHeader(
        path=path,
        rows=rows,
        columns=columns,
        spacing_y=spacing_y,
        spacing_x=spacing_x,
        slice_thickness=slice_thickness,
        position_z=position_z,
        instance_number=instance_number,
    )


def _sort_headers(headers: list[SliceHeader]) -> list[SliceHeader]:
    """Sort slices by position, instance number, or filename."""
    has_positions = all(header.position_z is not None for header in headers)

    if has_positions:
        return sorted(
            headers,
            key=lambda header: cast(float, header.position_z),
        )

    has_instance_numbers = all(header.instance_number is not None for header in headers)

    if has_instance_numbers:
        return sorted(
            headers,
            key=lambda header: cast(int, header.instance_number),
        )

    return sorted(
        headers,
        key=lambda header: header.path.name,
    )


def _slice_spacing(headers: list[SliceHeader]) -> float:
    """Return inter-slice spacing in millimeters."""
    positions = [
        header.position_z for header in headers if header.position_z is not None
    ]

    has_all_positions = len(positions) == len(headers)

    if has_all_positions and len(positions) > 1:
        differences = [
            abs(right - left)
            for left, right in zip(positions, positions[1:])
            if abs(right - left) > 1e-6
        ]

        if differences:
            return float(median(differences))

    thicknesses = [header.slice_thickness for header in headers]

    return float(median(thicknesses))


def load_series_info(
    series_dir: str | Path,
) -> tuple[SliceGeometry, ...]:
    """Load ordered geometry information for one CT series."""
    slice_paths = discover_slices(series_dir)

    headers = [_read_header(path) for path in slice_paths]

    headers = _sort_headers(headers)
    spacing_z = _slice_spacing(headers)

    geometries = [
        SliceGeometry(
            path=header.path,
            rows=header.rows,
            columns=header.columns,
            spacing_y=header.spacing_y,
            spacing_x=header.spacing_x,
            spacing_z=spacing_z,
        )
        for header in headers
    ]

    return tuple(geometries)
