"""Filesystem, tabular, and lightweight figure I/O utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def prepare_output_path(
    path: str | Path,
) -> Path:
    """Create the parent directory and return the normalized path."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def require_file(
    path: str | Path,
    *,
    description: str = "File",
) -> Path:
    """Require an existing file and return its normalized path."""
    source = Path(path)

    if not source.is_file():
        raise FileNotFoundError(f"{description} not found: {source}")

    return source


def read_csv(
    path: str | Path,
) -> pd.DataFrame:
    """Read a non-empty CSV file."""
    source = require_file(path, description="CSV file")
    frame = pd.read_csv(source)

    if frame.empty:
        raise ValueError(f"Empty CSV file: {source}")

    return frame


def read_pickle(
    path: str | Path,
) -> pd.DataFrame:
    """Read a non-empty DataFrame from a pickle file."""
    source = require_file(path, description="Pickle file")
    frame = pd.read_pickle(source)

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Pickle must contain a pandas DataFrame.")

    if frame.empty:
        raise ValueError(f"Empty metadata file: {source}")

    return frame


def write_csv(
    frame: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Write a DataFrame to CSV and return its path."""
    output = prepare_output_path(path)
    frame.to_csv(output, index=False)
    return output


def write_figure(
    figure: Figure,
    path: str | Path,
    *,
    dpi: int = 150,
) -> Path:
    """Write a Matplotlib figure to disk and return its path."""
    if dpi <= 0:
        raise ValueError(f"dpi must be > 0, got {dpi}.")

    output = prepare_output_path(path)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    return output


def save_figure(
    figure: Figure,
    path: str | Path,
    *,
    dpi: int = 150,
) -> Path:
    """Write and close a Matplotlib figure."""
    import matplotlib.pyplot as plt

    output = write_figure(figure, path, dpi=dpi)
    plt.close(figure)
    return output
