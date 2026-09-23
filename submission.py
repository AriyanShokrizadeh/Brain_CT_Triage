"""Official CLI entry point for Brain CT Triage inference."""

from pathlib import Path

import click
import pandas as pd

from model import BrainCTTriageModel
from src.data.series import discover_series
from src.utils.constants import SUBMISSION_COLUMNS
from src.utils.io import write_csv


def predict_dataset(
    data_dir: str | Path,
) -> pd.DataFrame:
    """Run inference for every CT series."""
    model = BrainCTTriageModel()
    rows: list[dict[str, str | float]] = []

    for series_dir in discover_series(data_dir):
        prediction = model.predict(series_dir)

        row: dict[str, str | float] = {
            "series_id": series_dir.name,
            **prediction,
        }
        rows.append(row)

    return pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)


@click.command()
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(
        exists=True,
        file_okay=False,
        path_type=Path,
    ),
    help="Directory containing one subdirectory per CT series.",
)
@click.option(
    "--predictions-file-path",
    required=True,
    type=click.Path(
        dir_okay=False,
        path_type=Path,
    ),
    help="Path to the output predictions CSV.",
)
def main(
    data_dir: Path,
    predictions_file_path: Path,
) -> None:
    """Run dataset-level inference."""
    predictions = predict_dataset(data_dir)
    write_csv(predictions, predictions_file_path)
    click.echo(f"Saved {len(predictions)} predictions to {predictions_file_path}")


if __name__ == "__main__":
    main()
