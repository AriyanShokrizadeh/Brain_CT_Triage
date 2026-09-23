"""Prepare train and prediction metadata."""

import click
from loguru import logger

from src.configs import (
    load_dataset_config,
    load_labels_config,
    load_paths_config,
    load_runtime_config,
)
from src.data.metadata.ingestion import MetadataIngestor
from src.data.metadata.validation import MetadataValidator
from src.utils.experiments import setup_experiment
from src.utils.io import write_csv


@click.command()
@click.option(
    "--validate-data/--no-validate-data",
    default=True,
    show_default=True,
    help="Validate prepared metadata before writing CSV files.",
)
def main(validate_data: bool) -> None:
    """Prepare train and prediction metadata."""
    paths = load_paths_config()
    labels = load_labels_config()
    dataset = load_dataset_config()
    runtime = load_runtime_config()

    setup_experiment(
        name=paths.experiments.metadata,
        paths=paths,
        runtime=runtime,
    )

    logger.info(
        "Preparing metadata | source={}",
        paths.data.source,
    )

    annotated, unannotated, predict = MetadataIngestor(
        dataset_config=dataset,
        labels_config=labels,
        paths_config=paths,
    )()

    outputs = (
        ("annotated", annotated, True, paths.data.train.annotated),
        ("unannotated", unannotated, False, paths.data.train.unannotated),
        ("predict", predict, True, paths.data.predict.annotated),
    )

    for name, metadata, labeled, path in outputs:
        if validate_data:
            metadata = MetadataValidator(
                dataset_config=dataset,
                labels_config=labels,
                metadata=metadata,
                labeled=labeled,
            )()

        output = write_csv(metadata, path)

        logger.info(
            "Saved metadata | split={} | rows={:,} | path={}",
            name,
            len(metadata),
            output,
        )

    logger.success("Metadata preparation finished")


if __name__ == "__main__":
    main()
