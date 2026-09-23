"""Multitask dataset."""

import pandas as pd
from monai.transforms.transform import Transform

from src.configs.schemas import DataModuleConfig, LabelsConfig, PathsConfig
from src.data.datasets.base import BaseDataset
from src.utils.types import Sample


class MultiTaskDataset(BaseDataset):
    """DICOM slices and annotations for multitask training."""

    def __init__(
        self,
        datamodule_config: DataModuleConfig,
        labels_config: LabelsConfig,
        paths_config: PathsConfig,
        *,
        dataframe: pd.DataFrame,
        transform: Transform,
    ) -> None:
        path_column = labels_config.metadata.relative_path
        keys = labels_config.keys

        dicom_dir = paths_config.data.train.dicoms
        annotation_dir = paths_config.data.train.annotations

        samples: list[Sample] = [
            {
                keys.image: dicom_dir / f"{relative_path}.dcm",
                keys.annotations: annotation_dir / f"{relative_path}.json",
            }
            for relative_path in dataframe[path_column]
        ]

        super().__init__(
            datamodule_config,
            paths_config,
            data=samples,
            transform=transform,
            cache_namespace=paths_config.experiments.multitask,
        )
