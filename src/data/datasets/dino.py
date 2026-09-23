"""DINO dataset."""

import pandas as pd
from monai.transforms.transform import Transform

from src.configs.schemas import DataModuleConfig, LabelsConfig, PathsConfig
from src.data.datasets.base import BaseDataset
from src.utils.types import Sample


class DinoDataset(BaseDataset):
    """DICOM slices for DINO pretraining."""

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
        image_key = labels_config.keys.image
        dicom_dir = paths_config.data.train.dicoms

        samples: list[Sample] = [
            {image_key: dicom_dir / f"{relative_path}.dcm"}
            for relative_path in dataframe[path_column]
        ]

        super().__init__(
            datamodule_config,
            paths_config,
            data=samples,
            transform=transform,
            cache_namespace=paths_config.experiments.dino,
        )
