"""Linear-probe dataset."""

import numpy as np
import pandas as pd
from monai.transforms.transform import Transform

from src.configs.schemas import DataModuleConfig, LabelsConfig, PathsConfig
from src.data.datasets.base import BaseDataset
from src.utils.types import Sample


class LinearProbeDataset(BaseDataset):
    """DICOM slices with slice-level targets."""

    def __init__(
        self,
        labels_config: LabelsConfig,
        datamodule_config: DataModuleConfig,
        paths_config: PathsConfig,
        *,
        dataframe: pd.DataFrame,
        transform: Transform,
    ) -> None:
        metadata = labels_config.metadata
        keys = labels_config.keys
        dicom_dir = paths_config.data.train.dicoms

        samples: list[Sample] = [
            {
                keys.image: dicom_dir / f"{row[metadata.relative_path]}.dcm",
                keys.hemorrhage_labels: (
                    row[metadata.hemorrhage_columns].to_numpy(dtype=np.float32)
                ),
                keys.fracture_prob: np.float32(row[metadata.fracture_prob]),
                keys.MLS_mm: np.float32(row[metadata.MLS_mm]),
            }
            for _, row in dataframe.iterrows()
        ]

        super().__init__(
            datamodule_config,
            paths_config,
            data=samples,
            transform=transform,
            cache_namespace=paths_config.experiments.probe,
        )
