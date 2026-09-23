"""Inference dataset."""

from monai.transforms.transform import Transform

from src.configs.schemas import (
    DataModuleConfig,
    LabelsConfig,
    PathsConfig,
)
from src.data.datasets.base import BaseDataset
from src.data.series import SliceGeometry
from src.utils.types import Sample


class InferenceDataset(BaseDataset):
    """Dataset for one ordered CT series."""

    def __init__(
        self,
        datamodule_config: DataModuleConfig,
        labels_config: LabelsConfig,
        paths_config: PathsConfig,
        *,
        slices: tuple[SliceGeometry, ...],
        transform: Transform,
    ) -> None:
        image_key = labels_config.keys.image

        samples: list[Sample] = [
            {
                image_key: geometry.path,
                "slice_index": index,
            }
            for index, geometry in enumerate(slices)
        ]

        super().__init__(
            datamodule_config,
            paths_config,
            data=samples,
            transform=transform,
            cache_namespace="inference",
        )
