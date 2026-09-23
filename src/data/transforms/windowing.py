"""CT intensity-windowing transforms."""

from __future__ import annotations

from typing import cast

from monai.transforms.compose import Compose
from monai.transforms.intensity.dictionary import ScaleIntensityRanged
from monai.transforms.transform import MapTransform
from monai.transforms.utility.dictionary import ConcatItemsd, CopyItemsd, DeleteItemsd

from src.configs.schemas import DatasetConfig, LabelsConfig
from src.utils.types import Sample


class CTWindowd(MapTransform):
    """Convert one HU image into brain, blood, and bone channels."""

    _PREFIX = "__ct_window_"

    def __init__(
        self,
        dataset_config: DatasetConfig,
        labels_config: LabelsConfig,
    ) -> None:
        image_key = labels_config.keys.image
        super().__init__(image_key)

        windows = dataset_config.dicom.ct_windows

        keys = {
            f"{self._PREFIX}brain": windows.brain,
            f"{self._PREFIX}blood": windows.blood,
            f"{self._PREFIX}bone": windows.bone,
        }

        temp_keys = tuple(keys)

        self.transform = Compose(
            [
                CopyItemsd(
                    keys=image_key,
                    times=len(temp_keys),
                    names=temp_keys,
                ),
                *[
                    ScaleIntensityRanged(
                        keys=key,
                        a_min=window.lower,
                        a_max=window.upper,
                        b_min=0.0,
                        b_max=1.0,
                        clip=True,
                    )
                    for key, window in keys.items()
                ],
                ConcatItemsd(
                    keys=temp_keys,
                    name=image_key,
                    dim=0,
                ),
                DeleteItemsd(keys=temp_keys),
            ]
        )

    def __call__(self, data: Sample) -> Sample:
        return cast(Sample, self.transform(data))
