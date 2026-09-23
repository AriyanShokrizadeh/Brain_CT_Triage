"""Task-specific batch collation."""

from typing import cast

from monai.data.utils import list_data_collate
from torch import Tensor

from src.configs.schemas.data.labels import KeysConfig
from src.data.batches import MultiTaskBatch
from src.utils.types import Sample


def multitask_collate(
    batch: list[Sample],
    *,
    keys: KeysConfig,
) -> MultiTaskBatch:
    """Collate dense targets and variable-length detection targets."""
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    detection_targets = [
        {
            keys.box: cast(Tensor, sample[keys.box]).float(),
            keys.label: cast(Tensor, sample[keys.label]).long(),
        }
        for sample in batch
    ]

    dense = cast(
        dict[str, Tensor],
        list_data_collate(
            [
                {
                    keys.image: sample[keys.image],
                    keys.hemorrhage_mask: sample[keys.hemorrhage_mask],
                    keys.keypoint_heatmap: sample[keys.keypoint_heatmap],
                }
                for sample in batch
            ]
        ),
    )

    images = dense[keys.image].float()
    masks = dense[keys.hemorrhage_mask].long()
    heatmaps = dense[keys.keypoint_heatmap].float()

    if masks.ndim != 4 or masks.shape[1] != 1:
        raise ValueError(
            "Expected hemorrhage masks with shape [B, 1, H, W], "
            f"got {tuple(masks.shape)}."
        )

    return MultiTaskBatch(
        images=images,
        masks=masks,
        heatmaps=heatmaps,
        detection_targets=detection_targets,
        keypoint_active=bool(heatmaps.gt(0).any().item()),
    )
