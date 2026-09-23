"""Visualization integration tests."""

from pathlib import Path

import pytest
from helper import random_batch

from src.data.datamodules.dino import DinoDataModule
from src.data.datamodules.multitask import MultiTaskDataModule
from src.data.datamodules.probe import LinearProbeDataModule
from src.utils.io import save_figure
from src.utils.types import Batch, Task
from src.utils.visualize import plot_ct_windows, plot_dino_views, plot_multitask_targets


def _save_figure(
    figure,
    path: Path,
) -> None:
    """Save a figure and verify the output."""
    output = save_figure(figure, path)

    assert output.is_file()
    assert output.stat().st_size > 0


def test_dino_views(
    dino_datamodule: DinoDataModule,
    figures_dir: Path,
) -> None:
    """Visualize DINO augmented views."""
    keys = dino_datamodule.labels.keys

    batch = random_batch(dino_datamodule.train_dataloader())

    _save_figure(
        plot_dino_views(batch[keys.image]),
        figures_dir / "dino_views.png",
    )


def test_ct_windows(
    probe_datamodule: LinearProbeDataModule,
    figures_dir: Path,
) -> None:
    """Visualize CT intensity windows."""
    keys = probe_datamodule.labels.keys

    batch = random_batch(probe_datamodule.val_dataloader())

    _save_figure(
        plot_ct_windows(batch[keys.image]),
        figures_dir / "ct_windows.png",
    )


@pytest.mark.parametrize(
    "task",
    ("hemorrhage", "fracture", "keypoint"),
)
def test_multitask_targets(
    multitask_datamodule: MultiTaskDataModule,
    figures_dir: Path,
    task: Task,
) -> None:
    """Visualize multitask targets."""
    keys = multitask_datamodule.labels.keys

    def has_target(batch: Batch) -> bool:
        if task == "hemorrhage":
            return bool(batch[keys.hemorrhage_mask][0].gt(0).any().item())

        if task == "fracture":
            return batch[keys.target][0][keys.box].numel() > 0

        return bool(batch[keys.keypoint_heatmap][0].gt(0).any().item())

    batch = random_batch(
        multitask_datamodule.val_dataloader(),
        condition=has_target,
    )

    detection = batch[keys.target][0]

    figure = plot_multitask_targets(
        batch[keys.image],
        batch[keys.hemorrhage_mask],
        batch[keys.keypoint_heatmap],
        detection[keys.box],
        hemorrhage_names=(multitask_datamodule.tasks.hemorrhage.class_names),
        keypoint_names=(multitask_datamodule.tasks.keypoints.class_names),
    )

    _save_figure(
        figure,
        figures_dir / f"multitask_{task}.png",
    )
