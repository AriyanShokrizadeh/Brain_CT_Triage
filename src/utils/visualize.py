"""Visualization utilities for model inputs and targets."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
from torch import Tensor

CT_WINDOW_NAMES = ("Brain", "Blood", "Bone")

KEYPOINT_LABELS = {
    "AnteriorFalxAttachment": "AnteriorFalx",
    "PosteriorFalxAttachment": "PosteriorFalx",
    "OutermostPointOfTheFalx": "Outermost",
}


def numpy(tensor: Tensor) -> np.ndarray:
    """Convert a tensor to a NumPy array."""
    return tensor.detach().cpu().numpy()


def plot_dino_views(
    views: Sequence[Tensor],
    *,
    sample_index: int = 0,
    channel: int = 0,
) -> Figure:
    """Plot DINO augmented views."""
    if not views:
        raise ValueError("DINO views cannot be empty.")

    columns = min(5, len(views))
    rows = int(np.ceil(len(views) / columns))

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3 * columns, 3 * rows),
        squeeze=False,
        layout="constrained",
    )
    axes = axes.ravel()

    for index, (axis, view) in enumerate(zip(axes, views, strict=False), start=1):
        axis.imshow(numpy(view[sample_index, channel]), cmap="gray")
        axis.set_title(f"View {index}")
        axis.axis("off")

    for axis in axes[len(views) :]:
        axis.axis("off")

    return figure


def plot_ct_windows(
    images: Tensor,
    *,
    sample_index: int = 0,
    names: Sequence[str] = CT_WINDOW_NAMES,
) -> Figure:
    """Plot CT window channels."""
    if images.ndim != 4:
        raise ValueError("Images must have shape (B, C, H, W).")

    if images.shape[1] != len(names):
        raise ValueError("Window names must match image channels.")

    figure, axes = plt.subplots(
        1, len(names), figsize=(4 * len(names), 4), squeeze=False, layout="constrained"
    )

    for channel, (axis, name) in enumerate(zip(axes[0], names, strict=True)):
        axis.imshow(numpy(images[sample_index, channel]), cmap="gray")
        axis.set_title(name)
        axis.axis("off")

    return figure


def plot_multitask_targets(
    images: Tensor,
    hemorrhage_mask: Tensor,
    keypoint_heatmaps: Tensor,
    fracture_boxes: Tensor,
    *,
    hemorrhage_names: Sequence[str],
    keypoint_names: Sequence[str],
    sample_index: int = 0,
) -> Figure:
    """Plot multitask supervision targets."""
    image = numpy(images[sample_index])
    mask = numpy(hemorrhage_mask[sample_index, 0])
    heatmaps = numpy(keypoint_heatmaps[sample_index])
    boxes = numpy(fracture_boxes)

    figure, axes = plt.subplots(1, 4, figsize=(16, 4), layout="constrained")

    backgrounds = (image[0], image[1], image[2], image[0])
    titles = ("Brain", "Hemorrhage", f"Fracture ({len(boxes)})", "Keypoints")

    for axis, background, title in zip(axes, backgrounds, titles, strict=True):
        axis.imshow(background, cmap="gray")
        axis.set_title(title)
        axis.axis("off")

    _plot_hemorrhage(axes[1], mask, hemorrhage_names)
    _plot_boxes(axes[2], boxes)

    visible = _plot_keypoints(axes[3], heatmaps, keypoint_names)
    axes[3].set_title(f"Keypoints ({visible}/{len(keypoint_names)})")

    return figure


def _plot_hemorrhage(
    axis: Axes,
    mask: np.ndarray,
    class_names: Sequence[str],
) -> None:
    """Overlay hemorrhage classes."""
    class_ids = np.unique(mask[mask > 0]).astype(int)

    if not class_ids.size:
        axis.set_title("Hemorrhage (absent)")
        return

    if class_ids.max() >= len(class_names):
        raise ValueError("Hemorrhage mask contains an unknown class ID.")

    num_classes = len(class_names) - 1
    cmap = plt.colormaps["tab10"].resampled(num_classes)

    axis.imshow(
        np.ma.masked_equal(mask, 0),
        cmap=cmap,
        vmin=1,
        vmax=num_classes,
        alpha=0.75,
        interpolation="nearest",
    )
    axis.contour(mask > 0, levels=[0.5], linewidths=1)

    axis.legend(
        handles=[
            Patch(facecolor=cmap(class_id - 1), label=class_names[class_id])
            for class_id in class_ids
        ],
        loc="lower left",
        fontsize=8,
    )


def _plot_boxes(axis: Axes, boxes: np.ndarray) -> None:
    """Draw MONAI YXYX fracture bounding boxes."""
    for index, (y1, x1, y2, x2) in enumerate(boxes, start=1):
        axis.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="red",
                linewidth=2,
            )
        )

        axis.text(
            x1,
            y1,
            f"Fracture {index}",
            color="red",
            fontsize=8,
            verticalalignment="bottom",
        )


def _plot_keypoints(
    axis: Axes,
    heatmaps: np.ndarray,
    names: Sequence[str],
) -> int:
    """Draw keypoints from heatmap maxima."""
    visible = 0

    for heatmap, name in zip(heatmaps, names, strict=True):
        if heatmap.max() <= 0:
            continue

        y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)
        axis.scatter(x, y, marker="x", s=60, label=KEYPOINT_LABELS.get(name, name))
        visible += 1

    if visible:
        axis.legend(loc="upper right", fontsize=8)

    return visible
