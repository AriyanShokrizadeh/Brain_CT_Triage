"""Series-level post-processing for multitask predictions."""

from math import isfinite
from statistics import median

from torch import Tensor

from src.configs.schemas import EvaluationConfig, TasksConfig
from src.data.series import SliceGeometry
from src.inference.results import DecodedBatch, SeriesPrediction
from src.utils.constants import HEMORRHAGE_OUTPUTS, MLS_KEYPOINTS
from src.utils.geometry import point_to_line_distance, rescaled_spacing


class SeriesPostprocessor:
    """Aggregate slice predictions into series-level predictions."""

    def __init__(
        self,
        slices: tuple[SliceGeometry, ...],
        tasks_config: TasksConfig,
        evaluation_config: EvaluationConfig,
    ) -> None:
        if not slices:
            raise ValueError("At least one slice is required.")

        hemorrhage_names = tasks_config.hemorrhage.class_names
        keypoint_names = tasks_config.keypoints.class_names

        missing_hemorrhage = set(HEMORRHAGE_OUTPUTS) - set(hemorrhage_names)
        if missing_hemorrhage:
            raise ValueError(
                f"Missing hemorrhage classes: {sorted(missing_hemorrhage)}."
            )

        missing_keypoints = set(MLS_KEYPOINTS) - set(keypoint_names)
        if missing_keypoints:
            raise ValueError(f"Missing MLS keypoints: {sorted(missing_keypoints)}.")

        self.slices = slices
        self.mls_top_k = evaluation_config.mls_top_k

        self.mls_indices = tuple(keypoint_names.index(name) for name in MLS_KEYPOINTS)

        self.hemorrhage_classes = {
            index: HEMORRHAGE_OUTPUTS[name]
            for index, name in enumerate(hemorrhage_names)
            if name in HEMORRHAGE_OUTPUTS
        }

        self.volumes = {output_name: 0.0 for output_name in HEMORRHAGE_OUTPUTS.values()}

        self.fracture_prob = 0.0
        self.mls_candidates: list[tuple[float, float]] = []

    def update(
        self,
        predictions: DecodedBatch,
        slice_indices: Tensor,
    ) -> None:
        """Accumulate predictions from one batch."""
        image_shape = (
            predictions.hemorrhage_labels.shape[-2],
            predictions.hemorrhage_labels.shape[-1],
        )

        for batch_index, slice_index in enumerate(slice_indices.tolist()):
            geometry = self.slices[slice_index]

            spacing = rescaled_spacing(
                original_shape=(
                    geometry.rows,
                    geometry.columns,
                ),
                original_spacing=(
                    geometry.spacing_x,
                    geometry.spacing_y,
                ),
                new_shape=image_shape,
            )

            self._update_hemorrhage(
                predictions.hemorrhage_labels[batch_index],
                spacing,
                geometry.spacing_z,
            )

            self._update_fracture(predictions.fracture_scores[batch_index])

            self._update_mls(
                predictions.keypoints_xy[batch_index],
                predictions.keypoint_scores[batch_index],
                spacing,
            )

    def _update_hemorrhage(
        self,
        labels: Tensor,
        spacing: tuple[float, float],
        spacing_z: float,
    ) -> None:
        """Accumulate hemorrhage volume for one slice."""
        spacing_x, spacing_y = spacing

        voxel_volume_ml = spacing_x * spacing_y * spacing_z / 1000.0

        counts = labels.flatten().bincount(minlength=max(self.hemorrhage_classes) + 1)

        for class_index, output_name in self.hemorrhage_classes.items():
            voxel_count = counts[class_index].item()
            self.volumes[output_name] += voxel_count * voxel_volume_ml

    def _update_fracture(
        self,
        score: Tensor,
    ) -> None:
        """Keep the highest fracture probability."""
        probability = float(score.item())

        if not isfinite(probability):
            return

        probability = max(
            0.0,
            min(probability, 1.0),
        )

        self.fracture_prob = max(
            self.fracture_prob,
            probability,
        )

    def _update_mls(
        self,
        points: Tensor,
        scores: Tensor,
        spacing: tuple[float, float],
    ) -> None:
        """Store one slice-level MLS candidate."""
        confidence = min(float(scores[index].item()) for index in self.mls_indices)

        if not isfinite(confidence):
            return

        selected_points = points[list(self.mls_indices)]

        distance = self._compute_mls(
            selected_points,
            spacing,
        )

        if distance is None or not isfinite(distance):
            return

        self.mls_candidates.append((confidence, distance))

    @staticmethod
    def _compute_mls(
        points: Tensor,
        spacing: tuple[float, float],
    ) -> float | None:
        """Compute midline shift in millimeters."""
        spacing_x, spacing_y = spacing

        points_mm = [
            (
                float(point[0]) * spacing_x,
                float(point[1]) * spacing_y,
            )
            for point in points
        ]

        anterior, posterior, outermost = points_mm

        return point_to_line_distance(
            outermost,
            anterior,
            posterior,
        )

    def finalize(self) -> SeriesPrediction:
        """Return final series-level predictions."""
        candidates = sorted(
            self.mls_candidates,
            key=lambda candidate: candidate[0],
            reverse=True,
        )[: self.mls_top_k]

        if candidates:
            distances = [distance for _, distance in candidates]
            mls_mm = float(median(distances))
        else:
            mls_mm = 0.0

        return SeriesPrediction(
            V_EDH=self.volumes["V_EDH"],
            V_SDH=self.volumes["V_SDH"],
            V_IPH=self.volumes["V_IPH"],
            V_SAH=self.volumes["V_SAH"],
            V_IVH=self.volumes["V_IVH"],
            fracture_prob=self.fracture_prob,
            MLS_mm=mls_mm,
        )
