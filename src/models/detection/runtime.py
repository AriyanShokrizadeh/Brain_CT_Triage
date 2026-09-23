"""MONAI RetinaNet runtime utilities."""

from collections.abc import Sequence
from typing import cast

import torch
from einops import rearrange
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from torch import Tensor, nn

from src.configs.schemas import DetectionConfig
from src.utils.types import DetBatch, DetOutput


class RetinaNetRuntime2D(nn.Module):
    """Use MONAI RetinaNet utilities with precomputed detector outputs."""

    def __init__(
        self,
        config: DetectionConfig,
        *,
        anchor_shapes: Sequence[Sequence[float]],
        feature_map_scales: Sequence[int],
        size_divisible: int,
        box_key: str,
        label_key: str,
    ) -> None:
        super().__init__()

        if size_divisible <= 0:
            raise ValueError("size_divisible must be positive.")

        self.size_divisible = size_divisible
        self.box_key = box_key
        self.label_key = label_key

        anchor_generator = AnchorGeneratorWithAnchorShape(
            feature_map_scales=feature_map_scales,
            base_anchor_shapes=anchor_shapes,
        )

        self.num_anchors = anchor_generator.num_anchors_per_location()[0]

        self.detector = RetinaNetDetector(
            network=nn.Identity(),
            anchor_generator=anchor_generator,
            spatial_dims=2,
            num_classes=1,
            size_divisible=size_divisible,
            cls_key="classification",
            box_reg_key="box_regression",
        )

        self.detector.set_target_keys(
            box_key=box_key,
            label_key=label_key,
        )

        self.detector.set_atss_matcher(
            num_candidates=config.atss_candidates,
            center_in_gt=False,
        )

        sampler = config.hard_negative
        self.detector.set_hard_negative_sampler(
            batch_size_per_image=sampler.batch_size,
            positive_fraction=sampler.positive_fraction,
            min_neg=sampler.minimum_negative,
            pool_size=sampler.pool_size,
        )

        inference = config.inference
        self.max_detections = inference.max_detections

        self.detector.set_box_selector_parameters(
            score_thresh=inference.score_threshold,
            topk_candidates_per_level=1000,
            nms_thresh=inference.nms_threshold,
            detections_per_img=self.max_detections,
        )

    @property
    def score_key(self) -> str:
        """Return the prediction score key."""
        return self.detector.pred_score_key

    def loss(
        self,
        images: Tensor,
        outputs: DetOutput,
        targets: DetBatch,
    ) -> dict[str, Tensor]:
        """Compute RetinaNet losses."""
        head_outputs, anchors, locations = self._prepare(images, outputs)
        device = images.device

        targets = [
            {
                self.box_key: target[self.box_key].to(device),
                self.label_key: target[self.label_key].to(device),
            }
            for target in targets
        ]

        losses = self.detector.compute_loss(
            head_outputs_reshape=head_outputs,
            targets=targets,
            anchors=anchors,
            num_anchor_locs_per_level=locations,
        )

        return {name: loss.to(device) for name, loss in losses.items()}

    @torch.inference_mode()
    def predict(
        self,
        images: Tensor,
        outputs: DetOutput,
    ) -> DetBatch:
        """Decode RetinaNet detections."""
        head_outputs, anchors, locations = self._prepare(
            images,
            outputs,
        )

        predictions = self.detector.postprocess_detections(
            head_outputs_reshape=head_outputs,
            anchors=anchors,
            image_sizes=[list(images.shape[-2:])] * images.shape[0],
            num_anchor_locs_per_level=locations,
        )

        return cast(DetBatch, predictions)

    def _prepare(
        self,
        images: Tensor,
        outputs: DetOutput,
    ) -> tuple[dict[str, Tensor], list[Tensor], list[int]]:
        """Prepare RetinaNet outputs and anchors for MONAI."""
        image_size = tuple(images.shape[-2:])

        if any(size % self.size_divisible for size in image_size):
            raise ValueError(
                f"Image size {image_size} must be divisible "
                f"by {self.size_divisible}."
            )

        cls_key = self.detector.cls_key
        reg_key = self.detector.box_reg_key

        cls_maps = outputs[cls_key]
        reg_maps = outputs[reg_key]

        cls_features = [
            rearrange(
                feature,
                "b (a c) h w -> b (h w a) c",
                a=self.num_anchors,
            )
            for feature in cls_maps
        ]

        reg_features = [
            rearrange(
                feature,
                "b (a c) h w -> b (h w a) c",
                a=self.num_anchors,
            )
            for feature in reg_maps
        ]

        cls_head = torch.cat(cls_features, dim=1)
        reg_head = torch.cat(reg_features, dim=1)

        head_outputs = {cls_key: cls_head, reg_key: reg_head}
        self.detector.generate_anchors(images, outputs)

        anchors = self.detector.anchors
        if anchors is None:
            raise RuntimeError("RetinaNet anchor generation failed.")

        anchors = [anchor.to(images.device) for anchor in anchors]
        locations = [feature.shape[-2] * feature.shape[-1] for feature in cls_maps]

        return head_outputs, anchors, locations
