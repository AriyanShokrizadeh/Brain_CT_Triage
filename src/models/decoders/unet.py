from __future__ import annotations

from typing import Sequence

from monai.networks.nets.flexible_unet import SegmentationHead, UNetDecoder
from torch import Tensor, nn

from src.configs.schemas import DecoderConfig
from src.models.backbones.outputs import BackboneOutput


class DenseDecoder2D(nn.Module):
    """MONAI U-Net decoder for one dense prediction task."""

    def __init__(
        self,
        decoder_config: DecoderConfig,
        *,
        input_channels: int,
        skip_channels: Sequence[int],
        bottleneck_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        encoder_channels = (
            input_channels,
            *skip_channels,
            bottleneck_channels,
        )
        decoder_channels = tuple(decoder_config.decoder_channels)
        expected_decoders = len(encoder_channels) - 1

        if len(decoder_channels) != expected_decoders:
            raise ValueError(
                f"Expected {expected_decoders} decoder channel levels, "
                f"got {len(decoder_channels)}."
            )

        self.decoder = UNetDecoder(
            spatial_dims=2,
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            act=decoder_config.activation,
            norm=decoder_config.normalization,
            dropout=decoder_config.dropout,
            bias=False,
            upsample="nontrainable",
            pre_conv="default",
            interp_mode="bilinear",
            align_corners=False,
            is_pad=True,
        )

        self.head = SegmentationHead(
            spatial_dims=2,
            in_channels=decoder_channels[-1],
            out_channels=out_channels,
            kernel_size=3,
            act=None,
        )

        self.skip_connect = len(skip_channels)

    def forward(
        self,
        image: Tensor,
        features: BackboneOutput,
    ) -> Tensor:
        decoded = self.decoder(
            [
                image,
                *features.skip_features,
                features.latent_features,
            ],
            skip_connect=self.skip_connect,
        )

        return self.head(decoded)
