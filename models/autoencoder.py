from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import ImageDecoder
from .encoder import ImageEncoder


@dataclass
class AutoEncoderOutput:
    reconstruction: torch.Tensor
    latent: torch.Tensor
    quantized_latent: torch.Tensor


class AutoEncoder(nn.Module):
    """Quantized convolutional autoencoder used by the AE-DLRT pipeline."""

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 16,
        base_channels: int = 128,
        latent_quant_step: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.latent_quant_step = latent_quant_step
        self.encoder = ImageEncoder(in_channels, latent_channels, base_channels)
        self.decoder = ImageDecoder(in_channels, latent_channels, base_channels)

    def quantize_latent(
        self,
        latent: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> torch.Tensor:
        step = self.latent_quant_step
        if self.training and not deterministic:
            noise = torch.empty_like(latent).uniform_(-0.5 * step, 0.5 * step)
            return latent + noise
        return torch.round(latent / step) * step

    def encode(self, image: torch.Tensor, *, deterministic: bool = False) -> torch.Tensor:
        return self.quantize_latent(
            self.encoder(image),
            deterministic=deterministic,
        )

    def decode(
        self,
        quantized_latent: torch.Tensor,
        *,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        reconstruction = self.decoder(quantized_latent)
        if output_size is not None and reconstruction.shape[-2:] != output_size:
            reconstruction = F.interpolate(
                reconstruction,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        return reconstruction

    def forward(
        self,
        image: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> AutoEncoderOutput:
        latent = self.encoder(image)
        quantized_latent = self.quantize_latent(
            latent,
            deterministic=deterministic,
        )
        reconstruction = self.decode(
            quantized_latent,
            output_size=image.shape[-2:],
        )
        return AutoEncoderOutput(
            reconstruction=reconstruction,
            latent=latent,
            quantized_latent=quantized_latent,
        )
