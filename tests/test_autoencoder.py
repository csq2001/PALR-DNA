from __future__ import annotations

from pathlib import Path

import torch

from core.ae_codec import load_ae
from models import AutoEncoder


def make_model() -> AutoEncoder:
    torch.manual_seed(7)
    return AutoEncoder(
        in_channels=3,
        latent_channels=16,
        base_channels=8,
        latent_quant_step=1.0,
    ).eval()


def test_autoencoder_contains_only_encoder_and_decoder_parameters() -> None:
    keys = set(make_model().state_dict())
    assert keys
    assert all(key.startswith(("encoder.", "decoder.")) for key in keys)
    assert not any("prior" in key or "residual" in key for key in keys)


def test_load_ae_round_trip(
    tmp_path: Path,
) -> None:
    source = make_model()
    checkpoint_path = tmp_path / "AE.pth"
    torch.save(
        {
            "model": source.state_dict(),
            "args": {
                "channels": 3,
                "latent_channels": 16,
                "base_channels": 8,
                "latent_quant_step": 1.0,
            },
        },
        checkpoint_path,
    )

    loaded, _ = load_ae(checkpoint_path, torch.device("cpu"))

    for key, value in source.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])


def test_deterministic_latent_quantization() -> None:
    model = make_model()
    latent = torch.tensor([-1.6, -0.4, 0.4, 1.6])
    expected = torch.tensor([-2.0, 0.0, 0.0, 2.0])
    assert torch.equal(
        model.quantize_latent(latent, deterministic=True),
        expected,
    )
