from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from models import AutoEncoder
from utils.dataset import PortraitImageDataset


AE_METADATA_KEYS = {
    "channels",
    "latent_channels",
    "latent_quant_step",
    "base_channels",
}


def filtered_ae_args(ae_args: dict) -> dict:
    return {key: ae_args[key] for key in sorted(AE_METADATA_KEYS) if key in ae_args}


def load_ae(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[AutoEncoder, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint.get("args", {})
    checkpoint_state = checkpoint["model"]
    channels = int(
        args.get(
            "channels",
            checkpoint_state.get("encoder.net.0.weight", torch.empty(0, 3)).shape[1],
        )
    )
    latent_channels = int(args.get("latent_channels", 16))
    model = AutoEncoder(
        in_channels=channels,
        latent_channels=latent_channels,
        base_channels=int(args.get("base_channels", 128)),
        latent_quant_step=float(args.get("latent_quant_step", 1.0)),
    ).to(device)
    try:
        model.load_state_dict(checkpoint_state)
    except RuntimeError as exc:
        raise ValueError(f"invalid AE checkpoint {checkpoint_path}: {exc}") from exc
    model.eval()
    if model.latent_channels != 16:
        raise ValueError(
            f"AE checkpoint latent_channels={model.latent_channels}, expected 16"
        )
    return model, args


def image_to_tensor(path: Path, channels: int) -> torch.Tensor:
    image = Image.open(path).convert("L" if channels == 1 else "RGB")
    dataset = PortraitImageDataset(
        path.parent,
        patch_size=None,
        training=False,
        channels=channels,
    )
    return dataset._to_tensor(image).unsqueeze(0)


def pad_to_multiple_of_8(image: torch.Tensor) -> torch.Tensor:
    height, width = image.shape[-2:]
    padded_height = (height + 7) // 8 * 8
    padded_width = (width + 7) // 8 * 8
    return F.pad(image, (0, padded_width - width, 0, padded_height - height))


def crop_like(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return tensor[..., : reference.shape[-2], : reference.shape[-1]]


@torch.no_grad()
def encode_image_to_latent(
    ae: AutoEncoder,
    image_path: Path,
    device: torch.device,
) -> torch.Tensor:
    image = image_to_tensor(image_path, ae.in_channels).to(device)
    padded = pad_to_multiple_of_8(image)
    latent = ae.encode(padded, deterministic=True)
    return (
        torch.round(latent[0])
        .clamp(-128, 127)
        .to(torch.int8)
        .cpu()
        .contiguous()
    )


@torch.no_grad()
def decode_latent_to_image(
    ae: AutoEncoder,
    latent: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    latent_tensor = latent.unsqueeze(0).to(device, dtype=torch.float32)
    return ae.decode(latent_tensor).clamp(0.0, 1.0)


def save_tensor_png(tensor: torch.Tensor, path: Path) -> None:
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)[0]
    if tensor.shape[0] == 1:
        array = (tensor.squeeze(0).numpy() * 255.0).round().astype("uint8")
        image = Image.fromarray(array, mode="L")
    else:
        array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
        image = Image.fromarray(array, mode="RGB")
    image.save(path)
