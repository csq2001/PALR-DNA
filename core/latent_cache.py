from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch.utils.data import DataLoader

from core.ae_codec import load_ae, pad_to_multiple_of_8
from models import AutoEncoder
from utils.dataset import PortraitImageDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache quantized AE latent bytes for DLRT training."
    )
    parser.add_argument("--data-root", default=os.getenv("AE_DATA_ROOT", "data"))
    parser.add_argument(
        "--checkpoint",
        default=os.getenv("AE_CHECKPOINT", "outputs/checkpoints/AE.pth"),
    )
    parser.add_argument(
        "--out-dir",
        default=os.getenv("AE_LATENT_CACHE", "outputs/latent_cache"),
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument(
        "--expected-latent-channels",
        type=int,
        default=16,
        help="Expected channel count; use 0 to accept any count.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Maximum images per split; 0 means all images.",
    )
    return parser.parse_args()


def quantize_latents_to_int8(latent: torch.Tensor) -> torch.Tensor:
    return torch.round(latent).clamp(-128, 127).to(torch.int8).cpu()


@torch.no_grad()
def cache_split(
    split: str,
    model: AutoEncoder,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    dataset = PortraitImageDataset(
        Path(args.data_root) / split,
        patch_size=None,
        training=False,
        channels=model.in_channels,
    )
    if args.max_images > 0:
        dataset.paths = dataset.paths[: args.max_images]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    names_out: list[str] = []
    latents_out: list[torch.Tensor] = []
    for images, names in loader:
        images = pad_to_multiple_of_8(images.to(device, non_blocking=True))
        latent = quantize_latents_to_int8(
            model.encode(images, deterministic=True)
        )
        if (
            args.expected_latent_channels > 0
            and int(latent.shape[1]) != args.expected_latent_channels
        ):
            raise ValueError(
                f"expected {args.expected_latent_channels} latent channels, "
                f"got {latent.shape[1]}"
            )
        for batch_index, name in enumerate(names):
            names_out.append(str(name))
            latents_out.append(latent[batch_index].contiguous())
        if len(names_out) % 25 == 0:
            print(f"split={split} images={len(names_out)}", flush=True)

    stacked = (
        torch.stack(latents_out)
        if latents_out
        else torch.empty(
            0,
            model.latent_channels,
            0,
            0,
            dtype=torch.int8,
        )
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}_latents.pt"
    torch.save(
        {
            "split": split,
            "names": names_out,
            "latents": stacked,
            "latent_shape": list(stacked.shape[1:]),
            "dtype": "int8",
            "checkpoint": str(args.checkpoint),
            "model": "AutoEncoder",
        },
        out_path,
    )
    print(
        f"wrote {out_path} images={len(names_out)} "
        f"latent_shape={list(stacked.shape)}",
        flush=True,
    )


def write_metadata(args: argparse.Namespace, model: AutoEncoder) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": "AutoEncoder",
        "checkpoint": str(args.checkpoint),
        "latent_channels": model.latent_channels,
        "latent_quant_step": model.latent_quant_step,
        "cache_format": (
            "{split}_latents.pt contains names and int8 latents [N,C,H,W]"
        ),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_ae(Path(args.checkpoint), device)
    write_metadata(args, model)
    print(
        f"device={device} checkpoint={args.checkpoint} out_dir={args.out_dir}",
        flush=True,
    )
    for split in args.splits:
        cache_split(split, model, args, device)


if __name__ == "__main__":
    main()
