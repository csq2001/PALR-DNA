from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from _bootstrap import add_project_root

add_project_root()

from core.ae_codec import crop_like, decode_latent_to_image, image_to_tensor, load_ae, save_tensor_png
from utils.metrics import mse, ms_ssim, psnr, ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct an image from corrected latent tensors.")
    parser.add_argument("--latent-pt", required=True, help="PT file from correct_dna_to_latent.py.")
    parser.add_argument("--ae-checkpoint", default="outputs/checkpoints/AE.pth")
    parser.add_argument(
        "--latent-key",
        default="top1_completion_latent",
        choices=["top1_completion_latent", "recovered_latent", "truth_latent"],
    )
    parser.add_argument("--reference-image", default="", help="Optional image for PSNR/SSIM metrics.")
    parser.add_argument("--output-dir", default="outputs/reconstructions")
    parser.add_argument("--name", default="", help="Optional output stem.")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def reconstruction_file_stem(name: str, latent_key: str) -> str:
    if latent_key == "top1_completion_latent":
        return f"{name}_recon"
    return f"{name}_{latent_key}"


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    latent_path = Path(args.latent_pt)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or latent_path.stem.replace("_corrected_latents", "")

    ae, _ = load_ae(Path(args.ae_checkpoint), device)
    payload = torch.load(latent_path, map_location="cpu", weights_only=False)
    recon = decode_latent_to_image(ae, payload[args.latent_key], device)

    metrics = {
        "latent_pt": str(latent_path),
        "latent_key": args.latent_key,
        "ae_checkpoint": args.ae_checkpoint,
    }
    if args.reference_image:
        reference = image_to_tensor(Path(args.reference_image), ae.in_channels).to(device).clamp(0.0, 1.0)
        recon = crop_like(recon, reference).clamp(0.0, 1.0)
        mse_value = float(mse(reference, recon).item()) * 65025.0
        metrics.update(
            {
                "mse": mse_value,
                "psnr": psnr(reference, recon),
                "ssim": float(ssim(reference, recon).item()),
                "ms_ssim": ms_ssim(reference, recon),
            }
        )

    file_stem = reconstruction_file_stem(stem, args.latent_key)
    image_path = output_dir / f"{file_stem}.png"
    metrics_path = output_dir / f"{file_stem}_metrics.json"
    save_tensor_png(recon, image_path)
    metrics["reconstruction_png"] = str(image_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"reconstructed image={image_path} metrics={metrics_path}")


if __name__ == "__main__":
    main()
