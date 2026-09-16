from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from _bootstrap import add_project_root

add_project_root()

from core.dna_codec import encode_latent_to_dna_records
from core.ae_codec import encode_image_to_latent, filtered_ae_args, load_ae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode one image into latent DNA strands.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--ae-checkpoint", default="outputs/checkpoints/AE.pth")
    parser.add_argument("--output-dir", default="outputs/encoded_dna")
    parser.add_argument("--position-nt", type=int, default=14)
    parser.add_argument("--name", default="", help="Optional output stem.")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or image_path.stem

    ae, ae_args = load_ae(Path(args.ae_checkpoint), device)
    latent_i8 = encode_image_to_latent(ae, image_path, device)
    channels, height, width = latent_i8.shape

    dna_path = output_dir / f"{stem}_dna.jsonl"
    with dna_path.open("w", encoding="utf-8") as handle:
        for record in encode_latent_to_dna_records(
            latent_i8,
            image_name=image_path.name,
            position_nt=args.position_nt,
        ):
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    latent_path = output_dir / f"{stem}_latent.pt"
    torch.save({"latent": latent_i8, "image": image_path.name}, latent_path)
    meta = {
        "image": image_path.name,
        "ae_checkpoint": args.ae_checkpoint,
        "ae_args": filtered_ae_args(ae_args),
        "latent_shape": [int(channels), int(height), int(width)],
        "position_nt": args.position_nt,
        "dna_jsonl": str(dna_path),
        "latent_pt": str(latent_path),
    }
    meta_path = output_dir / f"{stem}_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(f"encoded image={image_path} dna={dna_path} latent={latent_path} metadata={meta_path}")


if __name__ == "__main__":
    main()
