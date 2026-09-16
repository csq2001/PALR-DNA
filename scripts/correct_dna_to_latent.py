from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from _bootstrap import add_project_root

add_project_root()

from core.dlrt_decoder import correct_corrupted_records_to_latents, load_dlrt_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correct corrupted DNA strands into latent tensors.")
    parser.add_argument("--input-jsonl", required=True, help="Corrupted DNA JSONL from inject_dna_errors.py.")
    parser.add_argument("--dlrt-checkpoint", default="outputs/checkpoints/DLRT.pth")
    parser.add_argument("--output-dir", default="outputs/corrected_latents")
    parser.add_argument("--latent-shape", default="", help="Optional C,H,W. Inferred from row/col if omitted.")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--index-top-k", type=int, default=None, help="Defaults to --top-k when omitted.")
    parser.add_argument("--beam-size", type=int, default=256)
    parser.add_argument("--infer-batch-size", type=int, default=256)
    parser.add_argument("--name", default="", help="Optional output stem.")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def parse_shape(value: str, records: list[dict]) -> tuple[int, int, int]:
    if value.strip():
        c, h, w = [int(part) for part in value.split(",")]
        return c, h, w
    channels = len(records[0].get("bytes", [])) or 16
    height = max(int(record["row"]) for record in records) + 1
    width = max(int(record["col"]) for record in records) + 1
    return channels, height, width


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or input_path.stem.replace("_corrupted", "")

    records = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"no records found in {input_path}")

    channels, height, width = parse_shape(args.latent_shape, records)
    model, model_args = load_dlrt_checkpoint(Path(args.dlrt_checkpoint), device)
    corrected = correct_corrupted_records_to_latents(
        records=records,
        model=model,
        model_args=model_args,
        device=device,
        latent_shape=(channels, height, width),
        top_k=args.top_k,
        index_top_k=args.index_top_k,
        beam_size=args.beam_size,
        infer_batch_size=args.infer_batch_size,
    )

    latent_path = output_dir / f"{stem}_corrected_latents.pt"
    torch.save(
        {
            "recovered_latent": corrected["recovered_latent"],
            "top1_completion_latent": corrected["top1_completion_latent"],
            "truth_latent": corrected["truth_latent"],
            "latent_shape": [channels, height, width],
            "source_jsonl": str(input_path),
        },
        latent_path,
    )
    metrics = {**corrected["metrics"], "corrected_latents_pt": str(latent_path)}
    metrics_path = output_dir / f"{stem}_correction_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"corrected chains={metrics['decoded_chains']} "
        f"latent={latent_path} metrics={metrics_path}"
    )


if __name__ == "__main__":
    main()

