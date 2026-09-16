import argparse
import csv
import json
import math
import os
import time
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from core.dlrt_decoder import (
    decode_latent_from_dna,
    decode_latent_from_dna_hard_codebook,
    load_dlrt_checkpoint,
)
from core.dlrt_codec import INDEX_BYTES, PAYLOAD_BYTES
from core.ae_codec import crop_like, image_to_tensor, load_ae, pad_to_multiple_of_8, save_tensor_png
from utils.metrics import mse, ms_ssim, psnr, ssim

def load_image_records(args):
    split_dir = Path(args.data_root) / args.split
    cache_path = Path(args.latent_cache_dir) / f"{args.split}_latents.pt"
    using_latent_cache = False
    cached = None
    if not args.no_latent_cache and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        names = [str(name) for name in cached["names"]]
        latents = cached["latents"]
        using_latent_cache = True
    else:
        names = [
            path.name
            for path in sorted(split_dir.iterdir())
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        ]
        latents = [None] * len(names)
    if args.max_images and args.max_images > 0:
        names = names[: args.max_images]
        latents = latents[: args.max_images]
    records = [
        {
            "path": split_dir / name,
            "latent": latents[index].contiguous() if using_latent_cache else None,
        }
        for index, name in enumerate(names)
    ]
    return records, using_latent_cache


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CRC validation and MLLV index+payload byte recovery accuracy."
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--split", default="test")
    parser.add_argument("--latent-cache-dir", default="outputs/latent_cache")
    parser.add_argument("--no-latent-cache", action="store_true")
    parser.add_argument(
        "--image-path",
        default="",
        help="Optional image path. When set, evaluate only this image and encode it without the latent cache.",
    )
    parser.add_argument("--image-index", type=int, default=1, help="1-based index in the selected split.")
    parser.add_argument("--all-images", action="store_true", help="Evaluate all images in the selected split.")
    parser.add_argument(
        "--start-image",
        type=int,
        default=1,
        help="1-based start index in --all-images mode.",
    )
    parser.add_argument("--max-images", type=int, default=0, help="Limit image count in --all-images mode; 0 means all.")
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/DLRT.pth",
    )
    parser.add_argument("--ae-checkpoint", default="outputs/checkpoints/AE.pth")
    parser.add_argument(
        "--total-error-rates",
        default="0.01,0.02,0.03,0.04,0.05",
        help="Comma-separated total error rates. Split as sub=rate/2, ins=rate/4, del=rate/4.",
    )
    parser.add_argument(
        "--error-triples",
        default="",
        help="Optional semicolon-separated triples sub,ins,del; overrides --total-error-rates.",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--decoder",
        choices=("dlrt", "hard-codebook"),
        default="dlrt",
        help="Use the trained DLRT or a non-neural hard codebook decoder.",
    )
    parser.add_argument(
        "--top-k-values",
        default="",
        help="Optional comma-separated top-k values, e.g. 1,2,4,8,16. Overrides --top-k.",
    )
    parser.add_argument("--beam-size", type=int, default=256)
    parser.add_argument(
        "--hard-codebook-beam",
        type=int,
        default=256,
        help="Beam width for hard codebook edit-distance decoding.",
    )
    parser.add_argument("--infer-batch-size", type=int, default=256)
    parser.add_argument("--chain-limit", type=int, default=0, help="0 means decode all latent chains.")
    parser.add_argument(
        "--no-image-quality",
        action="store_true",
        help="Skip AE decoding and image quality metrics.",
    )
    parser.add_argument(
        "--mse-scale",
        type=float,
        default=65025.0,
        help="Scale MSE by this value. Use 65025 for [0,255] MSE, 1 for normalized [0,1] MSE.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--output-jsonl",
        default="outputs/checkpoints/single_image_decoding_accuracy.jsonl",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/checkpoints/single_image_decoding_accuracy.csv",
    )
    parser.add_argument(
        "--reconstruction-dir",
        default="",
        help="Optional directory for saving reconstructions with and without MLLV.",
    )
    return parser.parse_args()


def error_rate_label(total: float) -> str:
    return f"rate{int(round(total * 1000)):03d}"


def error_ratio_label(sub_rate: float, ins_rate: float, del_rate: float) -> str:
    total = sub_rate + ins_rate + del_rate
    if total <= 0:
        return "ratio000"
    fractions = [
        Fraction(rate / total).limit_denominator(20)
        for rate in (sub_rate, ins_rate, del_rate)
    ]
    denominator_lcm = 1
    for fraction in fractions:
        denominator_lcm = math.lcm(denominator_lcm, fraction.denominator)
    values = [fraction.numerator * (denominator_lcm // fraction.denominator) for fraction in fractions]
    divisor = 0
    for value in values:
        divisor = math.gcd(divisor, value)
    if divisor <= 0:
        return "ratio000"
    return "ratio" + "".join(str(value // divisor) for value in values)


def parse_error_schedule(args) -> list[tuple[str, float, float, float]]:
    if args.error_triples.strip():
        schedule = []
        for item in args.error_triples.split(";"):
            if not item.strip():
                continue
            sub_rate, ins_rate, del_rate = [float(value) for value in item.split(",")]
            total = sub_rate + ins_rate + del_rate
            label = f"{error_ratio_label(sub_rate, ins_rate, del_rate)}_{error_rate_label(total)}"
            schedule.append((label, sub_rate, ins_rate, del_rate))
        return schedule
    schedule = []
    for item in args.total_error_rates.split(","):
        if not item.strip():
            continue
        total = float(item)
        schedule.append((f"{total:.4f}", total * 0.5, total * 0.25, total * 0.25))
    return schedule


def parse_top_k_values(args) -> list[int]:
    if not args.top_k_values.strip():
        return [args.top_k]
    values = []
    for item in args.top_k_values.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 1:
            raise ValueError("--top-k-values must contain positive integers")
        values.append(value)
    if not values:
        raise ValueError("--top-k-values did not contain any valid values")
    return values


def load_one_image_record(args):
    if args.image_index < 1:
        raise ValueError("--image-index is 1-based and must be >= 1")
    records_args = SimpleNamespace(
        data_root=args.data_root,
        split=args.split,
        latent_cache_dir=args.latent_cache_dir,
        max_images=args.image_index,
        no_latent_cache=args.no_latent_cache,
    )
    records, using_latent_cache = load_image_records(records_args)
    if len(records) < args.image_index:
        raise IndexError(f"--image-index {args.image_index} exceeds split size {len(records)}")
    return records[args.image_index - 1], using_latent_cache


def load_selected_image_records(args):
    if args.image_path:
        image_path = Path(args.image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return [{"path": image_path, "latent": None}], False
    if args.all_images:
        if args.start_image < 1:
            raise ValueError("--start-image is 1-based and must be >= 1")
        fetch_limit = 0
        if args.max_images > 0:
            fetch_limit = args.start_image + args.max_images - 1
        records_args = SimpleNamespace(
            data_root=args.data_root,
            split=args.split,
            latent_cache_dir=args.latent_cache_dir,
            max_images=fetch_limit,
            no_latent_cache=args.no_latent_cache,
        )
        records, using_latent_cache = load_image_records(records_args)
        records = records[args.start_image - 1 :]
        return records, using_latent_cache
    record, using_latent_cache = load_one_image_record(args)
    return [record], using_latent_cache


def latent_from_record(record, ae, device):
    cached_latent = record["latent"]
    if cached_latent is not None:
        return cached_latent.contiguous()
    x = image_to_tensor(record["path"], ae.in_channels).to(device)
    padded = pad_to_multiple_of_8(x)
    y_hat = ae.encode(padded, deterministic=True)
    return torch.round(y_hat[0]).clamp(-128, 127).to(torch.int8).cpu()


def latent_recovery_accuracy(truth: torch.Tensor, recovered: torch.Tensor, decode_chains: int) -> tuple[float, float]:
    channels, height, width = truth.shape
    truth_flat = truth.view(channels, height * width)[:, :decode_chains]
    recovered_flat = recovered.view(channels, height * width)[:, :decode_chains]
    byte_correct = recovered_flat.eq(truth_flat)
    chain_correct = byte_correct.all(dim=0)
    byte_accuracy = byte_correct.float().mean().item()
    chain_accuracy = chain_correct.float().mean().item()
    return chain_accuracy, byte_accuracy


def index_payload_recovery_accuracy(
    truth: torch.Tensor,
    recovered: torch.Tensor,
    decode_chains: int,
) -> tuple[float, float]:
    target_bytes = INDEX_BYTES + PAYLOAD_BYTES
    truth_view = truth[:decode_chains, :target_bytes]
    recovered_view = recovered[:decode_chains, :target_bytes]
    byte_correct = recovered_view.eq(truth_view)
    chain_correct = byte_correct.all(dim=1)
    byte_accuracy = byte_correct.float().mean().item()
    chain_accuracy = chain_correct.float().mean().item()
    return chain_accuracy, byte_accuracy


@torch.no_grad()
def reconstruct_latent(
    ae,
    x_ref: torch.Tensor,
    latent: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    latent_tensor = latent.unsqueeze(0).to(device, dtype=torch.float32)
    return crop_like(ae.decode(latent_tensor), x_ref).clamp(0.0, 1.0)


def reconstruction_metrics(
    x_ref: torch.Tensor,
    recon: torch.Tensor,
    mse_scale: float,
    prefix: str,
) -> dict:
    x_ref = x_ref.clamp(0.0, 1.0)
    return {
        f"{prefix}_mse": float(mse(x_ref, recon).item()) * mse_scale,
        f"{prefix}_psnr": psnr(x_ref, recon),
        f"{prefix}_ssim": float(ssim(x_ref, recon).item()),
        f"{prefix}_ms_ssim": ms_ssim(x_ref, recon),
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def average(records: list[dict], key: str) -> float:
    if not records:
        return 0.0
    return sum(float(record[key]) for record in records) / len(records)


def standard_error(records: list[dict], key: str) -> float:
    if len(records) <= 1:
        return 0.0
    values = [float(record[key]) for record in records]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return (variance ** 0.5) / (len(values) ** 0.5)


def summarize_rows(
    rows: list[dict],
    schedule: list[tuple[str, float, float, float]],
    top_k_values: list[int],
) -> list[dict]:
    summaries = []
    metric_keys = [
        "crc_validated_chain_rate",
        "index_top1_accuracy",
        "index_topk_accuracy",
        "crc_erasure_chain_accuracy",
        "crc_erasure_byte_accuracy",
        "crc_mllv_chain_accuracy",
        "crc_mllv_byte_accuracy",
        "crc_erasure_mse",
        "crc_erasure_psnr",
        "crc_erasure_ssim",
        "crc_erasure_ms_ssim",
        "crc_mllv_mse",
        "crc_mllv_psnr",
        "crc_mllv_ssim",
        "crc_mllv_ms_ssim",
    ]
    for top_k in top_k_values:
        for rate_label, sub_rate, ins_rate, del_rate in schedule:
            rate_rows = [
                row
                for row in rows
                if row["rate_label"] == rate_label and int(row["top_k"]) == top_k
            ]
            summary = {
                "event": "summary",
                "top_k": top_k,
                "beam_size": int(rate_rows[0]["beam_size"]) if rate_rows else 0,
                "rate_label": rate_label,
                "sub_rate": sub_rate,
                "ins_rate": ins_rate,
                "del_rate": del_rate,
                "images": len(rate_rows),
            }
            for key in metric_keys:
                if rate_rows and key in rate_rows[0]:
                    summary[key] = average(rate_rows, key)
                    summary[f"{key}_se"] = standard_error(rate_rows, key)
            summaries.append(summary)
    return summaries


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, model_args = load_dlrt_checkpoint(Path(args.checkpoint), device)
    if args.decoder == "hard-codebook":
        model = None
    ae, _ = load_ae(Path(args.ae_checkpoint), device)
    image_records, using_latent_cache = load_selected_image_records(args)
    schedule = parse_error_schedule(args)
    top_k_values = parse_top_k_values(args)

    print(
        f"device={device} images={len(image_records)} "
        f"rates={len(schedule)} top_k_values={top_k_values} beam_size={args.beam_size}",
        flush=True,
    )

    rows = []
    jsonl_path = Path(args.output_jsonl)
    csv_path = Path(args.output_csv)
    if jsonl_path.exists():
        jsonl_path.unlink()
    first_image_index = args.start_image if args.all_images else args.image_index
    for image_index, image_record in enumerate(image_records, start=first_image_index):
        latent = latent_from_record(image_record, ae, device)
        x_ref = None
        if not args.no_image_quality:
            x_ref = image_to_tensor(image_record["path"], ae.in_channels).to(device)
        for top_k in top_k_values:
            for rate_label, sub_rate, ins_rate, del_rate in schedule:
                started = time.time()
                eval_seed = (
                    args.seed
                    + image_index * 1009
                    + int(round(sub_rate * 1_000_000)) * 3
                    + int(round(ins_rate * 1_000_000)) * 5
                    + int(round(del_rate * 1_000_000)) * 7
                )
                if args.decoder == "hard-codebook":
                    decoded = decode_latent_from_dna_hard_codebook(
                        model_args=model_args,
                        latent=latent,
                        sub_rate=sub_rate,
                        ins_rate=ins_rate,
                        del_rate=del_rate,
                        seed=eval_seed,
                        chain_limit=args.chain_limit,
                        hard_codebook_beam=args.hard_codebook_beam,
                    )
                else:
                    decoded = decode_latent_from_dna(
                        model=model,
                        model_args=model_args,
                        device=device,
                        latent=latent,
                        sub_rate=sub_rate,
                        ins_rate=ins_rate,
                        del_rate=del_rate,
                        top_k=top_k,
                        beam_size=args.beam_size,
                        infer_batch_size=args.infer_batch_size,
                        seed=eval_seed,
                        chain_limit=args.chain_limit,
                    )
                summary = decoded["summary"]
                decoded_chains = int(summary["decoded_chains"])
                crc_erasure_chain_acc, crc_erasure_byte_acc = index_payload_recovery_accuracy(
                    decoded["full_chain_truth"],
                    decoded["crc_erasure_full_chain"],
                    decoded_chains,
                )
                crc_mllv_chain_acc, crc_mllv_byte_acc = index_payload_recovery_accuracy(
                    decoded["full_chain_truth"],
                    decoded["crc_mllv_full_chain"],
                    decoded_chains,
                )
                crc_rate = float(summary["beam_crc_pass"]) / max(decoded_chains, 1)
                row = {
                    "event": "image",
                    "image": image_record["path"].name,
                    "image_index": image_index,
                    "using_latent_cache": using_latent_cache,
                    "decoder": args.decoder,
                    "top_k": top_k,
                    "beam_size": args.beam_size,
                    "hard_codebook_beam": args.hard_codebook_beam,
                    "rate_label": rate_label,
                    "sub_rate": sub_rate,
                    "ins_rate": ins_rate,
                    "del_rate": del_rate,
                    "decoded_chains": decoded_chains,
                    "target_bytes": INDEX_BYTES + PAYLOAD_BYTES,
                    "accuracy_target": "index_payload",
                    "index_top_k": int(summary["index_top_k"]),
                    "index_top1_accuracy": float(summary["index_top1_accuracy"]),
                    "index_topk_accuracy": float(summary["index_topk_accuracy"]),
                    "crc_validated_chains": int(summary["beam_crc_pass"]),
                    "crc_validated_chain_rate": crc_rate,
                    "crc_erasure_chain_accuracy": crc_erasure_chain_acc,
                    "crc_erasure_byte_accuracy": crc_erasure_byte_acc,
                    "crc_mllv_chain_accuracy": crc_mllv_chain_acc,
                    "crc_mllv_byte_accuracy": crc_mllv_byte_acc,
                    "observed_crc_readable": int(summary["observed_crc_readable"]),
                    "observed_crc_matches_truth": int(summary["observed_crc_matches_truth"]),
                    "average_errors_per_chain": float(summary["average_errors_per_chain"]),
                    "substitutions": int(summary["substitutions"]),
                    "insertions": int(summary["insertions"]),
                    "deletions": int(summary["deletions"]),
                    "elapsed_seconds": time.time() - started,
                }
                if x_ref is not None:
                    erasure_recon = reconstruct_latent(
                        ae,
                        x_ref,
                        decoded["recovered_latent"],
                        device,
                    )
                    mllv_recon = reconstruct_latent(
                        ae,
                        x_ref,
                        decoded["crc_top1_fallback_latent"],
                        device,
                    )
                    row.update(
                        reconstruction_metrics(
                            x_ref,
                            erasure_recon,
                            args.mse_scale,
                            "crc_erasure",
                        )
                    )
                    row.update(
                        reconstruction_metrics(
                            x_ref,
                            mllv_recon,
                            args.mse_scale,
                            "crc_mllv",
                        )
                    )
                    if args.reconstruction_dir:
                        reconstruction_dir = Path(args.reconstruction_dir)
                        reconstruction_dir.mkdir(parents=True, exist_ok=True)
                        total_error_percent = int(round((sub_rate + ins_rate + del_rate) * 100))
                        stem = image_record["path"].stem
                        name_prefix = (
                            f"{stem}_error{total_error_percent:02d}pct_"
                            f"top{top_k}_beam{args.beam_size}"
                        )
                        erasure_path = reconstruction_dir / f"{name_prefix}_without_mllv.png"
                        mllv_path = reconstruction_dir / f"{name_prefix}_with_mllv.png"
                        save_tensor_png(erasure_recon, erasure_path)
                        save_tensor_png(mllv_recon, mllv_path)
                        row["crc_erasure_reconstruction_png"] = str(erasure_path)
                        row["crc_mllv_reconstruction_png"] = str(mllv_path)
                    row["elapsed_seconds"] = time.time() - started
                rows.append(row)
                append_jsonl(jsonl_path, row)
                quality_text = ""
                if x_ref is not None:
                    quality_text = (
                        f"crc_mllv_psnr={row['crc_mllv_psnr']:.2f} "
                        f"crc_mllv_ms_ssim={row['crc_mllv_ms_ssim']:.4f} "
                    )
                print(
                    f"top_k={top_k} rate={rate_label} image={image_index:04d}/{len(image_records):04d} "
                    f"crc_rate={crc_rate:.4f} "
                    f"crc_erasure_chain_acc={crc_erasure_chain_acc:.4f} "
                    f"crc_mllv_chain_acc={crc_mllv_chain_acc:.4f} "
                    f"crc_mllv_byte_acc={crc_mllv_byte_acc:.4f} "
                    f"{quality_text}"
                    f"crc={row['crc_validated_chains']}/{decoded_chains} "
                    f"elapsed={row['elapsed_seconds']:.1f}s",
                    flush=True,
                )

    summary_rows = summarize_rows(rows, schedule, top_k_values)
    for summary in summary_rows:
        append_jsonl(jsonl_path, summary)
    write_csv(csv_path, summary_rows)
    print(f"done jsonl={args.output_jsonl} csv={args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
