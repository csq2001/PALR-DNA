from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.functional as F

from core.dlrt_data import make_loader
from core.dlrt_model import DNAByteNet
from core.dna_edit_code import codebook_metadata


@dataclass
class EpochStats:
    loss: float = 0.0
    byte_top1: float = 0.0
    byte_top4: float = 0.0
    byte_top8: float = 0.0
    chain_top1: float = 0.0
    samples: int = 0

    def update(self, loss, logits, target):
        batch = target.shape[0]
        with torch.no_grad():
            top8 = logits.topk(8, dim=-1).indices
            target_expanded = target.unsqueeze(-1)
            top1_ok = top8[..., :1].eq(target_expanded)
            top4_ok = top8[..., :4].eq(target_expanded).any(dim=-1)
            top8_ok = top8.eq(target_expanded).any(dim=-1)
            self.loss += float(loss) * batch
            self.byte_top1 += top1_ok.float().mean().item() * batch
            self.byte_top4 += top4_ok.float().mean().item() * batch
            self.byte_top8 += top8_ok.float().mean().item() * batch
            self.chain_top1 += top1_ok.squeeze(-1).all(dim=-1).float().mean().item() * batch
            self.samples += batch

    def average(self) -> dict:
        count = max(self.samples, 1)
        return {
            "loss": self.loss / count,
            "byte_top1": self.byte_top1 / count,
            "byte_top4": self.byte_top4 / count,
            "byte_top8": self.byte_top8 / count,
            "chain_top1": self.chain_top1 / count,
            "samples": self.samples,
        }


def parse_rate_range(value: str) -> tuple[float, float]:
    parts = value.split(",")
    if len(parts) == 1:
        low = high = float(parts[0])
    elif len(parts) == 2:
        low, high = float(parts[0]), float(parts[1])
    else:
        raise argparse.ArgumentTypeError("rate range must be VALUE or LOW,HIGH")
    if not 0.0 <= low <= high <= 1.0:
        raise argparse.ArgumentTypeError("rates must satisfy 0 <= low <= high <= 1")
    return low, high


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the DNA-to-Latent Recovery Transformer for latent bytes."
    )
    parser.add_argument("--cache-dir", default=os.getenv("AE_LATENT_CACHE", "outputs/latent_cache"))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--output", default="outputs/checkpoints/DLRT.pth")
    parser.add_argument(
        "--resume",
        default="",
        help="Resume model and optimizer state from this checkpoint; --epochs is the number of extra epochs to run.",
    )
    parser.add_argument("--metrics-log", default="outputs/checkpoints/dlrt_metrics.jsonl")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--samples-per-epoch", type=int, default=500_000)
    parser.add_argument("--val-samples", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument(
        "--batch-metrics-log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append batch-level metrics to the JSONL metrics log.",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--position-nt", type=int, default=14)
    parser.add_argument(
        "--byte-code-nt",
        type=int,
        default=7,
        help="Nucleotide length used to encode each byte in the DNA codebook.",
    )
    parser.add_argument(
        "--byte-codebook-mode",
        choices=("designed", "constrained-designed", "legacy", "random", "base4"),
        default="designed",
        help="Byte-to-DNA codebook construction used for ablation experiments.",
    )
    parser.add_argument(
        "--byte-codebook-seed",
        type=int,
        default=42,
        help="Seed for random or generated designed codebooks.",
    )
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--target-bytes", type=int, default=22)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--sub-rate", type=parse_rate_range, default=(0.002, 0.03))
    parser.add_argument("--ins-rate", type=parse_rate_range, default=(0.001, 0.015))
    parser.add_argument("--del-rate", type=parse_rate_range, default=(0.001, 0.015))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=0)
    return parser.parse_args()


def batch_metrics(loss: float, logits: torch.Tensor, target: torch.Tensor) -> dict:
    with torch.no_grad():
        top8 = logits.topk(8, dim=-1).indices
        target_expanded = target.unsqueeze(-1)
        top1_ok = top8[..., :1].eq(target_expanded)
        top4_ok = top8[..., :4].eq(target_expanded).any(dim=-1)
        top8_ok = top8.eq(target_expanded).any(dim=-1)
        return {
            "loss": float(loss),
            "byte_top1": top1_ok.float().mean().item(),
            "byte_top4": top4_ok.float().mean().item(),
            "byte_top8": top8_ok.float().mean().item(),
            "chain_top1": top1_ok.squeeze(-1).all(dim=-1).float().mean().item(),
            "samples": int(target.shape[0]),
        }


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    args,
    *,
    epoch: int,
    phase: str,
    training: bool,
) -> dict:
    model.train(training)
    stats = EpochStats()
    total_batches = len(loader)
    started = time.time()
    for batch_index, (input_ids, valid_mask, target) in enumerate(loader, start=1):
        input_ids = input_ids.to(device, non_blocking=True)
        valid_mask = valid_mask.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            logits = model(input_ids, valid_mask)
            loss = F.cross_entropy(
                logits.reshape(-1, 256),
                target.reshape(-1),
                label_smoothing=args.label_smoothing,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        stats.update(loss.item(), logits.detach(), target)
        should_log = (
            args.log_interval > 0
            and (
                batch_index == 1
                or batch_index == total_batches
                or batch_index % args.log_interval == 0
            )
        )
        if should_log:
            metrics = batch_metrics(loss.item(), logits.detach(), target)
            elapsed = time.time() - started
            record = {
                "event": "batch",
                "epoch": epoch,
                "phase": phase,
                "batch": batch_index,
                "total_batches": total_batches,
                "elapsed_seconds": elapsed,
                **metrics,
            }
            if args.batch_metrics_log:
                append_metrics(Path(args.metrics_log), record)
            print(
                f"event=batch epoch={epoch:03d} phase={phase} "
                f"batch={batch_index:05d}/{total_batches:05d} "
                f"loss={metrics['loss']:.4f} top1={metrics['byte_top1']:.4f} "
                f"top4={metrics['byte_top4']:.4f} top8={metrics['byte_top8']:.4f} "
                f"chain_top1={metrics['chain_top1']:.4f} elapsed={elapsed:.1f}s",
                flush=True,
            )
    return stats.average()


def append_metrics(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def save_checkpoint(path: Path, model, optimizer, epoch: int, args, val_metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "val_metrics": val_metrics,
        },
        path,
    )


def apply_resume_model_args(args, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args", {})
    for name in (
        "latent_channels",
        "target_bytes",
        "max_len",
        "d_model",
        "layers",
        "heads",
        "dropout",
        "byte_code_nt",
        "byte_codebook_mode",
        "byte_codebook_seed",
    ):
        if name in saved_args:
            setattr(args, name, saved_args[name])


def resume_checkpoint(path: Path, model, optimizer, device: torch.device) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = int(checkpoint.get("epoch", 0))
    val_metrics = checkpoint.get("val_metrics", {})
    best = float(val_metrics.get("byte_top8", -float("inf")))
    return start_epoch, best


def main() -> None:
    args = parse_args()
    if args.resume:
        apply_resume_model_args(args, Path(args.resume))
    args.codebook_metadata = codebook_metadata(
        length=args.byte_code_nt,
        mode=args.byte_codebook_mode,
        seed=args.byte_codebook_seed,
    )
    minimum_clean_len = args.target_bytes * args.byte_code_nt
    if args.max_len < minimum_clean_len:
        raise ValueError(
            f"--max-len {args.max_len} is shorter than the clean chain length "
            f"{minimum_clean_len}; increase --max-len for {args.byte_code_nt}-nt codewords."
        )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = Path(args.cache_dir)
    train_path = cache_dir / f"{args.train_split}_latents.pt"
    val_path = cache_dir / f"{args.val_split}_latents.pt"
    train_loader = make_loader(train_path, args, args.samples_per_epoch, args.seed, shuffle=False)
    val_loader = make_loader(val_path, args, args.val_samples, args.seed + 12345, shuffle=False)
    model = DNAByteNet(
        latent_channels=args.latent_channels,
        target_bytes=args.target_bytes,
        max_len=args.max_len,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    best = -float("inf")
    start_epoch = 0
    if args.resume:
        start_epoch, best = resume_checkpoint(Path(args.resume), model, optimizer, device)
        append_metrics(
            Path(args.metrics_log),
            {
                "event": "resume",
                "checkpoint": str(args.resume),
                "start_epoch": start_epoch,
                "best_byte_top8": best,
            },
        )
    print(
        f"device={device} train={train_path} val={val_path} "
        f"samples_per_epoch={args.samples_per_epoch} val_samples={args.val_samples} "
        f"start_epoch={start_epoch} extra_epochs={args.epochs} "
        f"codebook={args.byte_codebook_mode}-{args.byte_code_nt}nt "
        f"min_edit={args.codebook_metadata['min_levenshtein_distance']}",
        flush=True,
    )
    end_epoch = start_epoch + args.epochs
    for epoch in range(start_epoch + 1, end_epoch + 1):
        started = time.time()
        train_loader.dataset.set_epoch(epoch)
        val_loader.dataset.set_epoch(epoch)
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args,
            epoch=epoch,
            phase="train",
            training=True,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            args,
            epoch=epoch,
            phase="val",
            training=False,
        )
        elapsed = time.time() - started
        improved = val_metrics["byte_top8"] > best
        if improved:
            best = val_metrics["byte_top8"]
            save_checkpoint(Path(args.output), model, optimizer, epoch, args, val_metrics)
        if args.save_every and epoch % args.save_every == 0:
            periodic = Path(args.output).with_name(f"{Path(args.output).stem}_epoch{epoch:03d}.pth")
            save_checkpoint(periodic, model, optimizer, epoch, args, val_metrics)
        record = {
            "event": "epoch",
            "epoch": epoch,
            "elapsed_seconds": elapsed,
            "train": train_metrics,
            "val": val_metrics,
            "best_byte_top8": best,
            "improved": improved,
        }
        append_metrics(Path(args.metrics_log), record)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} train_top1={train_metrics['byte_top1']:.4f} "
            f"train_top8={train_metrics['byte_top8']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_top1={val_metrics['byte_top1']:.4f} "
            f"val_top4={val_metrics['byte_top4']:.4f} val_top8={val_metrics['byte_top8']:.4f} "
            f"val_chain_top1={val_metrics['chain_top1']:.4f} "
            f"elapsed={elapsed:.1f}s improved={improved}",
            flush=True,
        )
    print(f"best checkpoint: {args.output} byte_top8={best:.4f}", flush=True)

