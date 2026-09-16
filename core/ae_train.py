from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
PROJECT_TMP = os.path.join(os.getcwd(), "outputs", "tmp")
os.makedirs(PROJECT_TMP, exist_ok=True)
os.environ.setdefault("TMP", PROJECT_TMP)
os.environ.setdefault("TEMP", PROJECT_TMP)
os.environ.setdefault("TMPDIR", PROJECT_TMP)
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    os.path.join(PROJECT_TMP, "torchinductor"),
)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import AutoEncoder
from utils.dataset import PortraitImageDataset
from utils.metrics import ms_ssim, ms_ssim_loss, psnr
from utils.train_utils import load_checkpoint, save_checkpoint, set_seed


@dataclass
class EpochMetrics:
    loss: float = 0.0
    psnr: float = 0.0
    ms_ssim: float = 0.0
    batches: int = 0

    def update(
        self,
        *,
        loss: torch.Tensor,
        reference: torch.Tensor,
        reconstruction: torch.Tensor,
    ) -> None:
        self.loss += float(loss.detach())
        self.psnr += psnr(reference, reconstruction)
        self.ms_ssim += ms_ssim(reference, reconstruction)
        self.batches += 1

    def average(self) -> dict:
        count = max(self.batches, 1)
        return {
            "loss": self.loss / count,
            "psnr": self.psnr / count,
            "ms_ssim": self.ms_ssim / count,
            "batches": self.batches,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the quantized AE image-to-latent model."
    )
    parser.add_argument("--data-root", default=os.getenv("AE_DATA_ROOT", "data"))
    parser.add_argument("--epochs", type=int, default=int(os.getenv("AE_EPOCHS", "100")))
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("AE_BATCH_SIZE", "16")),
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=int(os.getenv("AE_PATCH_SIZE", "256")),
    )
    parser.add_argument("--lr", type=float, default=float(os.getenv("AE_LR", "1e-4")))
    parser.add_argument(
        "--lambda-distortion",
        type=float,
        default=float(os.getenv("AE_LAMBDA_DISTORTION", "20.0")),
    )
    parser.add_argument(
        "--lambda-l1",
        type=float,
        default=float(os.getenv("AE_LAMBDA_L1", "2.0")),
    )
    parser.add_argument(
        "--lambda-ms-ssim",
        type=float,
        default=float(os.getenv("AE_LAMBDA_MS_SSIM", "1.0")),
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=int(os.getenv("AE_CHANNELS", "3")),
    )
    parser.add_argument(
        "--latent-channels",
        type=int,
        default=int(os.getenv("AE_LATENT_CHANNELS", "16")),
    )
    parser.add_argument(
        "--latent-quant-step",
        type=float,
        default=float(os.getenv("AE_LATENT_QUANT_STEP", "1.0")),
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=int(os.getenv("AE_BASE_CHANNELS", "128")),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(os.getenv("AE_NUM_WORKERS", "2")),
    )
    parser.add_argument("--seed", type=int, default=int(os.getenv("AE_SEED", "42")))
    parser.add_argument(
        "--log-interval",
        type=int,
        default=int(os.getenv("AE_LOG_INTERVAL", "20")),
    )
    parser.add_argument(
        "--checkpoint",
        default=os.getenv("AE_CHECKPOINT", "outputs/checkpoints/AE.pth"),
    )
    parser.add_argument("--resume", default=os.getenv("AE_RESUME", ""))
    parser.add_argument(
        "--metrics-log",
        default=os.getenv("AE_METRICS_LOG", "outputs/checkpoints/ae_metrics.jsonl"),
    )
    parser.add_argument(
        "--save-metric",
        choices=["loss", "psnr"],
        default=os.getenv("AE_SAVE_METRIC", "psnr"),
    )
    return parser.parse_args()


def make_loader(
    root: str,
    split: str,
    args: argparse.Namespace,
    training: bool,
) -> DataLoader:
    dataset = PortraitImageDataset(
        Path(root) / split,
        patch_size=args.patch_size,
        training=training,
        channels=args.channels,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=training,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def make_model(args: argparse.Namespace, device: torch.device) -> AutoEncoder:
    return AutoEncoder(
        in_channels=args.channels,
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        latent_quant_step=args.latent_quant_step,
    ).to(device)


def make_optimizer(model: AutoEncoder, lr: float) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=lr)


def reconstruction_loss(output, image: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    reconstruction = output.reconstruction
    return (
        args.lambda_distortion * F.mse_loss(reconstruction, image)
        + args.lambda_l1 * F.l1_loss(reconstruction, image)
        + args.lambda_ms_ssim * ms_ssim_loss(image, reconstruction)
    )


def better(metric_name: str, current: float, best: float) -> bool:
    return current > best if metric_name == "psnr" else current < best


def initial_best(metric_name: str) -> float:
    return -float("inf") if metric_name == "psnr" else float("inf")


def run_epoch(
    model: AutoEncoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    training: bool,
) -> dict:
    model.train(training)
    metrics = EpochMetrics()
    phase = "train" if training else "val"
    total_batches = len(loader)

    for batch_index, (image, _) in enumerate(loader, start=1):
        image = image.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            output = model(image, deterministic=not training)
            loss = reconstruction_loss(output, image, args)
            if training:
                if optimizer is None:
                    raise RuntimeError("optimizer is required during training")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        metrics.update(
            loss=loss,
            reference=image,
            reconstruction=output.reconstruction,
        )
        if args.log_interval > 0 and (
            batch_index == 1
            or batch_index == total_batches
            or batch_index % args.log_interval == 0
        ):
            current = metrics.average()
            print(
                f"epoch={epoch:03d} phase={phase} "
                f"batch={batch_index:04d}/{total_batches:04d} "
                f"loss={current['loss']:.4f} psnr={current['psnr']:.2f} "
                f"ms_ssim={current['ms_ssim']:.4f}",
                flush=True,
            )

    return metrics.average()


def print_summary(epoch: int, train_metrics: dict, val_metrics: dict) -> None:
    print(
        f"epoch={epoch:03d} "
        f"train_loss={train_metrics['loss']:.4f} "
        f"train_psnr={train_metrics['psnr']:.2f} "
        f"train_ms_ssim={train_metrics['ms_ssim']:.4f} "
        f"val_loss={val_metrics['loss']:.4f} "
        f"val_psnr={val_metrics['psnr']:.2f} "
        f"val_ms_ssim={val_metrics['ms_ssim']:.4f}",
        flush=True,
    )


def append_metrics_log(args: argparse.Namespace, record: dict) -> None:
    if not args.metrics_log:
        return
    log_path = Path(args.metrics_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": datetime.now().isoformat(timespec="seconds"), **record}
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def train_model(
    model: AutoEncoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    optimizer = make_optimizer(model, args.lr)
    metric_name = args.save_metric
    best = initial_best(metric_name)
    saved = False

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args,
            epoch,
            training=True,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            None,
            device,
            args,
            epoch,
            training=False,
        )
        print_summary(epoch, train_metrics, val_metrics)

        current = val_metrics[metric_name]
        improved = better(metric_name, current, best)
        if improved:
            best = current
            save_checkpoint(args.checkpoint, model, optimizer, epoch, args)
            saved = True
            print(
                f"saved {args.checkpoint} metric={metric_name} value={best:.4f}",
                flush=True,
            )
        append_metrics_log(
            args,
            {
                "event": "epoch",
                "epoch": epoch,
                "lr": args.lr,
                "metric_name": metric_name,
                "metric_value": current,
                "best_metric_value": best,
                "improved": improved,
                "checkpoint_path": str(args.checkpoint),
                "train": train_metrics,
                "val": val_metrics,
            },
        )

    if not saved:
        save_checkpoint(args.checkpoint, model, optimizer, 0, args)
        print(f"saved fallback {args.checkpoint}", flush=True)

    load_checkpoint(args.checkpoint, model, map_location=device)
    print(f"loaded best checkpoint {args.checkpoint}", flush=True)


def resume_if_requested(
    model: AutoEncoder,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    if not args.resume:
        return
    resume_path = Path(args.resume)
    if not resume_path.exists():
        raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
    checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
    state = {
        key: value
        for key, value in checkpoint["model"].items()
        if key.startswith("encoder.") or key.startswith("decoder.")
    }
    model.load_state_dict(state)
    print(f"resumed AE weights from {resume_path}", flush=True)


def save_timestamped_checkpoint(checkpoint_path: str) -> Path:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint was not created: {path}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_path = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
    shutil.copy2(path, timestamped_path)
    print(f"saved timestamped checkpoint {timestamped_path}", flush=True)
    return timestamped_path


def print_config(args: argparse.Namespace, device: torch.device) -> None:
    print(f"device={device} cuda_available={torch.cuda.is_available()}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"config epochs={args.epochs} batch_size={args.batch_size} "
        f"patch_size={args.patch_size} lr={args.lr} channels={args.channels} "
        f"latent_channels={args.latent_channels} "
        f"lambda_distortion={args.lambda_distortion} "
        f"lambda_l1={args.lambda_l1} lambda_ms_ssim={args.lambda_ms_ssim} "
        f"save_metric={args.save_metric} metrics_log={args.metrics_log}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_config(args, device)

    train_loader = make_loader(args.data_root, "train", args, training=True)
    val_loader = make_loader(args.data_root, "val", args, training=False)
    model = make_model(args, device)
    resume_if_requested(model, args, device)
    train_model(model, train_loader, val_loader, device, args)
    save_timestamped_checkpoint(args.checkpoint)


if __name__ == "__main__":
    main()
