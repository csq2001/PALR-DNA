# AE-DLRT DNA Image Storage

Research code for storing an image as DNA strands and recovering it after simulated substitution, insertion, and deletion errors:

```text
image -> quantized AE latent -> DNA strands -> noisy DNA strands
      -> DLRT + CRC/MLLV recovery -> AE reconstruction
```

The repository includes pretrained AE and DLRT checkpoints and the 7-nt byte codebook. **You can run the image demonstration without downloading the training dataset or retraining either model.** Run all commands from the repository root.

## Repository contents

| Path | Purpose |
| --- | --- |
| `models/`, `core/`, `utils/` | Models, training, DNA coding and recovery, metrics |
| `scripts/` | Command-line stages of the pipeline |
| `tests/` | Small synthetic pipeline and unit tests |
| `data/splits/` | Train/validation/test source-row manifests; dataset images are not included |
| `examples/Confucius.png` | Independent demonstration image, outside the dataset splits |
| `outputs/checkpoints/AE.pth` | Published AE checkpoint |
| `outputs/checkpoints/DLRT.pth` | Published DLRT checkpoint |
| `outputs/codebook/7nt_byte_codebook.json` | Required DNA byte codebook |

Generated latent caches, training experiments, and reconstruction runs under `outputs/` are ignored by Git. The other PNGs in `examples/` are illustrative reconstructions, not dataset samples.

## Installation

Use an isolated Python environment. Python 3.12 was used for local development; clean-environment installation and GPU behavior should be checked on the target machine. The package versions are listed in `requirements.txt`. For a CUDA/ROCm server, choose the PyTorch build appropriate for that machine using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/); `requirements.txt` does not specify a GPU wheel index.

Linux (Bash):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

The tests exercise code connectivity with small synthetic inputs; they do not establish the image quality of the published checkpoints. The full reconstruction is computationally heavier than the tests. The three PyTorch-based inference stages accept `--cpu` if GPU use must be disabled.

## Reconstruct the example with pretrained checkpoints

The following example uses `examples/Confucius.png`, a 5% total DNA error rate (2.5% substitutions, 1.25% insertions, 1.25% deletions), random seed 42, Top-k 8, and beam size 256. Each invocation creates a new `outputs/runs/<timestamp>/` directory; all intermediate results and the reconstructed PNG remain together. Run the whole block from the repository root and stop if a stage fails.

Linux (Bash):

```bash
set -e
run_dir="outputs/runs/$(date +%Y%m%d_%H%M%S_%N)"
mkdir -p outputs/runs
mkdir "$run_dir"

python scripts/encode_image_to_dna.py --image examples/Confucius.png --ae-checkpoint outputs/checkpoints/AE.pth --output-dir "$run_dir/encoded_dna" --name Confucius_5pct
python scripts/inject_dna_errors.py --input-jsonl "$run_dir/encoded_dna/Confucius_5pct_dna.jsonl" --sub-rate 0.025 --ins-rate 0.0125 --del-rate 0.0125 --seed 42 --output-dir "$run_dir/corrupted_dna" --name Confucius_5pct
python scripts/correct_dna_to_latent.py --input-jsonl "$run_dir/corrupted_dna/Confucius_5pct_corrupted.jsonl" --dlrt-checkpoint outputs/checkpoints/DLRT.pth --top-k 8 --beam-size 256 --output-dir "$run_dir/corrected_latents" --name Confucius_5pct
python scripts/reconstruct_image.py --latent-pt "$run_dir/corrected_latents/Confucius_5pct_corrected_latents.pt" --ae-checkpoint outputs/checkpoints/AE.pth --latent-key top1_completion_latent --reference-image examples/Confucius.png --output-dir "$run_dir/reconstructions" --name Confucius_5pct

echo "Run directory: $run_dir"
```

Windows (PowerShell):

```powershell
$runDir = Join-Path 'outputs/runs' (Get-Date -Format 'yyyyMMdd_HHmmss_fff')
New-Item -ItemType Directory -Path 'outputs/runs' -Force | Out-Null
New-Item -ItemType Directory -Path $runDir -ErrorAction Stop | Out-Null

python scripts/encode_image_to_dna.py --image examples/Confucius.png --ae-checkpoint outputs/checkpoints/AE.pth --output-dir "$runDir/encoded_dna" --name Confucius_5pct
if ($LASTEXITCODE -ne 0) { throw 'DNA encoding failed' }
python scripts/inject_dna_errors.py --input-jsonl "$runDir/encoded_dna/Confucius_5pct_dna.jsonl" --sub-rate 0.025 --ins-rate 0.0125 --del-rate 0.0125 --seed 42 --output-dir "$runDir/corrupted_dna" --name Confucius_5pct
if ($LASTEXITCODE -ne 0) { throw 'Error injection failed' }
python scripts/correct_dna_to_latent.py --input-jsonl "$runDir/corrupted_dna/Confucius_5pct_corrupted.jsonl" --dlrt-checkpoint outputs/checkpoints/DLRT.pth --top-k 8 --beam-size 256 --output-dir "$runDir/corrected_latents" --name Confucius_5pct
if ($LASTEXITCODE -ne 0) { throw 'DNA correction failed' }
python scripts/reconstruct_image.py --latent-pt "$runDir/corrected_latents/Confucius_5pct_corrected_latents.pt" --ae-checkpoint outputs/checkpoints/AE.pth --latent-key top1_completion_latent --reference-image examples/Confucius.png --output-dir "$runDir/reconstructions" --name Confucius_5pct
if ($LASTEXITCODE -ne 0) { throw 'Image reconstruction failed' }

Write-Host "Run directory: $runDir"
```

The principal files in that run are:

```text
outputs/runs/<timestamp>/
  encoded_dna/Confucius_5pct_dna.jsonl
  corrupted_dna/Confucius_5pct_corrupted.jsonl
  corrected_latents/Confucius_5pct_corrected_latents.pt
  corrected_latents/Confucius_5pct_correction_metrics.json
  reconstructions/Confucius_5pct_recon.png
  reconstructions/Confucius_5pct_recon_metrics.json
```

The reconstruction metrics JSON reports `mse`, `psnr`, `ssim`, and `ms_ssim` when a reference image is supplied. MSE is on the 0–255 pixel scale (`normalized MSE × 255²`); PSNR is in dB. Metrics are computed on the decoder output before it is rounded to 8-bit PNG pixels. The default `top1_completion_latent` combines CRC-guided Top-k recovery with MLLV completion for unresolved strands. `recovered_latent` instead leaves unresolved positions as zero-valued erasures.

## Prepare the paper dataset

Dataset images are not committed. `data/splits/train.txt`, `val.txt`, and `test.txt` identify 4,346 / 543 / 542 source rows from [asahi417/wikiart-face](https://huggingface.co/datasets/asahi417/wikiart-face). The preparation script uses the pinned revision `5968b588b5e2da423bcf2683724b54751392e443`, applies EXIF orientation, converts to RGB, and **directly resizes** each selected image to 512 × 512 pixels with Lanczos resampling. It does not preserve the original aspect ratio.

```bash
python scripts/prepare_wikiart_face.py
```

This creates `data/train/`, `data/val/`, and `data/test/`. See [data/README.md](data/README.md) for the split format. The exact upstream revision used when the included checkpoints were originally trained cannot be established from the available project records; the pinned revision specifies future downloads.

## Train new models (optional)

The example below saves newly trained models under `outputs/experiments/` so it does not overwrite the published `AE.pth` or `DLRT.pth`. AE training uses 256 × 256 crops from the prepared 512 × 512 images. DLRT training uses cached AE latents from the train and validation splits; the test cache is generated for evaluation.

```bash
python train_AE.py --data-root data --epochs 100 --batch-size 16 --patch-size 256 --seed 42 --checkpoint outputs/experiments/ae/AE.pth --metrics-log outputs/experiments/ae/metrics.jsonl
python scripts/cache_latents.py --data-root data --checkpoint outputs/experiments/ae/AE.pth --out-dir outputs/experiments/latent_cache --splits train val test
python train_DLRT.py --cache-dir outputs/experiments/latent_cache --epochs 50 --samples-per-epoch 500000 --val-samples 100000 --batch-size 1024 --seed 42 --output outputs/experiments/dlrt/DLRT.pth --metrics-log outputs/experiments/dlrt/metrics.jsonl
```

These training commands are resource-intensive and use the current code defaults for parameters not shown. For all available options, run `python train_AE.py --help`, `python scripts/cache_latents.py --help`, or `python train_DLRT.py --help`. The included checkpoints are ready for the earlier demonstration without this training step.

## Evaluate the test split

After dataset preparation, `evaluate_decoding.py` can evaluate the pretrained checkpoints on the test split. A precomputed test latent cache is optional; if absent, the script encodes the test images with the AE. First check one image at 1% error:

```bash
python evaluate_decoding.py --data-root data --split test --checkpoint outputs/checkpoints/DLRT.pth --ae-checkpoint outputs/checkpoints/AE.pth --total-error-rates 0.01 --top-k 8 --beam-size 256 --seed 42 --all-images --max-images 1 --output-jsonl outputs/evaluation/smoke_images.jsonl --output-csv outputs/evaluation/smoke_summary.csv
```

For the complete 5 × 5 rate/Top-k sweep across the test split, which is substantially slower:

```bash
python evaluate_decoding.py --data-root data --split test --checkpoint outputs/checkpoints/DLRT.pth --ae-checkpoint outputs/checkpoints/AE.pth --total-error-rates 0.01,0.02,0.03,0.04,0.05 --top-k-values 1,2,4,8,16 --beam-size 256 --seed 42 --all-images --output-jsonl outputs/evaluation/full_images.jsonl --output-csv outputs/evaluation/full_summary.csv
```

Use distinct `--output-jsonl` and `--output-csv` paths for subsequent evaluations because each command replaces prior results at those paths. Add `--reconstruction-dir <directory>` if image files are also needed. The quick-start reconstruction above already writes its image and metrics to a timestamped run directory.
