# Dataset

The paper uses images from the
[asahi417/wikiart-face](https://huggingface.co/datasets/asahi417/wikiart-face)
dataset. Dataset images are intentionally excluded from the Git repository;
refer to the upstream dataset card for access and usage terms.

The preparation script pins upstream revision
`5968b588b5e2da423bcf2683724b54751392e443`, the `main` HEAD checked on
2026-09-16. This fixes the source version used by future downloads; the
revision used for the already trained checkpoints cannot be established from
the current project files alone. Use `--revision` to select another upstream
commit explicitly.

Install the data-download dependency and run the preparation script from the
project root:

```bash
pip install datasets
python scripts/prepare_wikiart_face.py
```

The script selects exactly the rows recorded by the split lists, converts each
image to RGB, directly resizes it to 512 x 512 pixels with Lanczos resampling,
and saves it as PNG. It creates the following layout:

```text
data/
  train/
  val/
  test/
  splits/
    train.txt
    val.txt
    test.txt
```

Each split-list line is a POSIX-style path relative to `data/`. The lists are
the authoritative record of the images used in the paper:

- `train.txt`: 4346 images
- `val.txt`: 543 images
- `test.txt`: 542 images

`examples/Confucius.png` is an independent demonstration image and is not
part of any dataset split.
