#!/usr/bin/env python3
"""Download, split, and resize the WikiArt Face images used in the paper."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath

from PIL import Image, ImageOps


DATASET_NAME = "asahi417/wikiart-face"
DATASET_REVISION = "5968b588b5e2da423bcf2683724b54751392e443"
SOURCE_SPLIT = "test"
OUTPUT_SIZE = (512, 512)
PAPER_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download WikiArt Face, select the paper splits, and save every "
            "image as a 512 x 512 RGB PNG."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_assignments(data_root: Path) -> dict[int, Path]:
    assignments: dict[int, Path] = {}
    for split in PAPER_SPLITS:
        list_path = data_root / "splits" / f"{split}.txt"
        if not list_path.is_file():
            raise FileNotFoundError(f"split list not found: {list_path}")
        for line in list_path.read_text(encoding="utf-8").splitlines():
            relative = PurePosixPath(line.strip())
            if not relative.parts:
                continue
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe path in {list_path}: {line!r}")
            if relative.parts[0] != split or len(relative.parts) != 2:
                raise ValueError(f"invalid {split} entry in {list_path}: {line!r}")
            if relative.suffix.lower() != ".png" or not relative.stem.isdigit():
                raise ValueError(f"expected a numeric PNG name in {list_path}: {line!r}")
            source_index = int(relative.stem)
            if source_index in assignments:
                raise ValueError(f"source row {source_index} occurs in multiple splits")
            assignments[source_index] = Path(*relative.parts)
    if not assignments:
        raise ValueError("the split lists contain no images")
    return assignments


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    assignments = load_assignments(data_root)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'datasets'. Install it with: pip install datasets"
        ) from exc

    source = load_dataset(DATASET_NAME, split=SOURCE_SPLIT, revision=args.revision)
    largest_index = max(assignments)
    if largest_index >= len(source):
        raise IndexError(
            f"split lists require source row {largest_index}, but the dataset "
            f"contains only {len(source)} rows"
        )

    written = 0
    skipped = 0
    for source_index, relative_path in sorted(assignments.items()):
        output_path = data_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not args.overwrite:
            with Image.open(output_path) as existing:
                if existing.size != OUTPUT_SIZE or existing.mode != "RGB":
                    raise ValueError(
                        f"existing image is not 512 x 512 RGB: {output_path}; "
                        "rerun with --overwrite"
                    )
            skipped += 1
            continue

        image = source[source_index]["image"]
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize(OUTPUT_SIZE, resample=Image.Resampling.LANCZOS)
        image.save(output_path, format="PNG")
        written += 1

        if written % 100 == 0:
            print(f"written={written} skipped={skipped}", flush=True)

    print(
        f"complete: selected={len(assignments)} written={written} skipped={skipped} "
        f"size={OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]} mode=RGB",
        flush=True,
    )


if __name__ == "__main__":
    main()
