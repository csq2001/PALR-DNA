from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_project_root

add_project_root()

from utils.dna_channel import mutate_dna


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inject random DNA base errors into encoded DNA strands.")
    parser.add_argument("--input-jsonl", required=True, help="Clean DNA JSONL from encode_image_to_dna.py.")
    parser.add_argument("--output-dir", default="outputs/corrupted_dna")
    parser.add_argument("--sub-rate", type=float, default=0.025)
    parser.add_argument("--ins-rate", type=float, default=0.0125)
    parser.add_argument("--del-rate", type=float, default=0.0125)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="", help="Optional output stem.")
    return parser.parse_args()


def rate_label(total: float) -> str:
    return f"rate{int(round(total * 1000)):03d}"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_rate = args.sub_rate + args.ins_rate + args.del_rate
    stem = args.name or f"{input_path.stem}_{rate_label(total_rate)}"
    output_path = output_dir / f"{stem}_corrupted.jsonl"

    count = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            position_id = int(record["position_id"])
            noisy_dna, errors = mutate_dna(
                record["dna"],
                sub_rate=args.sub_rate,
                ins_rate=args.ins_rate,
                del_rate=args.del_rate,
                seed=args.seed + position_id * 1009,
            )
            record.update(
                {
                    "sub_rate": args.sub_rate,
                    "ins_rate": args.ins_rate,
                    "del_rate": args.del_rate,
                    "noisy_dna": noisy_dna,
                    "errors": errors,
                }
            )
            dst.write(json.dumps(record, sort_keys=True) + "\n")
            count += 1

    print(f"corrupted chains={count} input={input_path} output={output_path}")


if __name__ == "__main__":
    main()

