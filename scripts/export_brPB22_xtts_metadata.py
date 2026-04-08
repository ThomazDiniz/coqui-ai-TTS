#!/usr/bin/env python3
"""Build Coqui-format metadata_train.csv / metadata_eval.csv from brPB22_g1bF01_char metadata.csv.

Usage (from repo root):
  python scripts/export_brPB22_xtts_metadata.py
  python scripts/export_brPB22_xtts_metadata.py --dataset data/brPB22_g1bF01_char --eval-fraction 0.15
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/brPB22_g1bF01_char"),
        help="Folder containing metadata.csv and wavs/",
    )
    parser.add_argument("--eval-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speaker-name", type=str, default="brPB22")
    args = parser.parse_args()

    meta_path = args.dataset / "metadata.csv"
    if not meta_path.is_file():
        raise SystemExit(f"Missing {meta_path}")

    rows: list[dict[str, str]] = []
    for line in meta_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        fn, text = line.split("|", 1)
        fn = fn.strip()
        text = text.strip()
        if not fn.endswith(".wav"):
            fn = f"{fn}.wav"
        rows.append(
            {
                "audio_file": f"wavs/{fn}",
                "text": text,
                "speaker_name": args.speaker_name,
            }
        )

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_eval = max(1, int(len(rows) * args.eval_fraction))
    eval_rows = sorted(rows[:n_eval], key=lambda r: r["audio_file"])
    train_rows = sorted(rows[n_eval:], key=lambda r: r["audio_file"])

    fieldnames = ["audio_file", "text", "speaker_name"]
    train_out = args.dataset / "metadata_train.csv"
    eval_out = args.dataset / "metadata_eval.csv"
    for path, data in ((train_out, train_rows), (eval_out, eval_rows)):
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="|")
            w.writeheader()
            w.writerows(data)

    print(f"Wrote {train_out} ({len(train_rows)} rows)")
    print(f"Wrote {eval_out} ({len(eval_rows)} rows)")


if __name__ == "__main__":
    main()
