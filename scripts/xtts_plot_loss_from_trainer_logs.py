from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Example:
#   --> TIME: 2026-04-08 08:51:53 -- STEP: 0/34 -- GLOBAL_STEP: 0
STEP_HDR_RE = re.compile(r"GLOBAL_STEP:\s*(\d+)")

# Example:
#     | > loss: 2.4136147499084473  (2.4136147499084473)
METRIC_RE = re.compile(r"^\s*\|\s*>\s*([A-Za-z0-9_]+)\s*:\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", re.IGNORECASE)


@dataclass(frozen=True)
class Point:
    global_step: int
    loss: float | None = None
    loss_text_ce: float | None = None
    loss_mel_ce: float | None = None
    current_lr: float | None = None
    epoch: int | None = None
    phase: str = "train"  # train|eval


def _clean_line(s: str) -> str:
    return ANSI_RE.sub("", s).rstrip("\n")


def parse_trainer_log(path: Path) -> list[Point]:
    points: list[Point] = []
    cur_epoch: int | None = None
    cur_phase = "train"
    cur_step: int | None = None
    cur_metrics: dict[str, float] = {}

    def flush():
        nonlocal cur_step, cur_metrics, cur_phase, cur_epoch
        if cur_step is None:
            cur_metrics = {}
            return
        if not cur_metrics:
            cur_step = None
            return
        points.append(
            Point(
                global_step=cur_step,
                loss=cur_metrics.get("loss"),
                loss_text_ce=cur_metrics.get("loss_text_ce"),
                loss_mel_ce=cur_metrics.get("loss_mel_ce"),
                current_lr=cur_metrics.get("current_lr"),
                epoch=cur_epoch,
                phase=cur_phase,
            )
        )
        cur_step = None
        cur_metrics = {}

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = _clean_line(raw)
            if not line.strip():
                continue

            # Epoch header
            if "> EPOCH:" in line:
                flush()
                m = re.search(r"> EPOCH:\s*(\d+)", line)
                if m:
                    cur_epoch = int(m.group(1))
                continue

            # Phase changes
            if "> EVALUATION" in line:
                flush()
                cur_phase = "eval"
                continue
            if "> TRAINING" in line:
                flush()
                cur_phase = "train"
                continue

            # New step header
            m = STEP_HDR_RE.search(line)
            if m:
                flush()
                cur_step = int(m.group(1))
                continue

            # Metric lines under a step header
            m = METRIC_RE.match(line)
            if m and cur_step is not None:
                k = m.group(1).lower()
                try:
                    v = float(m.group(2))
                except ValueError:
                    continue
                cur_metrics[k] = v
                continue

    flush()
    return points


def _ensure_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as e:
        raise RuntimeError("matplotlib is required to plot. Install it in your env.") from e


def plot_points(points: list[Point], out_png: Path, title: str) -> bool:
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    train = [p for p in points if p.phase == "train" and p.loss is not None]
    evalp = [p for p in points if p.phase == "eval" and p.loss is not None]

    if not train and not evalp:
        return False

    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    if train:
        ax.plot([p.global_step for p in train], [p.loss for p in train], color="#2563eb", linewidth=1.4, label="train loss")
    if evalp:
        ax.plot([p.global_step for p in evalp], [p.loss for p in evalp], color="#dc2626", linewidth=1.2, label="eval avg_loss")
    ax.set_xlabel("Global step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png.is_file()


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot XTTS fine-tuning loss from trainer_0_log.txt files.")
    ap.add_argument(
        "--training_root",
        default=str(Path("data") / "xtts-work" / "run" / "training"),
        help="Root directory containing run folders (default: data/xtts-work/run/training).",
    )
    ap.add_argument(
        "--pattern",
        default="trainer_0_log.txt",
        help="Filename to search for under training_root.",
    )
    ap.add_argument(
        "--write_root",
        default="xtts_training_loss_from_trainer_logs.png",
        help="Also write a copy in repo root with this filename.",
    )
    args = ap.parse_args()

    training_root = Path(args.training_root).resolve()
    if not training_root.exists():
        print(f"[xtts_plot_loss_from_trainer_logs] training_root not found: {training_root}", file=sys.stderr)
        return 2

    logs = sorted(training_root.rglob(args.pattern))
    if not logs:
        print(f"[xtts_plot_loss_from_trainer_logs] no {args.pattern} under {training_root}", file=sys.stderr)
        return 3

    wrote_any = False
    latest_png: Path | None = None
    latest_mtime = -1.0

    for log_path in logs:
        run_dir = log_path.parent
        points = parse_trainer_log(log_path)
        out_png = run_dir / "training_loss_from_trainer_log.png"
        title = f"XTTS GPT fine-tuning — loss (from trainer log)\n{run_dir.name}"
        ok = plot_points(points, out_png, title=title)
        print(f"[xtts_plot_loss_from_trainer_logs] {run_dir.name}: points={len(points)} wrote={ok} -> {out_png}")
        if ok:
            wrote_any = True
            m = out_png.stat().st_mtime
            if m > latest_mtime:
                latest_mtime = m
                latest_png = out_png

    if wrote_any and latest_png is not None:
        root_out = Path(args.write_root).resolve()
        try:
            # Copy latest to repo root for convenience
            root_out.write_bytes(latest_png.read_bytes())
            print(f"[xtts_plot_loss_from_trainer_logs] wrote root copy: {root_out}")
        except Exception as e:
            print(f"[xtts_plot_loss_from_trainer_logs] failed to write root copy: {e}", file=sys.stderr)

    if not wrote_any:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

