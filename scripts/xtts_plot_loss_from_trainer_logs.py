from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Example:
#   --> TIME: 2026-04-08 08:51:53 -- STEP: 0/34 -- GLOBAL_STEP: 0
STEP_HDR_RE = re.compile(r"GLOBAL_STEP:\s*(\d+)")

# Eval batches (coqui-tts-trainer ConsoleLogger.print_eval_step): only "--> STEP: N"
# — not "STEP: N/M" and no GLOBAL_STEP on that line.
EVAL_SUBSTEP_RE = re.compile(r"-->\s*STEP:\s*(\d+)\s*$")

# Resumo de eval por época (print_epoch_end): sem GLOBAL_STEP antes das métricas.
EVAL_SUMMARY_HDR_RE = re.compile(r"EVAL\s+PERFORMANCE", re.IGNORECASE)

# Example:
#     | > loss: 2.4136147499084473  (2.4136147499084473)
#     | > avg_loss: 1.23 (+0.0)   # após strip ANSI
METRIC_RE = re.compile(r"^\s*\|\s*>\s*([A-Za-z0-9_]+)\s*:\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", re.IGNORECASE)
LOG_TIMESTAMP_RE = re.compile(
    r"(?:-->\s*)?TIME:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})",
)


@dataclass(frozen=True)
class Point:
    global_step: int
    loss: float | None = None
    loss_text_ce: float | None = None
    loss_mel_ce: float | None = None
    current_lr: float | None = None
    epoch: int | None = None
    phase: str = "train"  # train|eval
    step_time: float | None = None
    log_timestamp: str | None = None  # wall clock from trainer line, if present


def _pick_main_loss(metrics: dict[str, float]) -> float | None:
    """Train usa `loss`; resumo de eval por época usa `avg_*` (ex.: avg_loss)."""
    for k in ("loss", "avg_loss", "model_loss"):
        if k in metrics:
            return metrics[k]
    return None


def _clean_line(s: str) -> str:
    return ANSI_RE.sub("", s).rstrip("\n")


def parse_trainer_log(path: Path) -> list[Point]:
    points: list[Point] = []
    cur_epoch: int | None = None
    cur_phase = "train"
    cur_step: int | None = None
    cur_header_ts: str | None = None
    cur_metrics: dict[str, float] = {}
    last_global_step: int = 0

    def flush():
        nonlocal cur_step, cur_metrics, cur_phase, cur_epoch, cur_header_ts
        if cur_step is None:
            cur_metrics = {}
            return
        if not cur_metrics:
            cur_step = None
            return
        st = cur_metrics.get("step_time")
        main_loss = _pick_main_loss(cur_metrics)
        points.append(
            Point(
                global_step=cur_step,
                loss=main_loss,
                loss_text_ce=cur_metrics.get("loss_text_ce") or cur_metrics.get("avg_loss_text_ce"),
                loss_mel_ce=cur_metrics.get("loss_mel_ce") or cur_metrics.get("avg_loss_mel_ce"),
                current_lr=cur_metrics.get("current_lr"),
                epoch=cur_epoch,
                phase=cur_phase,
                step_time=st,
                log_timestamp=cur_header_ts,
            )
        )
        cur_step = None
        cur_metrics = {}
        cur_header_ts = None

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

            # Resumo eval por época (avg_loss, etc.) — sem GLOBAL_STEP neste bloco
            if EVAL_SUMMARY_HDR_RE.search(line):
                flush()
                cur_phase = "eval"
                cur_step = last_global_step
                continue

            # New step header (train)
            m = STEP_HDR_RE.search(line)
            if m:
                flush()
                cur_step = int(m.group(1))
                last_global_step = cur_step
                tm = LOG_TIMESTAMP_RE.search(line)
                cur_header_ts = tm.group(1) if tm else None
                continue

            # Cabeçalho de batch na avaliação: "--> STEP: k" (sem GLOBAL_STEP)
            em = EVAL_SUBSTEP_RE.search(line)
            if em and cur_phase == "eval":
                flush()
                cur_step = last_global_step
                tm = LOG_TIMESTAMP_RE.search(line)
                cur_header_ts = tm.group(1) if tm else None
                continue

            # Metric lines under a step header
            m = METRIC_RE.match(line)
            if m:
                k = m.group(1).lower()
                try:
                    v = float(m.group(2))
                except ValueError:
                    continue
                if cur_step is None and cur_phase == "eval":
                    cur_step = last_global_step
                if cur_step is not None:
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


def _aggregate_mean_loss_by_epoch(points: list[Point], phase: str) -> tuple[list[int], list[float]]:
    """Média da loss por número de época (apenas pontos com epoch definido)."""
    bucket: dict[int, list[float]] = defaultdict(list)
    for p in points:
        if p.phase != phase or p.loss is None or p.epoch is None:
            continue
        bucket[p.epoch].append(p.loss)
    if not bucket:
        return [], []
    epochs = sorted(bucket.keys())
    means = [sum(bucket[e]) / len(bucket[e]) for e in epochs]
    return epochs, means


def plot_loss_by_epoch(points: list[Point], out_png: Path, title: str) -> bool:
    """Gráfico loss média por época (train e, se houver, eval)."""
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    te, tm = _aggregate_mean_loss_by_epoch(points, "train")
    ee, em = _aggregate_mean_loss_by_epoch(points, "eval")
    if not te and not ee:
        return False

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    if te:
        ax.plot(te, tm, color="#2563eb", linewidth=1.8, marker="o", markersize=4, label="train (média/época)")
    if ee:
        ax.plot(ee, em, color="#dc2626", linewidth=1.8, marker="s", markersize=4, label="eval (média/época)")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss (média dos steps na época)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png.is_file()


def plot_loss_step_and_epoch(points: list[Point], out_png: Path, title_prefix: str) -> bool:
    """Um PNG com dois painéis: loss vs global step e loss média vs época."""
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    train = [p for p in points if p.phase == "train" and p.loss is not None]
    evalp = [p for p in points if p.phase == "eval" and p.loss is not None]
    te, tm = _aggregate_mean_loss_by_epoch(points, "train")
    ee, em = _aggregate_mean_loss_by_epoch(points, "eval")

    if not train and not evalp and not te and not ee:
        return False

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 9), dpi=120, sharex=False)

    if train:
        ax0.plot([p.global_step for p in train], [p.loss for p in train], color="#2563eb", linewidth=1.2, label="train")
    if evalp:
        ax0.plot([p.global_step for p in evalp], [p.loss for p in evalp], color="#dc2626", linewidth=1.2, label="eval")
    ax0.set_xlabel("Global step")
    ax0.set_ylabel("Loss")
    ax0.set_title(f"{title_prefix}\nPor step")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper right")

    if te:
        ax1.plot(te, tm, color="#2563eb", linewidth=1.8, marker="o", markersize=4, label="train (média)")
    if ee:
        ax1.plot(ee, em, color="#dc2626", linewidth=1.8, marker="s", markersize=4, label="eval (média)")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss (média na época)")
    ax1.set_title("Por época")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png.is_file()


def plot_step_time_by_global_step(points: list[Point], out_png: Path, title: str) -> bool:
    """Gráfico step_time (s) vs global step — útil para relatar desempenho por iteração."""
    train = [p for p in points if p.phase == "train" and p.step_time is not None]
    if len(train) < 2:
        return False
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=120)
    ax.plot(
        [p.global_step for p in train],
        [p.step_time for p in train],
        color="#059669",
        linewidth=1.2,
    )
    ax.set_xlabel("Global step")
    ax.set_ylabel("step_time (s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png.is_file()


TRAINER_CLOCK_RE = re.compile(r"-->\s*TIME:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})")
TRAINER_EPOCH_HDR_RE = re.compile(r">\s*EPOCH:\s*(\d+)")


def parse_epoch_clock_spans_seconds(log_path: Path) -> dict[int, float]:
    """Extensão temporal (relógio de parede) por época: max(TIME) − min(TIME) entre linhas do log dessa época.

    Aproxima o tempo de parede da época quando o trainer imprime TIME em cada step logado.
    """
    cur_epoch: int | None = None
    times_by_epoch: dict[int, list[datetime]] = defaultdict(list)

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = _clean_line(raw)
            if not line.strip():
                continue
            em = TRAINER_EPOCH_HDR_RE.search(line)
            if em:
                cur_epoch = int(em.group(1))
                continue
            cm = TRAINER_CLOCK_RE.search(line)
            if cm and cur_epoch is not None:
                try:
                    dt = datetime.strptime(cm.group(1), "%Y-%m-%d %H:%M:%S")
                    times_by_epoch[cur_epoch].append(dt)
                except ValueError:
                    continue

    spans: dict[int, float] = {}
    for e, tlist in times_by_epoch.items():
        if len(tlist) >= 2:
            spans[e] = (max(tlist) - min(tlist)).total_seconds()
        elif len(tlist) == 1:
            spans[e] = 0.0
    return spans


def sum_step_time_by_epoch(points: list[Point], phase: str = "train") -> dict[int, float]:
    """Soma dos `step_time` reportados pelo trainer por época (tempo acumulado dos steps logados)."""
    bucket: dict[int, list[float]] = defaultdict(list)
    for p in points:
        if p.phase != phase or p.epoch is None or p.step_time is None:
            continue
        bucket[p.epoch].append(p.step_time)
    return {e: sum(v) for e, v in bucket.items()}


def loss_train_eval_statistics(points: list[Point]) -> dict[str, float | int | None]:
    """Estatísticas simples para relatório (último, min, max por fase)."""
    train_losses = [p.loss for p in points if p.phase == "train" and p.loss is not None]
    eval_losses = [p.loss for p in points if p.phase == "eval" and p.loss is not None]

    def stats(vals: list[float]) -> dict[str, float | None]:
        if not vals:
            return {"last": None, "min": None, "max": None, "n": 0}
        return {"last": vals[-1], "min": min(vals), "max": max(vals), "n": len(vals)}

    out: dict[str, float | int | None] = {
        "train_points": len(train_losses),
        "eval_points": len(eval_losses),
    }
    ts = stats(train_losses)
    es = stats(eval_losses)
    for k, v in ts.items():
        out[f"train_loss_{k}"] = v
    for k, v in es.items():
        out[f"eval_loss_{k}"] = v
    return out


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

