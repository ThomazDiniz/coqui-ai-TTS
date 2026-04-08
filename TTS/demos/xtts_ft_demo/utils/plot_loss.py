"""Generate a loss plot from TensorBoard logs produced by coqui-tts-trainer."""

from __future__ import annotations

import logging
import os

_LOG = logging.getLogger("xtts_ft.plot_loss")


def save_training_loss_plot(log_dir: str, out_png: str) -> bool:
    """Read scalar `TrainIterStats/loss` from TensorBoard events under ``log_dir`` and save a PNG.

    Returns True if the file was written.
    """
    _LOG.info("[PLOT] begin log_dir=%s out_png=%s", log_dir, out_png)
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        _LOG.warning("[PLOT] tensorboard not available; skip loss PNG")
        return False

    log_dir = os.path.abspath(log_dir)
    if not os.path.isdir(log_dir):
        _LOG.warning("[PLOT] log_dir is not a directory: %s", log_dir)
        return False

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _LOG.warning("[PLOT] matplotlib not available; skip loss PNG")
        return False

    ea = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    try:
        ea.Reload()
    except Exception:
        _LOG.exception("[PLOT] EventAccumulator.Reload failed for %s", log_dir)
        return False

    scalar_tags = ea.Tags().get("scalars", [])
    _LOG.info("[PLOT] scalar tag count=%s sample=%s", len(scalar_tags), scalar_tags[:12])
    primary_tag = None
    for candidate in ("TrainIterStats/loss", "TrainIterStats/loss_mel_ce"):
        if candidate in scalar_tags:
            primary_tag = candidate
            break
    if primary_tag is None:
        # any TrainIterStats loss-like tag
        for t in scalar_tags:
            if t.startswith("TrainIterStats/") and "loss" in t.lower():
                primary_tag = t
                break
    if primary_tag is None:
        _LOG.warning("[PLOT] no TrainIterStats loss tag found among scalars")
        return False

    events = ea.Scalars(primary_tag)
    if not events:
        _LOG.warning("[PLOT] no scalar events for tag %s", primary_tag)
        return False

    steps = [e.step for e in events]
    values = [e.value for e in events]
    _LOG.info("[PLOT] using tag=%s points=%s step_range=[%s,%s]", primary_tag, len(steps), min(steps), max(steps))

    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    ax.plot(steps, values, color="#2563eb", linewidth=1.2, label=primary_tag.split("/")[-1])
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("XTTS GPT fine-tuning — training loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    eval_tag = None
    for candidate in ("EvalStats/loss",):
        if candidate in scalar_tags:
            eval_tag = candidate
            break
    if eval_tag is None:
        for t in scalar_tags:
            if t.startswith("EvalStats/") and "loss" in t.lower():
                eval_tag = t
                break
    if eval_tag:
        ev = ea.Scalars(eval_tag)
        if ev:
            _LOG.info("[PLOT] overlay eval tag=%s points=%s", eval_tag, len(ev))
            ax.plot([e.step for e in ev], [e.value for e in ev], color="#dc2626", linewidth=1.0, alpha=0.85, label="eval")
            ax.legend(loc="upper right")
    else:
        _LOG.info("[PLOT] no eval loss tag found (optional)")

    out_abs = os.path.abspath(out_png)
    parent = os.path.dirname(out_abs)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    ok = os.path.isfile(out_png)
    _LOG.info("[PLOT] wrote=%s bytes=%s", ok, os.path.getsize(out_png) if ok else 0)
    return ok
