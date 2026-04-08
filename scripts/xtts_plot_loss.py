from __future__ import annotations

import argparse
import os
import sys

# Allow running from source tree without requiring editable install.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from TTS.demos.xtts_ft_demo.utils.plot_loss import save_training_loss_plot  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Generate XTTS fine-tune loss PNG from TensorBoard event logs.")
    p.add_argument(
        "--run_dir",
        required=True,
        help="Training run directory (contains events.out.tfevents.*), e.g. data/xtts-work/run/training/GPT_XTTS_FT-...",
    )
    p.add_argument(
        "--out_png",
        default="training_loss.png",
        help="Output PNG name (default: training_loss.png in run_dir unless absolute path).",
    )
    args = p.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    out_png = args.out_png
    if not os.path.isabs(out_png):
        out_png = os.path.join(run_dir, out_png)

    ok = save_training_loss_plot(run_dir, out_png)
    if not ok:
        print(f"[xtts_plot_loss] failed (no PNG written) run_dir={run_dir}", file=sys.stderr)
        return 2
    print(f"[xtts_plot_loss] wrote: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

