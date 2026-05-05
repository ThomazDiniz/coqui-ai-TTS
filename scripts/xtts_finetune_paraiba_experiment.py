#!/usr/bin/env python3
"""XTTS GPT fine-tuning experiment runner with research artifacts (loss plots, timings, machine info).

Uses the same training path as the Gradio demo: ``TTS.demos.xtts_ft_demo.utils.gpt_train.train_gpt``.
Base weights are the official multilingual XTTS-v2 checkpoint. **Portuguese (Brazilian fine-tune)** uses
language code ``pt`` — the model has a single Portuguese slot (there is no separate ``pt-br`` id).
Brazilian accent and wording come **from your audio and transcripts**, not from a different code.

Stages (``--stage``):

- ``prep`` — apenas prepara ``metadata_train.csv`` / ``metadata_eval.csv`` (Whisper ou metadata) e grava
  ``prepared_manifest.json`` em ``--experiment-dir``.
- ``train`` — lê o manifest e executa o fine-tune (use o mesmo ``--experiment-dir`` da etapa prep).
- ``all`` — prep + train numa única execução.

Examples::

  rem Default: repo data/all_char + wavs/ (metadata_train/eval prontos evita Whisper)
  python scripts/xtts_finetune_paraiba_experiment.py --epochs 10

  rem Pasta explicita
  python scripts/xtts_finetune_paraiba_experiment.py --data-root "D:\\meus_dados\\all_char" --epochs 10

  # CSVs Coqui ja prontos (pipe, audio_file|text|...):
  python scripts/xtts_finetune_paraiba_experiment.py --data-root "D:\\dados" --skip-dataset-prep
"""

from __future__ import annotations

import argparse
import csv
import io
import importlib.util
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

# Repo root = parent of scripts/
REPO_ROOT = Path(__file__).resolve().parent.parent

_LOG = logging.getLogger("xtts_finetune_paraiba")


def setup_verbose_logging() -> None:
    """Console logs: experiment, TTS, trainer, xtts_ft demo modules."""
    root = logging.getLogger()
    root.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(logging.DEBUG)
    h.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    root.addHandler(h)
    root.setLevel(logging.INFO)
    for name in (
        "xtts_ft",
        "xtts_ft.train",
        "xtts_ft.formatter",
        "xtts_ft.plot_loss",
        "TTS",
        "trainer",
    ):
        logging.getLogger(name).setLevel(logging.INFO)


def log_banner(title: str) -> None:
    bar = "=" * 72
    _LOG.info("%s", bar)
    _LOG.info(" %s", title)
    _LOG.info("%s", bar)
    sys.stdout.flush()


def _normalize_pt_language(code: str, label: str) -> str:
    """Mapeia aliases pt-BR para o unico codigo `pt` aceite pelo XTTS e pelo Whisper."""
    raw = (code or "").strip()
    c = raw.lower().replace("_", "-")
    if c in ("pt", "pt-br") or c.startswith("pt-br"):
        if raw.lower() != XTTS_LANG_PORTUGUESE:
            _LOG.info(
                "%s=%r -> %r (unico codigo PT no XTTS/Whisper; PT-BR pelo conteudo dos teus dados).",
                label,
                raw,
                XTTS_LANG_PORTUGUESE,
            )
        return XTTS_LANG_PORTUGUESE
    if c in ("br", "brazil", "brazilian"):
        _LOG.info("%s=%r -> %r (portugues para o modelo).", label, raw, XTTS_LANG_PORTUGUESE)
        return XTTS_LANG_PORTUGUESE
    return raw


def _log_pt_br_target(prep_lang: str, train_lang: str) -> None:
    _LOG.info(
        "Alvo: portugues brasileiro (PT-BR) — Whisper prep=%s, XTTS train=%s. "
        "O modelo usa o codigo %r para portugues; sotaque e lexico BR vêm dos áudios/textos.",
        prep_lang,
        train_lang,
        XTTS_LANG_PORTUGUESE,
    )
    sys.stdout.flush()


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/d"
    if seconds < 60:
        return f"{seconds:.1f} s"
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m} min {s:.1f} s ({seconds:.0f} s total)"


MANIFEST_NAME = "prepared_manifest.json"

# Pasta padrão no repositório Coqui (sem dependências externas tipo Arrow/F5).
# all_char: wavs em data/all_char/wavs; com metadata_train/eval o prep Whisper nao corre.
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "all_char"

AUDIO_FILE_EXTENSIONS = frozenset({".wav", ".mp3", ".flac"})

# XTTS-v2 e Whisper usam um único código ISO para português: "pt". Não há embedding "pt-br";
# fine-tune para **português brasileiro** = dados (áudio/texto) em PT-BR com estes códigos.
XTTS_LANG_PORTUGUESE = "pt"


@dataclass
class TrainerLogSummary:
    log_path: str
    epoch_lines: list[str]
    time_lines: list[str]
    step_time_samples: list[float]
    approx_epoch_wall_times_s: list[float | None]


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
TIME_LINE_RE = re.compile(
    r"-->\s*TIME:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})"
)
STEP_TIME_RE = re.compile(r"^\s*\|\s*>\s*step_time\s*:\s*([0-9.eE+-]+)", re.IGNORECASE)
EPOCH_LINE_RE = re.compile(r">\s*EPOCH:\s*(\d+)\s*/\s*(\d+)")


def _strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def _parse_trainer_log_summary(log_path: Path) -> TrainerLogSummary:
    epoch_lines: list[str] = []
    time_lines: list[str] = []
    step_times: list[float] = []
    ts_list: list[datetime | None] = []

    if not log_path.is_file():
        return TrainerLogSummary(
            str(log_path), epoch_lines, time_lines, step_times, []
        )

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = _strip_ansi(raw.rstrip("\n"))
            if "> EPOCH:" in line or "EPOCH:" in line:
                if EPOCH_LINE_RE.search(line):
                    epoch_lines.append(line.strip())
            m_t = TIME_LINE_RE.search(line)
            if m_t:
                time_lines.append(line.strip())
                try:
                    ts_list.append(datetime.strptime(m_t.group(1), "%Y-%m-%d %H:%M:%S"))
                except ValueError:
                    ts_list.append(None)
            m_st = STEP_TIME_RE.match(line)
            if m_st:
                try:
                    step_times.append(float(m_st.group(1)))
                except ValueError:
                    pass

    epoch_durations: list[float | None] = []
    if ts_list:
        last_e_ts: datetime | None = None
        for ts in ts_list:
            if ts is None:
                epoch_durations.append(None)
                continue
            if last_e_ts is None:
                epoch_durations.append(None)
            else:
                epoch_durations.append((ts - last_e_ts).total_seconds())
            last_e_ts = ts

    return TrainerLogSummary(
        str(log_path), epoch_lines, time_lines, step_times, epoch_durations
    )


def _collect_machine_info() -> dict:
    import platform

    info: dict = {
        "utc_iso": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": sys.version,
        "processor": platform.processor(),
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_devices"] = []
            for i in range(torch.cuda.device_count()):
                try:
                    p = torch.cuda.get_device_properties(i)
                    info["cuda_devices"].append(
                        {
                            "index": i,
                            "name": p.name,
                            "total_memory_gb": round(p.total_memory / 1024**3, 3),
                            "major": p.major,
                            "minor": p.minor,
                            "multi_processor_count": p.multi_processor_count,
                        }
                    )
                except Exception as e:
                    info["cuda_devices"].append({"index": i, "error": str(e)})
    except Exception as e:
        info["torch_error"] = str(e)

    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / 1024**3, 3)
        info["ram_available_gb"] = round(vm.available / 1024**3, 3)
        info["cpu_logical_count"] = psutil.cpu_count(logical=True)
        info["cpu_physical_count"] = psutil.cpu_count(logical=False)
        try:
            freq = psutil.cpu_freq()
            if freq:
                info["cpu_freq_mhz_current"] = round(freq.current, 1) if freq.current else None
                info["cpu_freq_mhz_max"] = round(freq.max, 1) if freq.max else None
        except Exception:
            pass
    except Exception as e:
        info["psutil_error"] = str(e)

    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        info["nvidia_smi_L"] = (out.stdout or "").strip() or None
        if out.returncode != 0:
            info["nvidia_smi_L_stderr"] = (out.stderr or "").strip() or None
    except Exception as e:
        info["nvidia_smi_error"] = str(e)

    try:
        outq = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_mode",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if outq.returncode == 0 and (outq.stdout or "").strip():
            rows = []
            for ln in (outq.stdout or "").strip().splitlines():
                parts = [x.strip() for x in ln.split(",")]
                if len(parts) >= 3:
                    rows.append(
                        {
                            "name": parts[0],
                            "driver_version": parts[1],
                            "memory_total": parts[2],
                            "compute_mode": parts[3] if len(parts) > 3 else None,
                        }
                    )
            info["nvidia_smi_query_gpu"] = rows
    except Exception as e:
        info["nvidia_smi_query_error"] = str(e)

    return info


def _write_machine_report_thesis(artifacts: Path, info: dict) -> Path:
    """Texto corrido para colar em dissertação (ambiente de execução reprodutível)."""
    lines = [
        "# Ambiente computacional (experimento)",
        "",
        f"- **Data/hora (UTC):** {info.get('utc_iso', 'n/d')}",
        f"- **SO:** {info.get('system', 'n/d')} {info.get('release', '')} ({info.get('machine', '')})",
        f"- **Plataforma:** {info.get('platform', 'n/d')}",
        f"- **CPU (relatório OS):** {info.get('processor') or 'n/d'}",
    ]
    if info.get("cpu_logical_count") is not None:
        lines.append(
            f"- **CPUs lógicas / físicas:** {info.get('cpu_logical_count')} / {info.get('cpu_physical_count', 'n/d')}"
        )
    if info.get("ram_total_gb") is not None:
        lines.append(
            f"- **RAM:** {info['ram_total_gb']} GiB total; "
            f"{info.get('ram_available_gb', 'n/d')} GiB disponível (instante da captura)."
        )
    py = (info.get("python") or "").replace("\n", " ")
    lines.append(f"- **Python:** {py[:500]}{'…' if len(py) > 500 else ''}")
    lines.append(f"- **PyTorch:** {info.get('torch_version', 'n/d')}")
    lines.append(f"- **CUDA (PyTorch):** disponível={info.get('cuda_available')}; versão toolkit={info.get('cuda_version')}")
    if info.get("cuda_devices"):
        for d in info["cuda_devices"]:
            if "name" in d:
                lines.append(
                    f"- **GPU {d.get('index')}:** {d.get('name')} — "
                    f"{d.get('total_memory_gb')} GiB — SM {d.get('major')}.{d.get('minor')} — "
                    f"{d.get('multi_processor_count')} MPs"
                )
    if info.get("nvidia_smi_query_gpu"):
        for row in info["nvidia_smi_query_gpu"]:
            lines.append(
                f"- **nvidia-smi:** {row.get('name')} | driver {row.get('driver_version')} | "
                f"VRAM {row.get('memory_total')} | {row.get('compute_mode') or ''}"
            )
    elif info.get("nvidia_smi_L"):
        lines.append(f"- **nvidia-smi -L:**\n\n```\n{info['nvidia_smi_L']}\n```")
    lines.extend(["", "*Valores obtidos automaticamente; para papers, confirme driver e versões no próprio sistema.*", ""])
    out = artifacts / "machine_report_thesis.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _find_xtts_finetune_bundle(xtts_work: Path) -> tuple[Path, Path, Path] | None:
    """Localiza checkpoint GPT treinado + config + vocab nos artefactos do demo (pasta XTTS baixada)."""
    training = xtts_work / "run" / "training"
    base = training / "XTTS_v2.0_original_model_files"
    cfg = base / "config.json"
    vocab = base / "vocab.json"
    if not cfg.is_file() or not vocab.is_file():
        return None
    runs = sorted(training.glob("GPT_XTTS_FT-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        return None
    run_dir = runs[0]
    for name in ("best_model.pth", "model.pth"):
        p = run_dir / name
        if p.is_file():
            return p, cfg, vocab
    ckpts = [
        p
        for p in run_dir.rglob("*.pth")
        if p.is_file() and "similarities" not in p.name.lower() and "speakers" not in p.name.lower()
    ]
    if not ckpts:
        return None
    ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0], cfg, vocab


def _read_first_n_eval_rows(eval_csv: Path, n: int) -> list[dict[str, str]]:
    rows = _read_coqui_metadata_csv(eval_csv)
    return rows[: max(0, n)]


def _run_audio_comparison_samples(
    artifacts: Path,
    eval_csv: Path,
    xtts_work: Path,
    language: str,
    n_samples: int,
) -> list[str]:
    """Gera `artifacts/audio_compare/`: cópia do WAV de referência + inferência pós fine-tune para audição."""
    if n_samples <= 0:
        return []
    bundle = _find_xtts_finetune_bundle(xtts_work)
    if bundle is None:
        _LOG.warning("[audio_compare] checkpoint/config não encontrados em %s — a saltar amostras.", xtts_work)
        (artifacts / "audio_compare_skipped.txt").write_text(
            f"xtts_work={xtts_work}\n",
            encoding="utf-8",
        )
        return []
    ckpt, xtts_cfg, vocab = bundle
    rows = _read_first_n_eval_rows(eval_csv, n_samples)
    if not rows:
        _LOG.warning("[audio_compare] eval CSV sem linhas: %s", eval_csv)
        return []

    out_dir = artifacts / "audio_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        import torchaudio
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
    except Exception as e:
        _LOG.warning("[audio_compare] import falhou: %s", e)
        (artifacts / "audio_compare_error.txt").write_text(str(e), encoding="utf-8")
        return []

    cfg = XttsConfig()
    cfg.load_json(str(xtts_cfg))
    model = Xtts.init_from_config(cfg)
    model.load_checkpoint(cfg, checkpoint_path=str(ckpt), vocab_path=str(vocab), use_deepspeed=False)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    manifest: list[dict[str, str]] = []
    written: list[str] = []
    for i, row in enumerate(rows, start=1):
        text = (row.get("text") or "").strip()
        ref = Path(row["audio_file"])
        if not text or not ref.is_file():
            continue
        tag = f"{i:02d}"
        ref_dest = out_dir / f"ref_{tag}.wav"
        try:
            shutil.copy2(ref, ref_dest)
        except OSError as e:
            _LOG.warning("[audio_compare] cópia ref falhou: %s", e)
            continue
        try:
            with torch.inference_mode():
                gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
                    audio_path=str(ref),
                    gpt_cond_len=cfg.gpt_cond_len,
                    max_ref_length=cfg.max_ref_len,
                    sound_norm_refs=cfg.sound_norm_refs,
                )
                out = model.inference(
                    text=text,
                    language=language,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    temperature=cfg.temperature,
                    length_penalty=cfg.length_penalty,
                    repetition_penalty=cfg.repetition_penalty,
                    top_k=cfg.top_k,
                    top_p=cfg.top_p,
                )
            wav = torch.tensor(out["wav"]).unsqueeze(0)
            infer_path = out_dir / f"infer_finetuned_{tag}.wav"
            torchaudio.save(str(infer_path), wav, 24000)
            written.extend([ref_dest.name, infer_path.name])
            manifest.append(
                {
                    "index": tag,
                    "reference_wav": str(ref_dest),
                    "inferred_wav": str(infer_path),
                    "source_audio": str(ref),
                    "text": text[:2000],
                    "language": language,
                }
            )
        except Exception as e:
            _LOG.warning("[audio_compare] inferência %s falhou: %s", tag, e)
            continue

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _LOG.info("[audio_compare] gravado em %s (%s pares)", out_dir, len(manifest))
    return written


def _export_trainer_timing_csv_and_json(plot_mod: object, log_path: Path, artifacts: Path, run_name: str) -> list[str]:
    """CSV/JSON com tempo por step e por época + estatísticas de loss."""
    written: list[str] = []
    points = plot_mod.parse_trainer_log(log_path)
    # Por step
    step_csv = artifacts / "training_time_by_step.csv"
    with step_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "global_step",
                "epoch",
                "phase",
                "step_time_s",
                "loss",
                "log_timestamp",
            ]
        )
        for p in points:
            w.writerow(
                [
                    p.global_step,
                    p.epoch if p.epoch is not None else "",
                    p.phase,
                    f"{p.step_time:.6f}" if p.step_time is not None else "",
                    f"{p.loss:.8f}" if p.loss is not None else "",
                    p.log_timestamp or "",
                ]
            )
    written.append(step_csv.name)

    wall = plot_mod.parse_epoch_clock_spans_seconds(log_path)
    sum_st = plot_mod.sum_step_time_by_epoch(points, "train")
    epochs = sorted(set(wall.keys()) | set(sum_st.keys()))
    epoch_rows = []
    for e in epochs:
        epoch_rows.append(
            {
                "epoch": e,
                "wall_span_log_clock_s": wall.get(e),
                "sum_step_time_s_train": sum_st.get(e),
                "n_train_steps_with_step_time": len(
                    [p for p in points if p.phase == "train" and p.epoch == e and p.step_time is not None]
                ),
            }
        )
    ep_path = artifacts / "training_time_by_epoch.json"
    ep_path.write_text(
        json.dumps(
            {
                "run_name": run_name,
                "log_file": str(log_path),
                "note": (
                    "wall_span_log_clock_s: max(TIME)−min(TIME) por época no trainer_0_log.txt; "
                    "sum_step_time_s_train: soma dos step_time reportados nos steps de treino."
                ),
                "epochs": epoch_rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(ep_path.name)

    stats = plot_mod.loss_train_eval_statistics(points)
    stats_path = artifacts / "training_statistics.json"
    stats_path.write_text(
        json.dumps(
            {
                "run_name": run_name,
                "log_file": str(log_path),
                **stats,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(stats_path.name)
    return written


def _write_pip_freeze(target: Path) -> None:
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        target.write_text(out.stdout or "", encoding="utf-8")
    except Exception as e:
        target.write_text(f"# pip freeze failed: {e}\n", encoding="utf-8")


def _git_head(repo: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            return (out.stdout or "").strip() or None
    except OSError:
        return None
    return None


def _split_coqui_metadata(
    rows: list[dict[str, str]],
    eval_fraction: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rng = random.Random(seed)
    rows = rows.copy()
    rng.shuffle(rows)
    n_eval = max(1, int(len(rows) * eval_fraction))
    eval_rows = sorted(rows[:n_eval], key=lambda r: r["audio_file"])
    train_rows = sorted(rows[n_eval:], key=lambda r: r["audio_file"])
    return train_rows, eval_rows


def _canonical_coqui_row(row: dict) -> dict[str, str] | None:
    """Map variedade de cabecalhos para audio_file + text (Coqui pipe-CSV)."""
    if not row:
        return None
    clean: dict[str, str] = {}
    for k, v in row.items():
        if k is None:
            continue
        kk = str(k).strip().strip("\ufeff").lower()
        clean[kk] = (v or "").strip()
    audio = None
    for key in (
        "audio_file",
        "audio_path",
        "wav_path",
        "wav",
        "path",
        "file",
        "audio",
        "clip",
    ):
        if clean.get(key):
            audio = clean[key]
            break
    text = None
    for key in ("text", "transcript", "sentence", "normalized_text", "label", "labels"):
        if clean.get(key):
            text = clean[key]
            break
    if not audio or not text:
        return None
    out: dict[str, str] = {"audio_file": audio, "text": text}
    for extra in ("speaker_name", "language", "emotion"):
        if clean.get(extra):
            out[extra] = clean[extra]
    return out


def _read_coqui_metadata_dictreader(raw: str, delimiter: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    f = io.StringIO(raw)
    reader = csv.DictReader(f, delimiter=delimiter)
    for row in reader:
        canon = _canonical_coqui_row(row)
        if canon:
            rows.append(canon)
    return rows


def _read_coqui_metadata_pipe_lines(lines: list[str]) -> list[dict[str, str]]:
    """Linhas 'rel/wav.wav|texto' sem cabecalho (ex.: brPB22) ou com cabecalho audio_file|text."""
    rows: list[dict[str, str]] = []
    start = 0
    if lines:
        p0 = [x.strip() for x in lines[0].split("|")]
        p0l = [x.lower() for x in p0]
        if p0l and p0l[0] in ("audio_file", "audio", "path", "file", "wav"):
            start = 1
    for ln in lines[start:]:
        if "|" not in ln:
            continue
        parts = [x.strip() for x in ln.split("|")]
        if len(parts) < 2:
            continue
        audio = parts[0]
        text = "|".join(parts[1:]).strip()
        if not audio or not text:
            continue
        rows.append({"audio_file": audio, "text": text})
    return rows


def _read_metadata_file_text(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "utf-16", "cp1252"):
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return path.read_bytes().decode("utf-8", errors="replace")


def _read_coqui_metadata_csv(path: Path) -> list[dict[str, str]]:
    """Aceita Coqui pipe-CSV, CSV por virgula/tab, ou pipe por linha sem cabecalho."""
    raw = _read_metadata_file_text(path)

    lines = [ln.rstrip("\r\n") for ln in raw.splitlines()]
    lines = [ln for ln in lines if ln.strip()]
    if not lines:
        return []

    for delim in ("|", ",", "\t", ";"):
        got = _read_coqui_metadata_dictreader(raw, delim)
        if got:
            return got

    if any("|" in ln for ln in lines[: min(50, len(lines))]):
        got = _read_coqui_metadata_pipe_lines(lines)
        if got:
            return got

    return []


def _write_coqui_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="|", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _wav_input_dir(data_root: Path, audio_subdir: str) -> Path:
    """Pasta com .wav para prep Whisper; audio-subdir vazio ou '.' = proprio data-root."""
    s = (audio_subdir or "").strip()
    if not s or s == ".":
        return data_root
    return data_root / s


def _resolve_audio_path(data_root: Path, rel: str, audio_subdir: str | None) -> Path:
    """Resolve caminho relativo no metadata: tenta data-root/rel e data-root/<audio_subdir>/rel."""
    rel = rel.replace("\\", "/").strip()
    if not rel:
        return data_root.resolve()
    p = Path(rel)
    if p.is_absolute():
        return p.resolve()
    rel = rel.lstrip("/")
    sd = (audio_subdir or "").strip("/\\")

    candidates: list[Path] = []
    candidates.append(data_root / rel)
    if sd and sd != ".":
        candidates.append(data_root / sd / rel)

    seen: set[str] = set()
    ordered: list[Path] = []
    for c in candidates:
        key = str(c)
        if key not in seen:
            seen.add(key)
            ordered.append(c)

    def _with_audio_extensions(path_obj: Path) -> list[Path]:
        # Alguns datasets trazem audio_file sem sufixo (ex.: "segment_123").
        # Aqui tentamos extensoes de audio comuns antes de concluir "missing".
        if path_obj.suffix:
            return [path_obj]
        return [path_obj, path_obj.with_suffix(".wav"), path_obj.with_suffix(".mp3"), path_obj.with_suffix(".flac")]

    for c in ordered:
        for cand in _with_audio_extensions(c):
            r = cand.resolve()
            if r.is_file():
                return r
    return ordered[0].resolve()


def _prepare_from_metadata_csv(
    data_root: Path,
    meta_path: Path,
    out_dir: Path,
    eval_fraction: float,
    seed: int,
    audio_subdir: str | None,
) -> tuple[Path, Path]:
    raw_rows = _read_coqui_metadata_csv(meta_path)
    if not raw_rows:
        raise SystemExit(
            f"No rows read from {meta_path}\n"
            "Formato esperado: CSV com cabecalho (delimitador | ou , ou tab) e colunas "
            "audio_file + text (ou nomes equivalentes: wav_path, transcript, etc.), "
            "ou uma linha por amostra: caminho/relativo.wav|texto do utterance.\n"
            "Confirme UTF-8, linhas nao vazias e que o ficheiro nao e so cabecalho."
        )

    fieldnames = list(raw_rows[0].keys())
    if "audio_file" not in fieldnames or "text" not in fieldnames:
        raise SystemExit(f"{meta_path} must have at least audio_file|text columns")

    # Absolute paths so train CSV can live under experiment dir while audios stay under data_root
    # (coqui formatter joins root_path=dirname(csv) with audio_file; absolute audio_file wins in os.path.join).
    abs_rows: list[dict[str, str]] = []
    missing = 0
    for r in raw_rows:
        r = dict(r)
        rel = r["audio_file"]
        p = _resolve_audio_path(data_root, rel, audio_subdir)
        r["audio_file"] = str(p)
        if not p.is_file():
            missing += 1
        abs_rows.append(r)

    n = len(abs_rows)
    if missing == n:
        raise SystemExit(
            f"Nenhum ficheiro de audio encontrado para as {n} linhas em {meta_path}.\n"
            f"data-root={data_root}\n"
            f"audio-subdir (tambem usado como prefixo de resolucao)={audio_subdir!r}\n"
            "Os caminhos no CSV devem existir como data-root/<caminho> ou "
            "data-root/<audio-subdir>/<caminho>. Ajuste --audio-subdir ou o CSV."
        )
    if missing:
        _LOG.warning(
            "%s de %s caminhos de audio nao existem em disco (data_root=%s, audio_subdir=%r).",
            missing,
            n,
            data_root,
            audio_subdir,
        )

    train_rows, eval_rows = _split_coqui_metadata(abs_rows, eval_fraction, seed)
    train_out = out_dir / "metadata_train.csv"
    eval_out = out_dir / "metadata_eval.csv"
    _write_coqui_csv(train_out, train_rows, fieldnames)
    _write_coqui_csv(eval_out, eval_rows, fieldnames)

    return train_out, eval_out


def _run_dataset_prep_whisper(
    audio_dir: Path,
    out_dir: Path,
    language: str,
    eval_percentage: float,
    speaker_name: str,
) -> tuple[Path, Path, float]:
    from TTS.demos.xtts_ft_demo.utils.formatter import format_audio_list, list_audios

    files = sorted(list_audios(str(audio_dir)))
    if not files:
        raise SystemExit(f"No audio files found under {audio_dir}")
    log_banner("Dataset prep (Whisper + segmentação — pode demorar com muitos áudios)")
    _LOG.info("Pasta de áudio de entrada: %s", audio_dir)
    _LOG.info("Arquivos de áudio encontrados (recursivo): %s", len(files))
    for i, p in enumerate(files[:12]):
        _LOG.info("  [%s] %s", i + 1, p)
    if len(files) > 12:
        _LOG.info("  ... e mais %s arquivos", len(files) - 12)
    _LOG.info("Saída (metadata + wavs cortados): %s", out_dir)
    _LOG.info("Whisper language=%s eval%%=%s speaker=%s", language, eval_percentage * 100, speaker_name)
    sys.stdout.flush()
    t0 = time.perf_counter()
    train_csv, eval_csv, audio_total_s = format_audio_list(
        files,
        target_language=language,
        out_path=str(out_dir),
        eval_percentage=eval_percentage,
        speaker_name=speaker_name,
    )
    prep_s = time.perf_counter() - t0
    _LOG.info("Preparação concluída em %.1f s; duração total dos áudios de entrada: %.1f s", prep_s, audio_total_s)
    sys.stdout.flush()
    if audio_total_s < 120:
        _LOG.warning(
            "Duração total %.1f s abaixo do guideline de 2 min do demo Gradio; treino pode ficar instável.",
            audio_total_s,
        )
    return Path(train_csv), Path(eval_csv), prep_s


def _save_prepared_manifest(
    exp_dir: Path,
    *,
    train_csv: Path,
    eval_csv: Path,
    prep_seconds: float | None,
    prep_mode: str,
    data_root: Path,
    audio_subdir: str,
    args_snapshot: dict,
) -> Path:
    path = exp_dir / MANIFEST_NAME
    doc = {
        "version": 1,
        "prep_mode": prep_mode,
        "train_csv": str(train_csv.resolve()),
        "eval_csv": str(eval_csv.resolve()),
        "dataset_prep_seconds": prep_seconds,
        "data_root": str(data_root),
        "audio_subdir": audio_subdir,
        "args": args_snapshot,
    }
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _LOG.info("Manifest gravado: %s", path)
    return path


def _find_metadata_csv(data_root: Path) -> tuple[Path | None, Path]:
    """Localiza metadata.csv e a raiz para resolver caminhos relativos dos audios.

    Se data-root termina em ``dataset`` e ``metadata.csv`` está só na pasta pai, usa-o e
    resolve caminhos relativamente ao pai.
    """
    direct = data_root / "metadata.csv"
    if direct.is_file():
        return direct, data_root
    if data_root.name.lower() == "dataset":
        parent_meta = data_root.parent / "metadata.csv"
        if parent_meta.is_file():
            _LOG.info(
                "metadata.csv na pasta pai (data-root termina em dataset): %s — "
                "caminhos de audio resolvem-se relativamente a %s",
                parent_meta,
                data_root.parent,
            )
            return parent_meta, data_root.parent
    return None, data_root


def _count_audio_files_under(root: Path) -> int:
    """Conta .wav/.mp3/.flac recursivamente (mesma ideia que o demo xtts_ft)."""
    if not root.is_dir():
        return 0
    n = 0
    try:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in AUDIO_FILE_EXTENSIONS:
                n += 1
    except OSError:
        return 0
    return n


def _discover_whisper_audio_dir(data_root: Path, audio_subdir: str) -> Path:
    """Se audio-subdir vazio e não há ficheiros na raiz, tenta wavs/, audio/, dataset/, raw/."""
    base = _wav_input_dir(data_root, audio_subdir)
    if _count_audio_files_under(base) > 0:
        return base
    sd = (audio_subdir or "").strip()
    if not sd or sd == ".":
        for sub in ("wavs", "audio", "dataset", "raw"):
            cand = data_root / sub
            if cand.is_dir() and _count_audio_files_under(cand) > 0:
                _LOG.info(
                    "Áudio encontrado em %s (subpasta automática; coloque .wav/.mp3/.flac em data-root ou aqui).",
                    cand,
                )
                return cand
    return base


def _log_data_root_inventory(data_root: Path, audio_subdir: str) -> None:
    """Resume o que existe em data-root antes do prep (sem Arrow: só ficheiros e CSV)."""
    tr = data_root / "metadata_train.csv"
    ev = data_root / "metadata_eval.csv"
    mc = data_root / "metadata.csv"
    meta_pair = tr.is_file() and ev.is_file()
    _LOG.info("Inventário data-root=%s", data_root)
    _LOG.info(
        "  metadata_train.csv + metadata_eval.csv: %s",
        "sim — reutilização direta" if meta_pair else "não",
    )
    _LOG.info("  metadata.csv: %s", "sim" if mc.is_file() else "não")
    adir = _discover_whisper_audio_dir(data_root, audio_subdir)
    n_aud = _count_audio_files_under(adir)
    _LOG.info("  áudios (%s): %s ficheiros", adir, n_aud)
    if not meta_pair and not mc.is_file() and n_aud == 0:
        _LOG.warning(
            "Nenhum CSV de metadata nem áudio encontrado. Adicione .wav/.mp3/.flac em %s ou subpastas wavs/, audio/, dataset/, raw/.",
            data_root,
        )


def _maybe_copy_training_csvs_to_data_root(
    data_root: Path,
    train_csv: Path,
    eval_csv: Path,
    prep_mode: str,
) -> None:
    """Grava metadata_train/eval na pasta de dados para a próxima execução reutilizar sem Whisper."""
    if prep_mode not in ("whisper", "metadata_csv"):
        return
    try:
        shutil.copy2(train_csv, data_root / "metadata_train.csv")
        shutil.copy2(eval_csv, data_root / "metadata_eval.csv")
        _LOG.info(
            "metadata_train.csv e metadata_eval.csv copiados para %s (próximo run usa estes CSVs).",
            data_root,
        )
    except OSError as e:
        _LOG.warning("Não foi possível copiar CSVs para data-root: %s", e)


def _load_prepared_manifest(exp_dir: Path) -> dict:
    path = exp_dir / MANIFEST_NAME
    if not path.is_file():
        raise SystemExit(
            f"Manifest não encontrado: {path}\n"
            f"Execute antes a etapa --stage prep com o mesmo --experiment-dir."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_dataset_paths(
    args: argparse.Namespace,
    data_root: Path,
    exp_dir: Path,
    artifacts: Path,
) -> tuple[Path, Path, float | None, str]:
    """Retorna train_csv, eval_csv, segundos de prep, modo (rótulo)."""
    if args.skip_dataset_prep:
        train_csv = data_root / "metadata_train.csv"
        eval_csv = data_root / "metadata_eval.csv"
        if not train_csv.is_file() or not eval_csv.is_file():
            raise SystemExit(
                "Com --skip-dataset-prep, são necessários metadata_train.csv e metadata_eval.csv em data-root."
            )
        log_banner("Dataset: usando CSVs prontos (--skip-dataset-prep)")
        _LOG.info("train=%s", train_csv)
        _LOG.info("eval =%s", eval_csv)
        return train_csv, eval_csv, 0.0, "skip_csv"

    if (data_root / "metadata_train.csv").is_file() and (data_root / "metadata_eval.csv").is_file():
        train_csv = data_root / "metadata_train.csv"
        eval_csv = data_root / "metadata_eval.csv"
        log_banner("Dataset: metadata_train/eval já existem em data-root")
        _LOG.info("train=%s", train_csv)
        _LOG.info("eval =%s", eval_csv)
        return train_csv, eval_csv, 0.0, "existing_in_root"

    meta_path, resolve_root = _find_metadata_csv(data_root)
    if meta_path is not None:
        prep_dir = exp_dir / "prepared_from_metadata"
        prep_dir.mkdir(parents=True, exist_ok=True)
        log_banner("Dataset: a partir de metadata.csv (split train/eval)")
        _LOG.info("metadata.csv=%s", meta_path)
        _LOG.info("Raiz de resolucao de caminhos de audio=%s", resolve_root)
        t_prep = time.perf_counter()
        train_csv, eval_csv = _prepare_from_metadata_csv(
            resolve_root,
            meta_path,
            prep_dir,
            args.eval_fraction,
            args.seed,
            args.audio_subdir,
        )
        prep_seconds = time.perf_counter() - t_prep
        shutil.copy2(meta_path, artifacts / "source_metadata.csv")
        _LOG.info("Split feito em %.1f s", prep_seconds)
        return train_csv, eval_csv, prep_seconds, "metadata_csv"

    audio_dir = _discover_whisper_audio_dir(data_root, args.audio_subdir)
    if not audio_dir.is_dir():
        raise SystemExit(
            f"Sem metadata.csv em {data_root} e pasta de áudio inexistente: {audio_dir}\n"
            f"Coloque .wav/.mp3/.flac em data-root ou defina --audio-subdir (ex.: dataset)."
        )
    if _count_audio_files_under(audio_dir) == 0:
        raise SystemExit(
            f"Nenhum áudio .wav/.mp3/.flac encontrado sob {audio_dir} (recursivo).\n"
            f"Adicione ficheiros ou forneça metadata.csv / metadata_train+eval."
        )
    prep_dir = exp_dir / "prepared_whisper"
    prep_dir.mkdir(parents=True, exist_ok=True)
    train_csv, eval_csv, prep_seconds = _run_dataset_prep_whisper(
        audio_dir,
        prep_dir,
        args.prep_language,
        args.eval_fraction,
        args.speaker_name,
    )
    return train_csv, eval_csv, prep_seconds, "whisper"


def _write_experiment_config(
    exp_dir: Path,
    cfg: dict,
) -> None:
    (exp_dir / "experiment_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _run_training_and_artifacts(
    args: argparse.Namespace,
    exp_dir: Path,
    artifacts: Path,
    train_csv: Path,
    eval_csv: Path,
    prep_seconds: float | None,
    cfg: dict,
) -> int:
    t_stage_wall = time.perf_counter()
    shutil.copy2(train_csv, artifacts / "metadata_train.csv")
    shutil.copy2(eval_csv, artifacts / "metadata_eval.csv")

    timings: dict = {"dataset_prep_seconds": prep_seconds}

    xtts_work = exp_dir / "xtts_work"
    xtts_work.mkdir(parents=True, exist_ok=True)

    max_frames = int(float(args.max_audio_seconds) * 22050)

    log_banner("Fine-tuning GPT (train_gpt)")
    _LOG.info("train_csv=%s", train_csv)
    _LOG.info("eval_csv =%s", eval_csv)
    _LOG.info(
        "epochs=%s batch=%s grad_acumm=%s lang=%s max_audio_frames=%s ckpt_every=%s save_n=%s",
        args.epochs,
        args.batch_size,
        args.grad_acumm,
        args.train_language,
        max_frames,
        args.checkpoint_every_epochs,
        args.save_n_checkpoints,
    )
    _LOG.info("output_path (xtts_work)=%s", xtts_work)
    sys.stdout.flush()

    from TTS.demos.xtts_ft_demo.utils.gpt_train import train_gpt

    t_gpt = time.perf_counter()
    _config_path, _base_ckpt, _vocab, trainer_out, _speaker_ref, loss_plot_path = train_gpt(
        args.train_language,
        args.epochs,
        args.batch_size,
        args.grad_acumm,
        str(train_csv),
        str(eval_csv),
        output_path=str(xtts_work),
        max_audio_length=max_frames,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        save_n_checkpoints=args.save_n_checkpoints,
    )
    gpt_s = time.perf_counter() - t_gpt
    timings["training_gpt_wall_seconds"] = round(gpt_s, 3)
    timings["training_wall_seconds"] = round(gpt_s, 3)
    _LOG.info(
        "Tempo etapa train_gpt: %s",
        _format_duration(gpt_s),
    )

    training_root = xtts_work / "run" / "training"
    if trainer_out:
        timings["trainer_output_path"] = trainer_out

    plot_artifacts: list[str] = []
    if loss_plot_path and os.path.isfile(loss_plot_path):
        dest_demo = artifacts / "training_loss_demo_callback.png"
        shutil.copy2(loss_plot_path, dest_demo)
        plot_artifacts.append(dest_demo.name)

    t_post = time.perf_counter()
    log_banner("Pós-processamento (logs, gráficos, TensorBoard)")
    log_path = _find_trainer_log(training_root)
    trainer_summary_path: Path | None = None
    if log_path:
        shutil.copy2(log_path, artifacts / "trainer_0_log.txt")
        summ = _parse_trainer_log_summary(log_path)
        summary_dict = asdict(summ)
        summary_dict["step_time_mean"] = (
            sum(summ.step_time_samples) / len(summ.step_time_samples)
            if summ.step_time_samples
            else None
        )
        summary_dict["step_time_median"] = None
        if summ.step_time_samples:
            srt = sorted(summ.step_time_samples)
            mid = len(srt) // 2
            summary_dict["step_time_median"] = (
                srt[mid] if len(srt) % 2 else (srt[mid - 1] + srt[mid]) / 2
            )
        trainer_summary_path = artifacts / "trainer_log_summary.json"
        trainer_summary_path.write_text(
            json.dumps(summary_dict, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    plot_artifacts.extend(_run_loss_plot_scripts(training_root, artifacts))

    events = sorted(training_root.rglob("events.out.tfevents.*"))
    (artifacts / "tensorboard_event_files.txt").write_text(
        "\n".join(str(p) for p in events) + ("\n" if events else ""),
        encoding="utf-8",
    )

    post_s = time.perf_counter() - t_post
    timings["postprocess_wall_seconds"] = round(post_s, 3)
    timings["train_stage_total_wall_seconds"] = round(time.perf_counter() - t_stage_wall, 3)
    timings["epochs_requested"] = args.epochs
    timings["checkpoint_every_epochs"] = args.checkpoint_every_epochs
    timings["save_n_checkpoints"] = args.save_n_checkpoints
    timings["experiment_dir"] = str(exp_dir)
    _LOG.info(
        "Tempo pos-processamento (logs, PNGs, relatorio): %s",
        _format_duration(post_s),
    )
    _LOG.info(
        "Tempo total etapa TREINO (parede): %s",
        _format_duration(timings["train_stage_total_wall_seconds"]),
    )

    timings_json = {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in timings.items()
    }
    (artifacts / "timings.json").write_text(
        json.dumps(timings_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    cfg["timings_final"] = timings_json
    _write_experiment_config(exp_dir, cfg)

    prep_doc: dict | None = None
    prep_json = exp_dir / "timings_prep.json"
    if prep_json.is_file():
        try:
            prep_doc = json.loads(prep_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prep_doc = None

    audio_names: list[str] = []
    try:
        audio_names = _run_audio_comparison_samples(
            artifacts,
            eval_csv,
            xtts_work,
            args.train_language,
            max(0, int(args.inference_samples)),
        )
    except Exception as e:
        _LOG.warning("Amostras de áudio (comparação): %s", e)
        (artifacts / "audio_compare_error.txt").write_text(str(e), encoding="utf-8")

    _write_relatorio_experimento(
        artifacts,
        exp_dir,
        prep_doc=prep_doc,
        train_timings=timings,
        plot_files=sorted(set(plot_artifacts)),
        trainer_summary_path=trainer_summary_path,
        audio_compare_files=audio_names,
    )

    print("")
    _LOG.info("=== Treino e artefatos concluídos ===")
    _LOG.info("Artefatos: %s", artifacts)
    _LOG.info("Checkpoints: %s", training_root)
    _log_artifact_checklist(artifacts, training_root)
    (artifacts / ".stage_train_ok").write_text(
        datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
    )
    sys.stdout.flush()
    return 0


def _find_trainer_log(training_root: Path) -> Path | None:
    logs = sorted(training_root.rglob("trainer_0_log.txt"))
    if not logs:
        return None
    return max(logs, key=lambda p: p.stat().st_mtime)


def _load_plot_loss_module():
    path = REPO_ROOT / "scripts" / "xtts_plot_loss_from_trainer_logs.py"
    spec = importlib.util.spec_from_file_location("xtts_plot_loss_from_trainer_logs", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    # importlib exige registo em sys.modules para que dataclasses resolvam o módulo corretamente
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _run_loss_plot_scripts(training_root: Path, artifacts: Path) -> list[str]:
    """Gera PNGs de perda por step e por época; devolve nomes de ficheiros criados em artifacts/."""
    written: list[str] = []
    # PNG from TensorBoard (inside run folder)
    try:
        from TTS.demos.xtts_ft_demo.utils.plot_loss import save_training_loss_plot

        for d in sorted(training_root.glob("GPT_XTTS_FT-*"), key=lambda p: p.stat().st_mtime, reverse=True):
            loss_png = d / "training_loss.png"
            if save_training_loss_plot(str(d), str(loss_png)):
                if loss_png.is_file():
                    dest = artifacts / "training_loss_tensorboard.png"
                    shutil.copy2(loss_png, dest)
                    written.append(dest.name)
                break
    except Exception as e:
        (artifacts / "training_loss_tensorboard_error.txt").write_text(str(e), encoding="utf-8")

    # PNG from trainer text log (step, época, painel duplo)
    try:
        plot_mod = _load_plot_loss_module()
        logs = sorted(training_root.rglob("trainer_0_log.txt"))
        if logs:
            log_path = max(logs, key=lambda p: p.stat().st_mtime)
            points = plot_mod.parse_trainer_log(log_path)
            run_name = log_path.parent.name
            title_step = f"XTTS GPT fine-tuning — loss (trainer log)\n{run_name}"
            out_step = artifacts / "training_loss_from_trainer_log.png"
            if plot_mod.plot_points(points, out_step, title=title_step) and out_step.is_file():
                written.append(out_step.name)

            title_ep = f"Loss média por época — {run_name}"
            out_ep = artifacts / "training_loss_by_epoch.png"
            if plot_mod.plot_loss_by_epoch(points, out_ep, title=title_ep) and out_ep.is_file():
                written.append(out_ep.name)

            out_both = artifacts / "training_loss_step_and_epoch.png"
            if plot_mod.plot_loss_step_and_epoch(
                points, out_both, title_prefix=f"XTTS GPT — {run_name}"
            ) and out_both.is_file():
                written.append(out_both.name)

            out_st = artifacts / "training_step_time_by_global_step.png"
            if plot_mod.plot_step_time_by_global_step(
                points, out_st, title=f"step_time (s) vs global step — {run_name}"
            ) and out_st.is_file():
                written.append(out_st.name)

            summary_txt = artifacts / "trainer_log_metrics_summary.txt"
            summary_txt.write_text(
                f"log_file={log_path}\npoints_parsed={len(points)}\n",
                encoding="utf-8",
            )
            written.extend(_export_trainer_timing_csv_and_json(plot_mod, log_path, artifacts, run_name))
    except Exception as e:
        (artifacts / "trainer_log_plot_error.txt").write_text(str(e), encoding="utf-8")

    return written


def _log_artifact_checklist(artifacts: Path, training_root: Path) -> None:
    """Regista checklist final de artefatos esperados (presente/ausente)."""
    run_dirs = sorted(training_root.glob("GPT_XTTS_FT-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    run_dir = run_dirs[0] if run_dirs else None
    latest_ckpt = None
    if run_dir:
        ckpts = sorted(run_dir.glob("checkpoint_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
        if ckpts:
            latest_ckpt = ckpts[0]

    checks: list[tuple[str, Path | None]] = [
        ("checkpoint_final_preferido", (run_dir / "best_model.pth") if run_dir else None),
        ("checkpoint_fallback_model", (run_dir / "model.pth") if run_dir else None),
        ("checkpoint_mais_recente", latest_ckpt),
        ("tempo_medio_por_epoca_json", artifacts / "training_time_by_epoch.json"),
        ("tempo_por_step_csv", artifacts / "training_time_by_step.csv"),
        ("loss_por_epoca_png", artifacts / "training_loss_by_epoch.png"),
        ("loss_por_step_png", artifacts / "training_loss_from_trainer_log.png"),
        ("loss_tensorboard_png", artifacts / "training_loss_tensorboard.png"),
        ("inferencia_amostras_manifest", artifacts / "audio_compare" / "manifest.json"),
    ]

    _LOG.info("Checklist final de artefatos esperados:")
    for label, p in checks:
        ok = bool(p and p.is_file())
        _LOG.info("  - [%s] %s: %s", "OK" if ok else "MISSING", label, p if p else "(n/a)")


def _write_relatorio_experimento(
    artifacts: Path,
    exp_dir: Path,
    *,
    prep_doc: dict | None,
    train_timings: dict | None,
    plot_files: list[str],
    trainer_summary_path: Path | None,
    audio_compare_files: list[str] | None = None,
) -> Path:
    """Relatório Markdown com tempos por etapa, lista de gráficos e resumo de métricas."""
    lines: list[str] = [
        "# Relatório — fine-tune XTTS GPT",
        "",
        f"- **experiment_dir:** `{exp_dir}`",
        f"- **artifacts:** `{artifacts}`",
        f"- **UTC:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Tempos por etapa",
        "",
    ]

    if prep_doc:
        lines.extend(
            [
                "### Preparação de dados (prep)",
                "",
                f"| Métrica | Valor |",
                f"| --- | --- |",
                f"| Modo | `{prep_doc.get('prep_mode', 'n/d')}` |",
                f"| Tempo interno dataset (prep/split/whisper) | {_format_duration(prep_doc.get('dataset_prep_seconds'))} |",
                f"| **Tempo de parede total etapa PREP** | **{_format_duration(prep_doc.get('stage_prep_wall_seconds'))}** |",
                "",
            ]
        )
    else:
        lines.extend(["*(Prep não executado nesta invocação ou dados indisponíveis.)*", ""])

    if train_timings:
        lines.extend(
            [
                "### Treino",
                "",
                f"| Métrica | Valor |",
                f"| --- | --- |",
                f"| Treino `train_gpt` (GPU/step) | {_format_duration(train_timings.get('training_gpt_wall_seconds') or train_timings.get('training_wall_seconds'))} |",
                f"| Pós-processamento (logs, gráficos, JSON) | {_format_duration(train_timings.get('postprocess_wall_seconds'))} |",
                f"| **Total etapa treino (parede)** | **{_format_duration(train_timings.get('train_stage_total_wall_seconds'))}** |",
                f"| Épocas pedidas | {train_timings.get('epochs_requested', 'n/d')} |",
                "",
            ]
        )

    lines.extend(["## Gráficos de perda (gerados em `artifacts/`)", ""])
    if plot_files:
        for name in sorted(plot_files):
            lines.append(f"- `{name}` — perda por **global step** e/ou **época** (ver legenda no PNG).")
    else:
        lines.append("- *(Nenhum gráfico PNG registado — ver ficheiros `*_error.txt` se o treino correu.)*")
    lines.append("")

    lines.extend(
        [
            "- `training_loss_step_and_epoch.png` — painel superior: loss vs step; inferior: média por época.",
            "- `training_loss_by_epoch.png` — só média por época.",
            "- `training_loss_from_trainer_log.png` — loss vs step (parser do `trainer_0_log.txt`).",
            "- `training_step_time_by_global_step.png` — tempo por step (métrica `step_time` do log) vs global step.",
            "- `training_time_by_step.csv` — colunas: global_step, epoch, phase, step_time_s, loss, log_timestamp.",
            "- `training_time_by_epoch.json` — tempo de parede aproximado por época (extensão dos carimbos TIME) e soma de `step_time` por época.",
            "- `training_statistics.json` — min/máx/última loss (train/eval) e contagens de pontos.",
            "- `machine_report_thesis.md` — texto sobre CPU, RAM, GPU e software para relatório/dissertação.",
            "",
        ]
    )

    ac = audio_compare_files or []
    lines.extend(["## Áudio (referência vs inferência pós fine-tune)", ""])
    if ac:
        lines.append(f"Pasta `artifacts/audio_compare/`: **{len(ac) // 2}** pares `ref_XX.wav` + `infer_finetuned_XX.wav` (ver `manifest.json`).")
        for name in sorted(ac)[:24]:
            lines.append(f"- `{name}`")
        if len(ac) > 24:
            lines.append(f"- … e mais {len(ac) - 24} ficheiros.")
    else:
        lines.append(
            "- *(Nenhuma amostra gerada — defina `--inference-samples N` ou ver `audio_compare_skipped.txt` / erros.)*"
        )
    lines.append("")

    summary: dict | None = None
    if trainer_summary_path and trainer_summary_path.is_file():
        try:
            summary = json.loads(trainer_summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = None
    if summary:
        lines.extend(["## Resumo do log do trainer", ""])
        n_ep = len(summary.get("epoch_lines") or [])
        n_st = len(summary.get("step_time_samples") or [])
        lines.append(f"- Linhas de época capturadas no log: **{n_ep}**")
        lines.append(f"- Amostras de `step_time` no log: **{n_st}**")
        stm = summary.get("step_time_mean")
        if stm is not None:
            lines.append(f"- Tempo médio por step (log): **{stm:.4f} s**")
        lines.append("")

    lines.append("---")
    lines.append("*Gerado automaticamente por `scripts/xtts_finetune_paraiba_experiment.py`.*")
    lines.append("")

    out = artifacts / "relatorio_experimento.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    _LOG.info("Relatório Markdown: %s", out)
    return out


def _args_snapshot(args: argparse.Namespace) -> dict:
    skip = {"experiment_dir", "data_root"}
    out: dict = {}
    for k, v in vars(args).items():
        if k in skip:
            continue
        if isinstance(v, Path):
            out[k] = str(v) if v is not None else None
        else:
            out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="XTTS fine-tuning experiment with research artifacts.")
    ap.add_argument(
        "--stage",
        choices=["prep", "train", "all"],
        default="all",
        help="prep=só preparar CSVs; train=só treinar (precisa prepared_manifest.json); all=prep+train.",
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=f"Pasta com áudios e/ou CSVs (default: {DEFAULT_DATA_ROOT}).",
    )
    ap.add_argument(
        "--no-persist-training-csv-to-data-root",
        action="store_true",
        help="Não copiar metadata_train/eval gerados para data-root após prep (Whisper/metadata).",
    )
    ap.add_argument(
        "--audio-subdir",
        type=str,
        default="",
        help="Subpasta com .wav quando não há metadata.csv (prep Whisper). Vazio ou '.' = usar só --data-root. "
        "Com metadata.csv, também tenta data-root/<subdir>/... ao resolver caminhos.",
    )
    ap.add_argument(
        "--skip-dataset-prep",
        action="store_true",
        help="Use data-root/metadata_train.csv e metadata_eval.csv como estão.",
    )
    ap.add_argument(
        "--prep-language",
        type=str,
        default=XTTS_LANG_PORTUGUESE,
        help=f"Código Whisper na preparação (default: {XTTS_LANG_PORTUGUESE}). "
        "Português brasileiro: use este código; o sotaque BR vem dos áudios.",
    )
    ap.add_argument("--eval-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--speaker-name", type=str, default="paraiba_ft")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-acumm", type=int, default=2)
    ap.add_argument(
        "--max-audio-seconds",
        type=float,
        default=11.6,
        help="Comprimento máximo do clipe em segundos (×22050 frames).",
    )
    ap.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="Pasta do experimento. Obrigatório para --stage train. Default: data/xtts-experiments/paraiba-TIMESTAMP",
    )
    ap.add_argument(
        "--train-language",
        type=str,
        default=XTTS_LANG_PORTUGUESE,
        help=f"Idioma XTTS para train_gpt (default: {XTTS_LANG_PORTUGUESE}). "
        "Único código PT no checkpoint; fine-tune PT-BR = dados em português do Brasil.",
    )
    ap.add_argument(
        "--inference-samples",
        type=int,
        default=3,
        help="Após o treino: gerar N amostras em artifacts/audio_compare/ (ref_*.wav + infer_finetuned_*.wav). 0=desativar.",
    )
    ap.add_argument(
        "--checkpoint-every-epochs",
        type=int,
        default=2,
        help="Guardar checkpoint do Trainer a cada N épocas completas (via save_step alinhado ao dataset).",
    )
    ap.add_argument(
        "--save-n-checkpoints",
        type=int,
        default=10,
        help="Rotação: manter no máximo este número de checkpoints recentes (Trainer save_n_checkpoints).",
    )
    args = ap.parse_args()
    args.prep_language = _normalize_pt_language(args.prep_language, "prep-language")
    args.train_language = _normalize_pt_language(args.train_language, "train-language")
    setup_verbose_logging()

    if args.stage in ("prep", "all") and args.data_root is None:
        args.data_root = DEFAULT_DATA_ROOT
        _LOG.info("Usando data-root por defeito do repositório: %s", args.data_root)
    if args.stage == "train" and args.experiment_dir is None:
        _LOG.error("--experiment-dir é obrigatório para --stage train")
        return 2

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir = (
        args.experiment_dir.resolve()
        if args.experiment_dir
        else (REPO_ROOT / "data" / "xtts-experiments" / f"paraiba-{ts}")
    )
    exp_dir.mkdir(parents=True, exist_ok=True)
    artifacts = exp_dir / "artifacts"
    artifacts.mkdir(exist_ok=True)

    log_banner(f"Início — stage={args.stage} — experiment_dir={exp_dir}")
    _log_pt_br_target(args.prep_language, args.train_language)
    _LOG.info("PYTHONUNBUFFERED=%s", os.environ.get("PYTHONUNBUFFERED"))
    _LOG.info("Repo: %s", REPO_ROOT)
    sys.stdout.flush()

    train_csv: Path | None = None
    eval_csv: Path | None = None
    prep_seconds: float | None = None
    prep_mode: str = ""

    if args.stage in ("prep", "all"):
        t_prep_wall_start = time.perf_counter()
        assert args.data_root is not None
        data_root = args.data_root.resolve()
        data_root.mkdir(parents=True, exist_ok=True)
        _log_data_root_inventory(data_root, args.audio_subdir)

        cfg = {
            "stage": args.stage,
            "data_root": str(data_root),
            "audio_subdir": args.audio_subdir,
            "skip_dataset_prep": args.skip_dataset_prep,
            "prep_language": args.prep_language,
            "eval_fraction": args.eval_fraction,
            "seed": args.seed,
            "speaker_name": args.speaker_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_acumm": args.grad_acumm,
            "checkpoint_every_epochs": args.checkpoint_every_epochs,
            "save_n_checkpoints": args.save_n_checkpoints,
            "max_audio_seconds": args.max_audio_seconds,
            "train_language": args.train_language,
            "locale_target": "pt-BR (Brazilian Portuguese via training data; XTTS code pt only)",
            "base_model_note": (
                "Fine-tuning from Hugging Face coqui/XTTS-v2. Portuguese uses a single language id "
                f"'{XTTS_LANG_PORTUGUESE}' (no separate pt-BR tensor). "
                "Brazilian pronunciation and wording are learned from your PT-BR audio and transcripts."
            ),
            "repo_git_head": _git_head(REPO_ROOT),
        }
        _write_experiment_config(exp_dir, cfg)
        _mi_prep = _collect_machine_info()
        (exp_dir / "machine_environment.json").write_text(
            json.dumps(_mi_prep, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _write_machine_report_thesis(artifacts, _mi_prep)
        _write_pip_freeze(exp_dir / "environment_pip_freeze.txt")

        train_csv, eval_csv, prep_seconds, prep_mode = _resolve_dataset_paths(
            args, data_root, exp_dir, artifacts
        )
        if not args.no_persist_training_csv_to_data_root:
            _maybe_copy_training_csvs_to_data_root(data_root, train_csv, eval_csv, prep_mode)
        shutil.copy2(train_csv, artifacts / "metadata_train.csv")
        shutil.copy2(eval_csv, artifacts / "metadata_eval.csv")
        prep_wall_s = time.perf_counter() - t_prep_wall_start
        prep_timing_doc = {
            "dataset_prep_seconds": prep_seconds,
            "prep_mode": prep_mode,
            "stage_prep_wall_seconds": round(prep_wall_s, 3),
            "utc_iso_prep_finished": datetime.now(timezone.utc).isoformat(),
        }
        (exp_dir / "timings_prep.json").write_text(
            json.dumps(prep_timing_doc, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _LOG.info(
            "Tempo etapa PREP (parede, inclui manifest e artefatos): %s",
            _format_duration(prep_wall_s),
        )
        _LOG.info(
            "Tempo interno preparacao dataset (split/whisper/metadata): %s",
            _format_duration(prep_seconds),
        )
        _save_prepared_manifest(
            exp_dir,
            train_csv=train_csv,
            eval_csv=eval_csv,
            prep_seconds=prep_seconds,
            prep_mode=prep_mode,
            data_root=data_root,
            audio_subdir=args.audio_subdir,
            args_snapshot=_args_snapshot(args),
        )
        log_banner("Etapa PREP concluída — manifest e CSVs em artifacts/")
        _LOG.info("Próximo passo (treino): --stage train --experiment-dir \"%s\"", exp_dir)
        sys.stdout.flush()
        (artifacts / ".stage_prep_ok").write_text(
            datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
        )
        if args.stage == "prep":
            _write_relatorio_experimento(
                artifacts,
                exp_dir,
                prep_doc=prep_timing_doc,
                train_timings=None,
                plot_files=[],
                trainer_summary_path=None,
            )
            return 0

    if args.stage in ("train", "all"):
        if args.stage == "train":
            man = _load_prepared_manifest(exp_dir)
            train_csv = Path(man["train_csv"])
            eval_csv = Path(man["eval_csv"])
            prep_seconds = man.get("dataset_prep_seconds")
            if not train_csv.is_file() or not eval_csv.is_file():
                _LOG.error("CSV do manifest inexistente: train=%s eval=%s", train_csv, eval_csv)
                return 3
            cfg_path = exp_dir / "experiment_config.json"
            if cfg_path.is_file():
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            else:
                cfg = {}
            _mi_tr = _collect_machine_info()
            (exp_dir / "machine_environment_at_train.json").write_text(
                json.dumps(_mi_tr, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            _write_machine_report_thesis(artifacts, _mi_tr)
        else:
            assert train_csv is not None and eval_csv is not None
            cfg_path = exp_dir / "experiment_config.json"
            cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
            _mi_tr = _collect_machine_info()
            (exp_dir / "machine_environment_at_train.json").write_text(
                json.dumps(_mi_tr, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            _write_machine_report_thesis(artifacts, _mi_tr)

        assert train_csv is not None and eval_csv is not None
        return _run_training_and_artifacts(
            args, exp_dir, artifacts, train_csv, eval_csv, prep_seconds, cfg
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
