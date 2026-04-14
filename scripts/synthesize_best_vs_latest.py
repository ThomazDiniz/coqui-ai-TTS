from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import torch
import torchaudio

# Allow running from source tree without requiring editable install.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from TTS.tts.configs.xtts_config import XttsConfig  # noqa: E402
from TTS.tts.models.xtts import Xtts  # noqa: E402


CKPT_RE = re.compile(r"checkpoint_(\d+)\.pth$", re.IGNORECASE)


def _find_latest_checkpoint(run_dir: Path) -> Path | None:
    ckpts = []
    for p in run_dir.glob("checkpoint_*.pth"):
        if not p.is_file():
            continue
        m = CKPT_RE.search(p.name)
        step = int(m.group(1)) if m else -1
        ckpts.append((step, p.stat().st_mtime, p))
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: (x[0], x[1]))
    return ckpts[-1][2]


def _load_rows(eval_csv: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with eval_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            rows.append({k: (v or "").strip() for k, v in row.items()})
    return rows


def _resolve_audio_path(
    raw_audio_path: str,
    eval_csv: Path,
    data_root: Path | None,
) -> Path | None:
    p = Path(raw_audio_path)
    if p.is_file():
        return p

    candidates: list[Path] = []
    # Relative to metadata location.
    candidates.append((eval_csv.parent / p).resolve())
    # Relative to experiment data root from prepared_manifest.
    if data_root is not None:
        candidates.append((data_root / p).resolve())
    # As plain absolute string fallback.
    try:
        candidates.append(Path(raw_audio_path).resolve())
    except OSError:
        pass

    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def _pick_samples(
    rows: list[dict[str, str]],
    eval_csv: Path,
    data_root: Path | None,
    n: int,
    seed: int,
) -> list[dict[str, str]]:
    valid: list[dict[str, str]] = []
    for row in rows:
        text = (row.get("text") or "").strip()
        raw_audio = (row.get("audio_file") or "").strip()
        if not text or not raw_audio:
            continue
        resolved = _resolve_audio_path(raw_audio, eval_csv, data_root)
        if resolved is None:
            continue
        valid.append(
            {
                "text": text,
                "audio_file": raw_audio,
                "resolved_audio_file": str(resolved),
                "speaker_name": (row.get("speaker_name") or "").strip(),
            }
        )
    if len(valid) < n:
        raise RuntimeError(f"Não há linhas válidas suficientes no eval CSV: {len(valid)} < {n}")
    rng = random.Random(seed)
    return rng.sample(valid, n)


def _load_texts_file(path: Path) -> list[str]:
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        t = raw.strip()
        if t:
            lines.append(t)
    return lines


def _load_model(config_path: Path, vocab_path: Path, ckpt_path: Path, device: str) -> tuple[Xtts, XttsConfig]:
    cfg = XttsConfig()
    cfg.load_json(str(config_path))
    model = Xtts.init_from_config(cfg)
    model.load_checkpoint(cfg, checkpoint_path=str(ckpt_path), vocab_path=str(vocab_path), use_deepspeed=False)
    if device == "cuda":
        model.cuda()
    else:
        model.cpu()
    model.eval()
    return model, cfg


def _synthesize_one(
    model: Xtts,
    cfg: XttsConfig,
    ref_wav: Path,
    text: str,
    language: str,
    out_wav: Path,
) -> None:
    with torch.inference_mode():
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=str(ref_wav),
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
    wav = torch.tensor(out["wav"]).unsqueeze(0).cpu()
    torchaudio.save(str(out_wav), wav, 24000)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Gera comparação de síntese entre best_model.pth e checkpoint mais recente."
    )
    ap.add_argument("--run-dir", required=True, help="Pasta do run GPT_XTTS_FT-... (contém best_model.pth)")
    ap.add_argument("--n", type=int, default=5, help="Quantidade de frases aleatórias (default: 5)")
    ap.add_argument("--seed", type=int, default=42, help="Seed para seleção aleatória de frases")
    ap.add_argument("--language", default="pt", help="Idioma XTTS (default: pt)")
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Dispositivo para inferência",
    )
    ap.add_argument("--out-dir", default="", help="Pasta de saída. Default: <run-dir>/best_vs_latest_compare")
    ap.add_argument(
        "--reference-wav",
        default="",
        help="WAV de referência fixo para todas as sínteses (opcional).",
    )
    ap.add_argument(
        "--texts-file",
        default="",
        help="Arquivo .txt com 1 frase por linha (usa frases fixas, ignora seleção aleatória).",
    )
    ap.add_argument(
        "--text",
        action="append",
        default=[],
        help="Frase fixa (pode repetir --text várias vezes).",
    )
    ap.add_argument(
        "--prefix",
        default="",
        help="Prefixo opcional para nomes de saída (ex.: pb0003_custom).",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        print(f"[ERRO] run-dir não existe: {run_dir}", file=sys.stderr)
        return 2

    best_model = run_dir / "best_model.pth"
    if not best_model.is_file():
        print(f"[ERRO] best_model.pth não encontrado em: {run_dir}", file=sys.stderr)
        return 2

    latest_ckpt = _find_latest_checkpoint(run_dir)
    if latest_ckpt is None:
        print(f"[ERRO] nenhum checkpoint_*.pth encontrado em: {run_dir}", file=sys.stderr)
        return 2

    training_root = run_dir.parent
    base_files = training_root / "XTTS_v2.0_original_model_files"
    config_path = base_files / "config.json"
    vocab_path = base_files / "vocab.json"
    if not config_path.is_file() or not vocab_path.is_file():
        print(f"[ERRO] config/vocab não encontrados em: {base_files}", file=sys.stderr)
        return 2

    exp_dir = run_dir.parents[3]
    manifest_path = exp_dir / "prepared_manifest.json"
    if not manifest_path.is_file():
        print(f"[ERRO] prepared_manifest.json não encontrado em: {exp_dir}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    eval_csv = Path(manifest["eval_csv"]).resolve()
    data_root_raw = manifest.get("data_root")
    data_root = Path(data_root_raw).resolve() if data_root_raw else None
    if not eval_csv.is_file():
        print(f"[ERRO] eval CSV não encontrado: {eval_csv}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (run_dir / "best_vs_latest_compare")
    out_dir.mkdir(parents=True, exist_ok=True)

    custom_texts: list[str] = []
    if args.texts_file:
        tf = Path(args.texts_file).resolve()
        if not tf.is_file():
            print(f"[ERRO] texts-file não encontrado: {tf}", file=sys.stderr)
            return 2
        custom_texts.extend(_load_texts_file(tf))
    if args.text:
        custom_texts.extend([t.strip() for t in args.text if t and t.strip()])

    if custom_texts:
        if not args.reference_wav:
            print("[ERRO] use --reference-wav quando fornecer frases fixas (--text/--texts-file).", file=sys.stderr)
            return 2
        ref_wav = Path(args.reference_wav).resolve()
        if not ref_wav.is_file():
            print(f"[ERRO] reference-wav não encontrado: {ref_wav}", file=sys.stderr)
            return 2
        samples = [
            {
                "text": text,
                "audio_file": str(ref_wav),
                "resolved_audio_file": str(ref_wav),
                "speaker_name": "",
            }
            for text in custom_texts
        ]
    else:
        rows = _load_rows(eval_csv)
        samples = _pick_samples(rows, eval_csv, data_root, max(1, int(args.n)), int(args.seed))

    tag_prefix = (args.prefix.strip() + "_") if args.prefix.strip() else ""

    print(f"[INFO] run_dir={run_dir}")
    print(f"[INFO] best_model={best_model.name}")
    print(f"[INFO] latest_checkpoint={latest_ckpt.name}")
    print(f"[INFO] samples={len(samples)} seed={args.seed}")
    if args.reference_wav:
        print(f"[INFO] reference_wav={Path(args.reference_wav).resolve()}")
    if custom_texts:
        print("[INFO] modo=frases fixas")
    print(f"[INFO] out_dir={out_dir}")

    t0 = time.perf_counter()
    model_best, cfg_best = _load_model(config_path, vocab_path, best_model, args.device)
    t_best_load = time.perf_counter() - t0
    print(f"[INFO] best model carregado em {t_best_load:.2f}s")

    t1 = time.perf_counter()
    model_latest, cfg_latest = _load_model(config_path, vocab_path, latest_ckpt, args.device)
    t_latest_load = time.perf_counter() - t1
    print(f"[INFO] latest checkpoint carregado em {t_latest_load:.2f}s")

    report_rows: list[dict[str, str]] = []
    for i, row in enumerate(samples, start=1):
        tag = f"{i:02d}"
        text = row["text"]
        ref = Path(row["resolved_audio_file"])
        ref_out = out_dir / f"{tag_prefix}ref_{tag}.wav"
        best_out = out_dir / f"{tag_prefix}best_{tag}.wav"
        latest_out = out_dir / f"{tag_prefix}latest_{tag}.wav"

        # Copy reference for easy listening side-by-side.
        ref_wav, ref_sr = torchaudio.load(str(ref))
        torchaudio.save(str(ref_out), ref_wav, ref_sr)

        _synthesize_one(model_best, cfg_best, ref, text, args.language, best_out)
        _synthesize_one(model_latest, cfg_latest, ref, text, args.language, latest_out)

        report_rows.append(
            {
                "index": tag,
                "text": text,
                "speaker_name": row.get("speaker_name", ""),
                "source_audio_file": row["audio_file"],
                "resolved_source_audio_file": str(ref),
                "reference_copy": str(ref_out),
                "best_model_output": str(best_out),
                "latest_checkpoint_output": str(latest_out),
            }
        )
        print(f"[OK] sample {tag} gerado")

    out_manifest = {
        "run_dir": str(run_dir),
        "best_model": str(best_model),
        "latest_checkpoint": str(latest_ckpt),
        "config": str(config_path),
        "vocab": str(vocab_path),
        "eval_csv": str(eval_csv),
        "language": args.language,
        "seed": int(args.seed),
        "n_samples": len(report_rows),
        "prefix": args.prefix,
        "generated_at_epoch_seconds": time.time(),
        "samples": report_rows,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(out_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "selected_texts.txt").write_text(
        "\n".join(f"{r['index']}|{r['text']}" for r in report_rows) + "\n",
        encoding="utf-8",
    )

    print(f"[DONE] arquivos gerados em: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
