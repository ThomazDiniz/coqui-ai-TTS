from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import torch
import torchaudio

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


CHECKPOINT_RE = re.compile(r"checkpoint_(\d+)\.pth$", re.IGNORECASE)
BEST_STEP_RE = re.compile(r"best_model_(\d+)\.pth$", re.IGNORECASE)


def discover_models(run_dir: Path) -> list[tuple[str, Path]]:
    models: list[tuple[str, Path]] = []
    best = run_dir / "best_model.pth"
    if best.is_file():
        models.append(("best", best))

    for p in sorted(run_dir.glob("checkpoint_*.pth")):
        m = CHECKPOINT_RE.match(p.name)
        if m:
            models.append((m.group(1), p))

    for p in sorted(run_dir.glob("best_model_*.pth")):
        m = BEST_STEP_RE.match(p.name)
        if m:
            sid = f"{m.group(1)}_best"
            models.append((sid, p))

    if not models:
        raise RuntimeError(f"Nenhum checkpoint encontrado em {run_dir}")
    return models


def load_model(config_path: Path, vocab_path: Path, ckpt_path: Path, device: str) -> tuple[Xtts, XttsConfig]:
    cfg = XttsConfig()
    cfg.load_json(str(config_path))
    model = Xtts.init_from_config(cfg)
    model.load_checkpoint(cfg, checkpoint_path=str(ckpt_path), vocab_path=str(vocab_path), use_deepspeed=False)
    if device == "cuda" and torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
    model.eval()
    return model, cfg


def synth_one(
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
    ap = argparse.ArgumentParser(description="Synthesize fixed phrases for each XTTS checkpoint.")
    ap.add_argument("--run-dir", required=True, help="GPT_XTTS_FT run directory containing checkpoints.")
    ap.add_argument("--reference-wav", required=True, help="Reference wav path (speaker style).")
    ap.add_argument("--language", default="pt", help="XTTS language code.")
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Inference device.",
    )
    ap.add_argument("--out-dir", default="", help="Output directory (default: run-dir/outputs)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        print(f"[ERRO] run-dir inexistente: {run_dir}", file=sys.stderr)
        return 2

    ref_wav = Path(args.reference_wav).resolve()
    if not ref_wav.is_file():
        print(f"[ERRO] reference-wav inexistente: {ref_wav}", file=sys.stderr)
        return 2

    training_root = run_dir.parent
    config_path = training_root / "XTTS_v2.0_original_model_files" / "config.json"
    vocab_path = training_root / "XTTS_v2.0_original_model_files" / "vocab.json"
    if not config_path.is_file() or not vocab_path.is_file():
        print(f"[ERRO] config/vocab não encontrados em {training_root / 'XTTS_v2.0_original_model_files'}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (run_dir / "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = [
        "Hoje acordei cedo, preparei um café forte e organizei a mesa para estudar com calma.",
        "Enquanto o trem passava devagar, uma criança sorria e apontava para as nuvens alaranjadas.",
        "O pesquisador analisou os dados, escreveu um relatório objetivo e compartilhou as conclusões com a equipe.",
        "Estamos construindo uma rotina mais saudável, caminhando no bairro e cozinhando alimentos frescos todos os dias.",
        "A bibliotecária catalogou romances, dicionários e biografias, mantendo cada prateleira limpa e bem sinalizada.",
    ]

    models = discover_models(run_dir)
    print(f"[INFO] checkpoints detectados: {len(models)}")
    for sid, p in models:
        print(f"  - {sid}: {p.name}")

    # Optional reference copies for relatório/listening parity.
    for i, _ in enumerate(texts, start=1):
        tag = f"{i:02d}"
        ref_copy = out_dir / f"xtts_ref_frase{tag}.wav"
        wav, sr = torchaudio.load(str(ref_wav))
        torchaudio.save(str(ref_copy), wav, sr)

    generated: dict[str, list[str]] = {sid: [] for sid, _ in models}
    for sid, ckpt in models:
        print(f"[INFO] carregando modelo {sid} ({ckpt.name})")
        model, cfg = load_model(config_path, vocab_path, ckpt, args.device)
        for i, text in enumerate(texts, start=1):
            tag = f"{i:02d}"
            out_name = f"xtts_model_{sid}_frase{tag}.wav"
            out_wav = out_dir / out_name
            synth_one(model, cfg, ref_wav, text, args.language, out_wav)
            generated[sid].append(out_name)
            print(f"[OK] {out_name}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # CSV no formato solicitado: texto|arquivo_modelo1|arquivo_modelo2|...
    csv_path = out_dir / "xtts_outputs_map.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|")
        for idx, text in enumerate(texts):
            row = [text]
            for sid, _ in models:
                row.append(generated[sid][idx])
            w.writerow(row)

    manifest = {
        "run_dir": str(run_dir),
        "reference_wav": str(ref_wav),
        "language": args.language,
        "device": args.device,
        "models": [{"id": sid, "checkpoint": str(p)} for sid, p in models],
        "outputs_dir": str(out_dir),
        "csv_map": str(csv_path),
    }
    (out_dir / "xtts_outputs_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] outputs em: {out_dir}")
    print(f"[DONE] csv: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

