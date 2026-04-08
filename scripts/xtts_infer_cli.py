from __future__ import annotations

import argparse
import os
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


def main() -> int:
    ap = argparse.ArgumentParser(description="XTTS v2 fine-tuned inference (no Gradio).")
    ap.add_argument("--checkpoint", required=True, help="Path to fine-tuned checkpoint (.pth), e.g. best_model.pth")
    ap.add_argument("--config", required=True, help="Path to XTTS config.json for the fine-tuned run")
    ap.add_argument("--vocab", required=True, help="Path to vocab.json (tokenizer)")
    ap.add_argument("--speaker_wav", required=True, help="Reference speaker wav path")
    ap.add_argument("--text", required=True, help="Text to synthesize")
    ap.add_argument("--language", default="pt", help="Language code (default: pt)")
    ap.add_argument(
        "--out_wav",
        default="",
        help="Output wav path. Default: data/xtts-work/xtts_infer_<timestamp>.wav",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cpu|cuda (default auto)")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint).resolve()
    cfg_path = Path(args.config).resolve()
    vocab = Path(args.vocab).resolve()
    speaker_wav = Path(args.speaker_wav).resolve()

    if not ckpt.is_file():
        print(f"[xtts_infer_cli] missing checkpoint: {ckpt}", file=sys.stderr)
        return 2
    if not cfg_path.is_file():
        print(f"[xtts_infer_cli] missing config: {cfg_path}", file=sys.stderr)
        return 2
    if not vocab.is_file():
        print(f"[xtts_infer_cli] missing vocab: {vocab}", file=sys.stderr)
        return 2
    if not speaker_wav.is_file():
        print(f"[xtts_infer_cli] missing speaker wav: {speaker_wav}", file=sys.stderr)
        return 2

    out_wav = args.out_wav.strip()
    if not out_wav:
        out_dir = Path("data") / "xtts-work"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_wav = str(out_dir / f"xtts_infer_{time.strftime('%Y%m%d-%H%M%S')}.wav")
    out_wav_path = Path(out_wav).resolve()
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[xtts_infer_cli] device={args.device} cuda_available={torch.cuda.is_available()}")
    print(f"[xtts_infer_cli] checkpoint={ckpt}")
    print(f"[xtts_infer_cli] config={cfg_path}")
    print(f"[xtts_infer_cli] vocab={vocab}")
    print(f"[xtts_infer_cli] speaker_wav={speaker_wav}")
    print(f"[xtts_infer_cli] language={args.language}")
    print(f"[xtts_infer_cli] out_wav={out_wav_path}")

    t0 = time.perf_counter()
    config = XttsConfig()
    config.load_json(str(cfg_path))

    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path=str(ckpt), vocab_path=str(vocab), use_deepspeed=False)
    if args.device.lower().startswith("cuda") and torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
    print(f"[xtts_infer_cli] model loaded in {time.perf_counter() - t0:.2f}s")

    t1 = time.perf_counter()
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=str(speaker_wav),
        gpt_cond_len=model.config.gpt_cond_len,
        max_ref_length=model.config.max_ref_len,
        sound_norm_refs=model.config.sound_norm_refs,
    )
    print(f"[xtts_infer_cli] conditioning in {time.perf_counter() - t1:.2f}s")

    t2 = time.perf_counter()
    out = model.inference(
        text=args.text,
        language=args.language,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=model.config.temperature,
        length_penalty=model.config.length_penalty,
        repetition_penalty=model.config.repetition_penalty,
        top_k=model.config.top_k,
        top_p=model.config.top_p,
    )
    print(f"[xtts_infer_cli] inference in {time.perf_counter() - t2:.2f}s")

    wav = torch.tensor(out["wav"]).unsqueeze(0).cpu()
    torchaudio.save(str(out_wav_path), wav, 24000)
    print(f"[xtts_infer_cli] wrote {out_wav_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

