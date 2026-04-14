#!/usr/bin/env python3
"""Valida pasta de dados para XTTS (PT-BR): exige wavs/ com audios; valida metadata.csv se existir."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

AUDIO_EXT = frozenset({".wav", ".mp3", ".flac"})


def main() -> int:
    ap = argparse.ArgumentParser(description="Valida data-root com wavs/ e metadata.csv opcional.")
    ap.add_argument("--data-root", type=Path, required=True, help="Pasta pai (contem wavs/).")
    args = ap.parse_args()
    root = args.data_root.resolve()
    if not root.is_dir():
        print(f"ERRO: pasta nao existe: {root}", file=sys.stderr)
        return 1
    wavs = root / "wavs"
    if not wavs.is_dir():
        print(f"ERRO: subpasta wavs/ nao encontrada em {root}", file=sys.stderr)
        print("  Crie wavs/ e coloque os .wav (ou .mp3/.flac) originais la dentro.", file=sys.stderr)
        return 1
    files = [p for p in wavs.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXT]
    if not files:
        print(f"ERRO: nenhum audio ({', '.join(sorted(AUDIO_EXT))}) em {wavs}", file=sys.stderr)
        return 1
    print(f"OK: {len(files)} arquivo(s) de audio em wavs/")

    meta = root / "metadata.csv"
    if not meta.is_file():
        print("AVISO: metadata.csv nao encontrado — o prep pode criar CSVs a partir de outros fluxos.")
        print("  Se voce usar apenas metadata.csv + wavs/, gere metadata.csv no formato id|texto (Coqui).")
        return 0

    raw = meta.read_text(encoding="utf-8-sig", errors="replace")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        print("ERRO: metadata.csv vazio.", file=sys.stderr)
        return 2

    erros = 0
    for i, line in enumerate(lines, 1):
        if "|" not in line:
            print(f"ERRO linha {i}: falta '|' (formato esperado: nome.wav|texto do utterance)", file=sys.stderr)
            erros += 1
            continue
        left, _txt = line.split("|", 1)
        rel = left.strip()
        if not rel:
            print(f"ERRO linha {i}: lado esquerdo do | vazio", file=sys.stderr)
            erros += 1
            continue
        if not any(rel.lower().endswith(ext) for ext in AUDIO_EXT):
            rel = rel + ".wav"
        rel_norm = rel.replace("\\", "/")
        p1 = wavs / rel
        p2 = wavs / Path(rel).name
        p3 = root / rel
        ok = p1.is_file() or p2.is_file() or p3.is_file()
        if not ok and rel_norm.lower().startswith("wavs/"):
            ok = (wavs / rel_norm[5:].lstrip("/")).is_file()
        if ok:
            continue
        print(f"AVISO linha {i}: audio nao encontrado para '{left.strip()}'", file=sys.stderr)
        erros += 1

    if erros:
        print(f"ERRO: {erros} linha(s) com problema no metadata.csv (veja avisos acima).", file=sys.stderr)
        return 2
    print("OK: metadata.csv parece consistente com os arquivos em wavs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
