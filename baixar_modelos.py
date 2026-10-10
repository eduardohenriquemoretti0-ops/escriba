"""Baixa os modelos do motor local para models/ (rodar UMA vez, com internet).

Depois disso o Escriba transcreve e identifica falantes sem internet.
O build do executável chama este script e copia models/ junto do programa.

Uso:
    python baixar_modelos.py                 # modelo padrão (large-v3-turbo)
    python baixar_modelos.py small medium    # outros tamanhos do Whisper
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"

# Modelo de voz (embedding de falante) usado pela diarização local.
# WeSpeaker ResNet34 (Apache-2.0), distribuído pelo projeto sherpa-onnx.
DIARIZATION = {
    "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
           "speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx",
    "file": MODELS_DIR / "diarizacao" / "wespeaker_en_voxceleb_resnet34_LM.onnx",
    "sha256": "e9848563da86f263117134dfd7ad63c92355b37de492b55e325400c9d9c39012",
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  baixando {url}")
    with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    tmp.replace(dest)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def baixar_whisper(nome: str) -> None:
    from faster_whisper import download_model

    dest = MODELS_DIR / "whisper" / nome
    if (dest / "model.bin").exists():
        print(f"[ok] Whisper {nome} já está em {dest}")
        return
    print(f"[..] Whisper {nome} (pode levar alguns minutos)")
    download_model(nome, output_dir=str(dest))
    print(f"[ok] Whisper {nome} -> {dest}")


def baixar_diarizacao() -> None:
    dest = DIARIZATION["file"]
    if not dest.exists():
        print("[..] Modelo de identificação de falantes")
        _download(DIARIZATION["url"], dest)
    expected = DIARIZATION["sha256"]
    if expected and _sha256(dest) != expected:
        dest.unlink()
        raise SystemExit("Modelo de falantes corrompido (hash não confere). Rode de novo.")
    print(f"[ok] Falantes -> {dest}")


if __name__ == "__main__":
    nomes = sys.argv[1:] or ["large-v3-turbo"]
    for n in nomes:
        baixar_whisper(n)
    baixar_diarizacao()
    print("\nPronto. O Escriba já funciona sem internet.")
