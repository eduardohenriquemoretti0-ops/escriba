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
# NVIDIA NeMo TitaNet-small (CC-BY-4.0), convertido p/ ONNX pelo projeto sherpa-onnx.
# Escolhido num teste com reunião real: o mais rápido e o que mais concordou com
# os outros dois modelos avaliados (WeSpeaker ResNet34 e 3D-Speaker ERes2Net).
DIARIZATION = {
    "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
           "speaker-recongition-models/nemo_en_titanet_small.onnx",
    "file": MODELS_DIR / "diarizacao" / "nemo_en_titanet_small.onnx",
    "sha256": "ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e",
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
