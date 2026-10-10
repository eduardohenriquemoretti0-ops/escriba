"""Conversão de mídia com ffmpeg.

Qualquer formato aceito (áudio ou vídeo) vira:
  - WAV 16 kHz mono PCM  -> entrada ideal do WhisperX local
  - MP3 64 kbps mono     -> upload leve para engines de API (2 h ~ 55 MB)
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from . import config


class MediaError(RuntimeError):
    pass


def validate_extension(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        aceitos = " ".join(sorted(config.ALLOWED_EXTENSIONS))
        raise MediaError(f"Formato '{ext}' não suportado. Aceitos: {aceitos}")
    return ext


def _run_ffmpeg(args: list[str]) -> None:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-y", *args],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-6:]
        raise MediaError(
            "Falha ao converter o arquivo (o áudio pode estar corrompido "
            "ou protegido por DRM).\n" + "\n".join(tail)
        )


def duration_seconds(src: Path) -> float:
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(src),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        raise MediaError("Não foi possível ler a duração do arquivo.") from None


def to_wav16k(src: Path, dst: Path) -> Path:
    """Extrai o áudio para WAV 16 kHz mono (entrada do WhisperX)."""
    _run_ffmpeg(["-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
                 "-c:a", "pcm_s16le", str(dst)])
    return dst


def to_mp3_asr(src: Path, dst: Path) -> Path:
    """Extrai o áudio para MP3 mono 64 kbps (upload rápido para APIs)."""
    _run_ffmpeg(["-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
                 "-b:a", "64k", str(dst)])
    return dst
