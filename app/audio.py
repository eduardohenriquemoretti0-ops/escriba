"""Validação e sondagem de áudio via ffprobe.

FIX #5 (escriba): ffprobe/ffmpeg SEMPRE com `-protocol_whitelist file` em
input controlado pelo usuário, para evitar SSRF/leitura de arquivo arbitrário
via protocolos `concat:`/`subfile:`/`http:` embutidos no conteúdo.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .utils import log

# MIME/extensões de áudio-vídeo aceitas (validar antes de chamar ffmpeg)
ALLOWED_SUFFIXES = {
    ".wav", ".mp3", ".m4a", ".ogg", ".oga", ".flac", ".aac",
    ".mp4", ".mov", ".webm", ".mkv", ".opus", ".wma", ".amr",
}


class AudioError(Exception):
    """Falha ao validar/sondar o arquivo de áudio."""


def check_suffix(path: Path) -> None:
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise AudioError(f"Extensão não suportada: {path.suffix!r}")


def probe_duration(path: Path, timeout: int = 20) -> float:
    """Retorna a duração em segundos. ffprobe restrito a protocolo `file`."""
    path = Path(path)
    if not path.is_file():
        raise AudioError(f"Arquivo não encontrado: {path}")
    check_suffix(path)
    cmd = [
        "ffprobe", "-v", "error",
        "-protocol_whitelist", "file",      # FIX #5 — anti-SSRF
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    ]
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except FileNotFoundError as e:
        raise AudioError("ffprobe não encontrado (instale ffmpeg).") from e
    except subprocess.TimeoutExpired as e:
        raise AudioError("ffprobe excedeu o tempo limite.") from e
    if out.returncode != 0:
        log.warning("ffprobe falhou: %s", out.stderr.strip()[:300])
        raise AudioError("Não foi possível ler o áudio (arquivo inválido?).")
    try:
        data = json.loads(out.stdout or "{}")
        return float(data["format"]["duration"])
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise AudioError("Metadados de duração ausentes.") from e
