"""Conversão de mídia com PyAV (FFmpeg embutido — não precisa instalar nada).

Qualquer formato aceito (áudio ou vídeo) vira:
  - WAV 16 kHz mono PCM  -> entrada dos motores locais (local / WhisperX)
  - MP3 64 kbps mono     -> upload leve para engines de API (2 h ~ 55 MB)

Antes isso dependia do executável `ffmpeg` no PATH; com PyAV as mesmas
bibliotecas do FFmpeg vêm dentro do pacote Python, então o programa
empacotado funciona em qualquer Windows/Linux sem instalação extra.
"""
from __future__ import annotations

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


def _open(src: Path):
    import av

    try:
        container = av.open(str(src))
    except Exception as exc:  # noqa: BLE001 — av.error.* variam por versão
        raise MediaError(
            "Não consegui abrir o arquivo (pode estar corrompido ou protegido "
            f"por DRM): {exc}") from exc
    if not container.streams.audio:
        container.close()
        raise MediaError("O arquivo não tem trilha de áudio.")
    return container


def duration_seconds(src: Path) -> float:
    container = _open(src)
    try:
        if container.duration:
            return container.duration / 1_000_000  # av.time_base = microssegundos
        stream = container.streams.audio[0]
        if stream.duration and stream.time_base:
            return float(stream.duration * stream.time_base)
    finally:
        container.close()
    raise MediaError("Não foi possível ler a duração do arquivo.")


def _transcode(src: Path, dst: Path, codec: str, fmt: str, bit_rate: int | None) -> Path:
    import av

    container = _open(src)
    try:
        sample_fmt = "s16p" if codec == "libmp3lame" else "s16"   # lame exige planar
        resampler = av.AudioResampler(format=sample_fmt, layout="mono", rate=16000)
        with av.open(str(dst), mode="w", format=fmt) as out:
            ostream = out.add_stream(codec, rate=16000, layout="mono")
            if bit_rate:
                ostream.bit_rate = bit_rate
            ostream.format = sample_fmt
            for frame in container.decode(audio=0):
                frame.pts = None
                for rframe in resampler.resample(frame):
                    for packet in ostream.encode(rframe):
                        out.mux(packet)
            for rframe in resampler.resample(None):
                for packet in ostream.encode(rframe):
                    out.mux(packet)
            for packet in ostream.encode(None):
                out.mux(packet)
    except MediaError:
        raise
    except Exception as exc:  # noqa: BLE001
        dst.unlink(missing_ok=True)
        raise MediaError(
            "Falha ao converter o arquivo (o áudio pode estar corrompido "
            f"ou protegido por DRM): {exc}") from exc
    finally:
        container.close()
    return dst


def to_wav16k(src: Path, dst: Path) -> Path:
    """Extrai o áudio para WAV 16 kHz mono (entrada dos motores locais)."""
    return _transcode(src, dst, "pcm_s16le", "wav", None)


def to_mp3_asr(src: Path, dst: Path) -> Path:
    """Extrai o áudio para MP3 mono 64 kbps (upload rápido para APIs)."""
    return _transcode(src, dst, "libmp3lame", "mp3", 64_000)
