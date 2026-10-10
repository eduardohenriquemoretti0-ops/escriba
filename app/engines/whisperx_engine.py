"""Motor WhisperX (100% local).

Whisper large-v3 + alinhamento + diarização pyannote. Nada sai da sua máquina —
ideal para reuniões sensíveis (clientes corporativos, órgãos públicos).

Requisitos práticos:
  - GPU NVIDIA com >= 8 GB VRAM para large-v3 confortável (RTX 3060/3060 Ti+).
    Em CPU funciona, mas 2 h de áudio podem levar muitas horas — nesse caso
    use WHISPER_MODEL=small ou medium.
  - HF_TOKEN no .env + aceite dos termos dos modelos pyannote no Hugging Face
    (veja o README, seção WhisperX).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .. import config
from .base import ProgressCB, Segment

_model_cache: dict = {}


def _pick_device() -> tuple[str, str]:
    """Retorna (device, compute_type)."""
    if config.WHISPER_DEVICE in ("cuda", "cpu"):
        dev = config.WHISPER_DEVICE
    else:
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            dev = "cpu"
    return dev, ("float16" if dev == "cuda" else "int8")


class WhisperXEngine:
    name = "whisperx"

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: Optional[str],
        min_speakers: Optional[int],
        max_speakers: Optional[int],
        progress: ProgressCB,
        vocabulary: Optional[list[str]] = None,   # reservado (não usado aqui)
    ) -> list[Segment]:
        try:
            import whisperx
        except ImportError as exc:
            raise RuntimeError(
                "WhisperX não está instalado. Rode: "
                "pip install -r requirements-local.txt"
            ) from exc

        device, compute_type = _pick_device()

        progress(f"Carregando modelo {config.WHISPERX_MODEL} ({device})")
        cache_key = (config.WHISPERX_MODEL, device)
        if cache_key not in _model_cache:
            _model_cache[cache_key] = whisperx.load_model(
                config.WHISPERX_MODEL, device, compute_type=compute_type
            )
        model = _model_cache[cache_key]

        audio = whisperx.load_audio(str(audio_path))

        progress("Transcrevendo (Whisper)")
        result = model.transcribe(
            audio,
            batch_size=config.WHISPER_BATCH_SIZE,
            language=language or None,
        )
        lang = result.get("language") or language or "pt"

        progress("Alinhando palavras")
        align_model, meta = whisperx.load_align_model(
            language_code=lang, device=device
        )
        result = whisperx.align(
            result["segments"], align_model, meta, audio, device,
            return_char_alignments=False,
        )

        progress("Identificando falantes (pyannote)")
        if not config.HF_TOKEN:
            raise RuntimeError(
                "HF_TOKEN não configurado no .env — necessário para a "
                "diarização pyannote (veja o README)."
            )
        try:  # a classe mudou de lugar entre versões do whisperx
            from whisperx.diarize import DiarizationPipeline
        except ImportError:
            DiarizationPipeline = whisperx.DiarizationPipeline  # type: ignore

        diarizer = DiarizationPipeline(
            use_auth_token=config.HF_TOKEN, device=device
        )
        dia_kwargs = {}
        if min_speakers:
            dia_kwargs["min_speakers"] = min_speakers
        if max_speakers:
            dia_kwargs["max_speakers"] = max_speakers
        dia_segments = diarizer(audio, **dia_kwargs)
        result = whisperx.assign_word_speakers(dia_segments, result)

        segments: list[Segment] = []
        for seg in result.get("segments", []):
            segments.append(
                Segment(
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    speaker=str(seg.get("speaker", "SPEAKER_?")),
                    text=(seg.get("text") or "").strip(),
                )
            )
        return segments
