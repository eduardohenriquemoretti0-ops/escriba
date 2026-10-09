"""Motor local: Faster-Whisper (grátis, roda offline na CPU/GPU)."""
from __future__ import annotations

from pathlib import Path

from ..config import settings
from ..utils import log
from .base import ProviderUnavailableError, TranscribeEngine, TranscriptResult


class LocalWhisperEngine(TranscribeEngine):
    name = "local-faster-whisper"

    def __init__(self) -> None:
        self._model = None  # lazy: só carrega quando for transcrever

    def available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except Exception:
            return False

    def _load(self):
        if self._model is None:
            import os

            from faster_whisper import WhisperModel

            # lê do ambiente em tempo de execução (permite override via --modelo/env)
            model = os.getenv("ESCRIBA_WHISPER_MODEL", settings.whisper_model)
            device = os.getenv("ESCRIBA_WHISPER_DEVICE", settings.whisper_device)
            compute = os.getenv("ESCRIBA_WHISPER_COMPUTE", settings.whisper_compute)
            log.info(
                "Carregando Faster-Whisper (modelo=%s device=%s compute=%s)...",
                model, device, compute,
            )
            self._model = WhisperModel(model, device=device, compute_type=compute)
        return self._model

    def transcribe(
        self, audio_path: Path, language: str | None = None
    ) -> TranscriptResult:
        if not self.available():
            raise ProviderUnavailableError(
                "faster-whisper não instalado (pip install faster-whisper)."
            )
        model = self._load()
        lang = language or settings.language
        try:
            segments, info = model.transcribe(
                str(audio_path), language=lang, beam_size=5, vad_filter=True
            )
            # segments é um gerador — materializa concatenando o texto.
            text = "".join(seg.text for seg in segments).strip()
        except Exception as e:  # erro real do motor → sinaliza indisponível p/ fallback
            raise ProviderUnavailableError(f"Falha no Faster-Whisper: {e}") from e
        return TranscriptResult(
            text=text,
            language=getattr(info, "language", lang),
            duration=getattr(info, "duration", None),
            engine=self.name,
        )
