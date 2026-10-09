"""Interface comum dos motores de transcrição."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class ProviderUnavailableError(Exception):
    """O motor de transcrição não está disponível (lib ausente, API fora, etc.)."""


@dataclass
class TranscriptResult:
    text: str
    language: str | None = None
    duration: float | None = None
    engine: str = ""
    warnings: list[str] = field(default_factory=list)


class TranscribeEngine(Protocol):
    """Contrato estável — trocar de motor não muda quem chama."""

    name: str

    def available(self) -> bool:
        """Checa disponibilidade SEM levantar exceção."""
        ...

    def transcribe(
        self, audio_path: Path, language: str | None = None
    ) -> TranscriptResult:
        """Transcreve; levanta ProviderUnavailableError se indisponível."""
        ...
