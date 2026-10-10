"""Contrato comum dos motores de transcrição.

Todo engine recebe um arquivo de áudio já normalizado e devolve uma lista de
Segment (início, fim, falante, texto). Trocar de motor não muda nada no resto
do app — worker, exports e UI só conhecem Segment.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol

ProgressCB = Callable[[str], None]  # recebe o nome da fase atual


@dataclass
class Segment:
    start: float          # segundos
    end: float            # segundos
    speaker: str          # rótulo cru do engine (ex.: "A", "SPEAKER_00")
    text: str


class TranscriptionEngine(Protocol):
    name: str

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: Optional[str],          # "pt", "en"... ou None = detectar
        min_speakers: Optional[int],
        max_speakers: Optional[int],
        progress: ProgressCB,
        vocabulary: Optional[list[str]] = None,   # nomes dos participantes
    ) -> list[Segment]:
        ...


def merge_segments(segments: list[Segment], max_gap: float = 1.5) -> list[Segment]:
    """Junta segmentos consecutivos do mesmo falante (pausas curtas),
    deixando a leitura no estilo Otter: um bloco por fala, não por frase."""
    merged: list[Segment] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        if (
            merged
            and merged[-1].speaker == seg.speaker
            and seg.start - merged[-1].end <= max_gap
        ):
            prev = merged[-1]
            prev.text = f"{prev.text} {text}".strip()
            prev.end = seg.end
        else:
            merged.append(Segment(seg.start, seg.end, seg.speaker, text))
    return merged
