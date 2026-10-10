"""Exportação da transcrição em TXT (estilo Otter), SRT, VTT e JSON.

Também produz a timeline compacta de falantes usada na UI
(no máximo ~240 blocos, para a lista carregar leve).
"""
from __future__ import annotations

import json
from typing import Optional

from .engines.base import Segment


def hms(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _srt_time(seconds: float) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _vtt_time(seconds: float) -> str:
    return _srt_time(seconds).replace(",", ".")


def speaker_map(segments: list[Segment]) -> dict[str, str]:
    """Mapeia rótulos crus (A / SPEAKER_00 / ...) -> 'Falante 1..N'
    na ordem em que aparecem na reunião."""
    mapping: dict[str, str] = {}
    for seg in segments:
        if seg.speaker not in mapping:
            mapping[seg.speaker] = f"Falante {len(mapping) + 1}"
    return mapping


def apply_names(default_names: dict[str, str],
                overrides: dict[str, str]) -> dict[str, str]:
    """Troca 'Falante N' pelo nome real, quando o cliente informou um."""
    out = {}
    for raw, label in default_names.items():
        real = (overrides.get(label) or "").strip()
        out[raw] = real or label
    return out


def to_txt(segments: list[Segment], names: dict[str, str],
           title: str, duration: Optional[float]) -> str:
    lines = [title, ""]
    if duration:
        lines.insert(1, f"Duração: {hms(duration)} · Falantes: "
                        f"{len(set(names.values()))}")
    for seg in segments:
        lines.append(f"{names[seg.speaker]} · {hms(seg.start)}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def to_srt(segments: list[Segment], names: dict[str, str]) -> str:
    blocks = []
    for i, seg in enumerate(segments, 1):
        blocks.append(
            f"{i}\n{_srt_time(seg.start)} --> {_srt_time(seg.end)}\n"
            f"{names[seg.speaker]}: {seg.text}\n"
        )
    return "\n".join(blocks)


def to_vtt(segments: list[Segment], names: dict[str, str]) -> str:
    blocks = ["WEBVTT", ""]
    for seg in segments:
        blocks.append(
            f"{_vtt_time(seg.start)} --> {_vtt_time(seg.end)}\n"
            f"<v {names[seg.speaker]}>{seg.text}\n"
        )
    return "\n".join(blocks)


def to_json(segments: list[Segment], names: dict[str, str],
            meta: dict) -> str:
    payload = {
        **meta,
        "falantes": names,
        "segmentos": [
            {
                "inicio": round(seg.start, 3),
                "fim": round(seg.end, 3),
                "falante": names[seg.speaker],
                "falante_original": seg.speaker,
                "texto": seg.text,
            }
            for seg in segments
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_timeline(segments: list[Segment], names: dict[str, str],
                   max_blocks: int = 240) -> list[list]:
    """[[inicio, fim, indice_do_falante], ...] — reamostrada se necessária.

    Dois falantes mapeados para o mesmo nome compartilham o mesmo índice
    (mesma cor), então a ordem é deduplicada."""
    order = list(dict.fromkeys(names.values()))
    blocks = [
        [round(s.start, 1), round(s.end, 1), order.index(names[s.speaker])]
        for s in segments
        if s.end > s.start
    ]
    if len(blocks) <= max_blocks:
        return blocks
    step = len(blocks) / max_blocks
    return [blocks[int(i * step)] for i in range(max_blocks)]
