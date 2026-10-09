"""Pipeline de alto nível reutilizado pelo CLI e pela API."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import summary, transcribe
from .audio import AudioError, probe_duration
from .config import settings
from .utils import atomic_write_text, log, safe_slug


@dataclass
class ProcessResult:
    transcript: str
    transcript_path: Path | None
    ata: str | None
    ata_path: Path | None
    warnings: list[str]


def process_audio(
    audio_path: Path,
    out_dir: Path | None = None,
    make_ata: bool | None = None,
    language: str | None = None,
) -> ProcessResult:
    """Transcreve um arquivo de áudio e, opcionalmente, gera a ata.

    Args:
        audio_path: caminho do áudio de entrada.
        out_dir: diretório de saída (padrão: ESCRIBA_DATA_DIR).
        make_ata: força ligar/desligar a ata (padrão: ESCRIBA_SUMMARY_ENABLED).
        language: idioma (padrão: ESCRIBA_LANGUAGE / auto).

    Returns:
        ProcessResult com texto, ata e caminhos salvos.
    """
    audio_path = Path(audio_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve() if out_dir else settings.data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    # Validação de áudio (duração + extensão). Não bloqueia se ffprobe faltar.
    try:
        dur = probe_duration(audio_path)
        log.info("Áudio OK: %.1f s", dur)
    except AudioError as e:
        warnings.append(f"Validação de áudio: {e} (seguindo mesmo assim)")
        log.warning(warnings[-1])

    # Transcrição (com fallback entre motores).
    result = transcribe.transcribe(audio_path, language=language)
    warnings.extend(result.warnings)
    log.info("Transcrição concluída (%d chars, motor=%s).", len(result.text), result.engine)

    stem = safe_slug(audio_path.stem)
    transcript_path = out_dir / f"{stem}.transcript.txt"
    atomic_write_text(transcript_path, result.text)

    # Ata (opcional).
    ata_md: str | None = None
    ata_path: Path | None = None
    want_ata = settings.summary_enabled if make_ata is None else make_ata
    if want_ata:
        try:
            ata = summary.generate_ata(result.text)
            warnings.extend(ata.warnings)
            ata_md = ata.markdown
            ata_path = out_dir / f"{stem}.ata.md"
            atomic_write_text(ata_path, ata_md)
            log.info("Ata gerada (modelo=%s).", ata.model)
        except summary.SummaryUnavailableError as e:
            warnings.append(f"Ata não gerada: {e}")
            log.warning(warnings[-1])

    return ProcessResult(
        transcript=result.text,
        transcript_path=transcript_path,
        ata=ata_md,
        ata_path=ata_path,
        warnings=warnings,
    )
