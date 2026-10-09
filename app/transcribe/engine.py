"""Factory + orquestração de fallback entre motores de transcrição.

FIX #1 (escriba): o fallback roda DENTRO de try/except. Força-se o primário;
se ele levantar ProviderUnavailableError, cai no próximo motor. "Sobe sem erro"
não é o mesmo que "funciona sob falha" — por isso a seleção de disponibilidade
e a transcrição ficam ambas protegidas.
"""
from __future__ import annotations

from pathlib import Path

from ..config import settings
from ..utils import log
from .base import ProviderUnavailableError, TranscribeEngine, TranscriptResult
from .local_whisper import LocalWhisperEngine


def _build_chain() -> list[TranscribeEngine]:
    """Ordena os motores conforme ESCRIBA_TRANSCRIBE_ENGINE; local é sempre fallback."""
    local = LocalWhisperEngine()
    chain: list[TranscribeEngine] = []
    if settings.engine == "api":
        api = _maybe_api_engine()
        if api is not None:
            chain.append(api)
    chain.append(local)  # local (grátis) sempre disponível como base/fallback
    return chain


def _maybe_api_engine() -> TranscribeEngine | None:
    """Ponto de extensão: motor de API paga opcional (lazy-init do client).

    Mantido como stub para não acoplar um provider específico. Quando o código
    real do flashdrive trouxer o provider (ex.: AssemblyAI), implementar aqui
    seguindo a interface TranscribeEngine, com o client instanciado DENTRO do
    método (nunca no nível de módulo).
    """
    log.info("ESCRIBA_TRANSCRIBE_ENGINE=api, mas nenhum provider de API configurado; usando local.")
    return None


def transcribe(audio_path: Path, language: str | None = None) -> TranscriptResult:
    """Tenta cada motor na ordem; cai no próximo em ProviderUnavailableError."""
    chain = _build_chain()
    last_error: Exception | None = None
    warnings: list[str] = []
    for engine in chain:
        try:
            if not engine.available():
                warnings.append(f"{engine.name} indisponível, pulando.")
                continue
            result = engine.transcribe(audio_path, language=language)
            result.warnings = warnings + result.warnings
            return result
        except ProviderUnavailableError as e:  # FIX #1 — fallback no except
            log.warning("Motor %s falhou (%s); tentando fallback.", engine.name, e)
            warnings.append(f"{engine.name} falhou: {e}")
            last_error = e
    raise ProviderUnavailableError(
        "Nenhum motor de transcrição disponível. "
        + (f"Último erro: {last_error}" if last_error else "")
    )
