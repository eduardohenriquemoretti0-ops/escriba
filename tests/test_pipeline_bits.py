"""Testes de escrita atômica, slug e fallback de transcrição."""
from pathlib import Path

import pytest

from app.transcribe import engine
from app.transcribe.base import (
    ProviderUnavailableError,
    TranscribeEngine,
    TranscriptResult,
)
from app.utils import atomic_write_text, safe_slug


def test_atomic_write(tmp_path: Path):
    alvo = tmp_path / "sub" / "saida.txt"
    atomic_write_text(alvo, "conteúdo")
    assert alvo.read_text(encoding="utf-8") == "conteúdo"
    assert not (alvo.with_suffix(".txt.tmp")).exists()  # temp removido


def test_safe_slug_sem_traversal():
    assert safe_slug("../../etc/passwd") == "passwd"
    assert safe_slug("reunião 2026.m4a").endswith(".m4a")


class _FalhaEngine(TranscribeEngine):
    name = "falha"

    def available(self) -> bool:
        return True

    def transcribe(self, audio_path, language=None):
        raise ProviderUnavailableError("forçado")


class _OkEngine(TranscribeEngine):
    name = "ok"

    def available(self) -> bool:
        return True

    def transcribe(self, audio_path, language=None):
        return TranscriptResult(text="oi", engine=self.name)


def test_fallback_dispara_no_except(monkeypatch):
    """FIX #1: primário forçado a falhar → cai no fallback (dentro do except)."""
    monkeypatch.setattr(engine, "_build_chain", lambda: [_FalhaEngine(), _OkEngine()])
    res = engine.transcribe(Path("qualquer.wav"))
    assert res.text == "oi"
    assert any("falha" in w for w in res.warnings)


def test_todos_falham_levanta(monkeypatch):
    monkeypatch.setattr(engine, "_build_chain", lambda: [_FalhaEngine()])
    with pytest.raises(ProviderUnavailableError):
        engine.transcribe(Path("qualquer.wav"))
