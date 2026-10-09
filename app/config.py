"""Configuração central do escriba (lê .env / ambiente)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()  # .env do diretório atual
    load_dotenv(Path.home() / ".env")  # variáveis globais do operador
except Exception:
    pass  # dotenv é opcional; variáveis de ambiente ainda funcionam


def _expand(p: str) -> Path:
    return Path(os.path.expanduser(p)).resolve()


@dataclass(frozen=True)
class Settings:
    # transcrição
    engine: str
    whisper_model: str
    whisper_device: str
    whisper_compute: str
    language: str | None
    # sumarização
    summary_enabled: bool
    model_small: str
    model_large: str
    summary_threshold_chars: int
    max_chars: int
    # api / segurança
    api_token: str
    ratelimit_max: int
    ratelimit_window_sec: int
    max_upload_mb: int
    # armazenamento
    data_dir: Path
    cleanup_days: int


def load_settings() -> Settings:
    lang = os.getenv("ESCRIBA_LANGUAGE", "").strip()
    return Settings(
        engine=os.getenv("ESCRIBA_TRANSCRIBE_ENGINE", "local").strip().lower(),
        whisper_model=os.getenv("ESCRIBA_WHISPER_MODEL", "small").strip(),
        whisper_device=os.getenv("ESCRIBA_WHISPER_DEVICE", "cpu").strip(),
        whisper_compute=os.getenv("ESCRIBA_WHISPER_COMPUTE", "int8").strip(),
        language=lang or None,
        summary_enabled=os.getenv("ESCRIBA_SUMMARY_ENABLED", "false").strip().lower()
        in ("1", "true", "yes", "sim"),
        model_small=os.getenv("ESCRIBA_MODEL_SMALL", "claude-haiku-5-5").strip(),
        model_large=os.getenv("ESCRIBA_MODEL_LARGE", "claude-sonnet-5-5").strip(),
        summary_threshold_chars=int(os.getenv("ESCRIBA_SUMMARY_THRESHOLD_CHARS", "400000")),
        max_chars=int(os.getenv("ESCRIBA_MAX_CHARS", "1500000")),
        api_token=os.getenv("ESCRIBA_API_TOKEN", "").strip(),
        ratelimit_max=int(os.getenv("ESCRIBA_RATELIMIT_MAX", "60")),
        ratelimit_window_sec=int(os.getenv("ESCRIBA_RATELIMIT_WINDOW_SEC", "60")),
        max_upload_mb=int(os.getenv("ESCRIBA_MAX_UPLOAD_MB", "500")),
        data_dir=_expand(os.getenv("ESCRIBA_DATA_DIR", "~/escriba/data")),
        cleanup_days=int(os.getenv("ESCRIBA_CLEANUP_DAYS", "7")),
    )


settings = load_settings()
