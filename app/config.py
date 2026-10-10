"""Configuração central do Escriba — lida do ambiente / arquivo .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# --- Diretórios -------------------------------------------------------------
DATA_DIR = Path(os.getenv("ESCRIBA_DATA_DIR", BASE_DIR / "data"))
JOBS_DIR = DATA_DIR / "jobs"
PROFILE_DIR = DATA_DIR / "profile"          # perfil + assinatura do cliente
JOBS_DIR.mkdir(parents=True, exist_ok=True)
PROFILE_DIR.mkdir(parents=True, exist_ok=True)

# --- Motor de transcrição ---------------------------------------------------
# "assemblyai"  -> API na nuvem, com diarização nativa (recomendado sem GPU)
# "whisperx"    -> 100% local (Whisper + pyannote), recomendado com GPU NVIDIA
DEFAULT_ENGINE = os.getenv("ESCRIBA_ENGINE", "assemblyai")

# --- AssemblyAI --------------------------------------------------------------
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")

# --- WhisperX (local) --------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN", "")               # token Hugging Face (pyannote)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")   # auto | cuda | cpu
WHISPER_BATCH_SIZE = int(os.getenv("WHISPER_BATCH_SIZE", "8"))

# --- Ata automática (opcional, via API Anthropic) ----------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
SUMMARY_MODEL = os.getenv("ESCRIBA_SUMMARY_MODEL", "claude-sonnet-4-6")

# --- E-mail (envio da ata e da transcrição) ----------------------------------
# Qualquer provedor SMTP serve (Proton, Gmail com senha de app, Zoho, Outlook...)
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("SMTP_FROM", "") or SMTP_USER
SMTP_SECURITY = os.getenv("SMTP_SECURITY", "starttls").strip().lower()  # starttls | ssl | none
EMAIL_MAX_ATTACH_MB = int(os.getenv("ESCRIBA_EMAIL_MAX_MB", "20"))
EMAIL_MAX_RECIPIENTS = 20

# --- Upload -------------------------------------------------------------------
# Formatos aceitos (áudio e vídeo). O ffmpeg extrai/normaliza o áudio de todos.
ALLOWED_EXTENSIONS = {
    # áudio
    ".m4a", ".mp3", ".wav", ".wma", ".m4p", ".aac", ".ogg", ".flac",
    # vídeo
    ".mp4", ".mov", ".avi", ".mpeg", ".mpg", ".wmv", ".mkv", ".3gp", ".webm",
}
MAX_UPLOAD_BYTES = int(os.getenv("ESCRIBA_MAX_UPLOAD_MB", "6000")) * 1024 * 1024
MAX_SIGNATURE_BYTES = 5 * 1024 * 1024

DEFAULT_LANGUAGE = os.getenv("ESCRIBA_LANGUAGE", "pt")  # "auto" para detectar
