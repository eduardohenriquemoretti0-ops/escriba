"""Configuração central do Escriba — lida do ambiente / arquivo .env."""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Rodando como programa empacotado (PyInstaller) ou pelo código-fonte?
FROZEN = getattr(sys, "frozen", False)
if FROZEN:
    # pasta do executável: models/ e .env ficam ao lado dele
    BASE_DIR = Path(sys.executable).resolve().parent
    # dados num lugar sempre gravável (Program Files não é)
    _DEFAULT_DATA = Path.home() / "Escriba"
else:
    BASE_DIR = Path(__file__).resolve().parent.parent
    _DEFAULT_DATA = BASE_DIR / "data"

load_dotenv(BASE_DIR / ".env")
load_dotenv(_DEFAULT_DATA / ".env")   # alternativa editável pelo usuário

# --- Diretórios -------------------------------------------------------------
DATA_DIR = Path(os.getenv("ESCRIBA_DATA_DIR", _DEFAULT_DATA))
MODELS_DIR = Path(os.getenv("ESCRIBA_MODELS_DIR", BASE_DIR / "models"))
JOBS_DIR = DATA_DIR / "jobs"
PROFILE_DIR = DATA_DIR / "profile"          # perfil + assinatura do cliente
JOBS_DIR.mkdir(parents=True, exist_ok=True)
PROFILE_DIR.mkdir(parents=True, exist_ok=True)

# --- Motor de transcrição ---------------------------------------------------
# "local"       -> 100% offline: Faster-Whisper + diarização ONNX (padrão)
# "assemblyai"  -> API na nuvem (precisa de internet e chave)
# "whisperx"    -> local com PyTorch + pyannote (GPU NVIDIA, instalação pesada)
DEFAULT_ENGINE = os.getenv("ESCRIBA_ENGINE", "local")

# --- AssemblyAI --------------------------------------------------------------
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")

# --- Motor local (Faster-Whisper + diarização ONNX) -------------------------
# large-v3-turbo: qualidade próxima do large-v3, bem mais rápido na CPU.
# Máquina fraca: small (rápido, erra mais) | medium
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")      # auto | cuda | cpu
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "auto")    # auto | int8 | float16...
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
CPU_THREADS = int(os.getenv("ESCRIBA_CPU_THREADS", str(os.cpu_count() or 4)))
DIARIZATION_MODEL = os.getenv(
    "ESCRIBA_DIARIZATION_MODEL",
    str(MODELS_DIR / "diarizacao" / "wespeaker_en_voxceleb_resnet34_LM.onnx"))
# Termos que o Whisper deve preferir em toda reunião (empresas, produtos...)
# separados por vírgula. Os participantes de cada reunião entram sozinhos.
EXTRA_TERMS = [t.strip() for t in os.getenv("ESCRIBA_TERMOS", "").split(",") if t.strip()]

# --- WhisperX (local com PyTorch, opcional) ---------------------------------
HF_TOKEN = os.getenv("HF_TOKEN", "")               # token Hugging Face (pyannote)
WHISPERX_MODEL = os.getenv("WHISPERX_MODEL", "large-v3")
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
