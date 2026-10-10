"""API FastAPI do escriba.

Ordem de segurança (FIX C4): o rate limit roda como MIDDLEWARE GLOBAL, ANTES
de qualquer verificação de auth, para TODOS os métodos, com chave = IP+rota
(não o token — trocar token não reinicia a cota). A auth (Bearer) é verificada
por dependência nas rotas, com comparação constant-time.
"""
from __future__ import annotations

import shutil
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import JSONResponse, PlainTextResponse

from .config import settings
from .jobs import JobStatus, JobStore
from .pipeline import process_audio
from .security import RateLimiter, token_matches, valid_job_id
from .transcribe.base import ProviderUnavailableError
from .utils import log, safe_slug

app = FastAPI(title="escriba", version="0.1.0")
store = JobStore(ttl_seconds=settings.cleanup_days * 86400)
limiter = RateLimiter(settings.ratelimit_max, settings.ratelimit_window_sec)


@app.middleware("http")
async def rate_limit_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """FIX C4 — rate limit ANTES do auth, por IP+rota, em todos os métodos."""
    client_ip = request.client.host if request.client else "desconhecido"
    key = f"{client_ip}:{request.url.path}"
    if not limiter.allow(key):
        return JSONResponse({"error": "rate limited"}, status_code=429)
    return await call_next(request)


def require_auth(authorization: str = Header(default="")) -> None:
    """Auth Bearer (constant-time). Desligada se ESCRIBA_API_TOKEN vazio."""
    if not settings.api_token:
        return  # sem token configurado = modo local aberto
    prefix = "Bearer "
    provided = authorization[len(prefix):] if authorization.startswith(prefix) else ""
    if not token_matches(provided, settings.api_token):
        raise HTTPException(status_code=401, detail="não autorizado")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "version": app.version}


@app.post("/transcrever")
async def transcrever(
    arquivo: UploadFile,
    ata: bool = False,
    _: None = Depends(require_auth),
) -> dict[str, Any]:
    """Recebe um áudio, transcreve (e opcionalmente gera a ata)."""
    max_bytes = settings.max_upload_mb * 1024 * 1024
    job = store.create(filename=safe_slug(arquivo.filename or "audio"))
    tmp_dir = Path(tempfile.mkdtemp(prefix="escriba-"))
    tmp_path = tmp_dir / safe_slug(arquivo.filename or "audio")
    try:
        # Grava o upload em disco com limite de tamanho.
        size = 0
        with open(tmp_path, "wb") as f:
            while chunk := await arquivo.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(status_code=413, detail="arquivo muito grande")
                f.write(chunk)

        job.status = JobStatus.RUNNING
        job.touch()
        res = process_audio(tmp_path, make_ata=ata)
        job.transcript = res.transcript
        job.ata = res.ata
        job.warnings = res.warnings
        job.status = JobStatus.DONE
        job.touch()
        return job.public() | {"transcript": res.transcript, "ata": res.ata}
    except ProviderUnavailableError as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.touch()
        raise HTTPException(status_code=503, detail=str(e)) from e
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)  # não guarda o áudio do cliente


@app.get("/jobs/{job_id}")
def get_job(job_id: str, _: None = Depends(require_auth)) -> dict[str, Any]:
    if not valid_job_id(job_id):  # FIX M2
        raise HTTPException(status_code=400, detail="job_id inválido")
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job não encontrado")
    return job.public()


@app.get("/jobs/{job_id}/transcript", response_class=PlainTextResponse)
def get_transcript(job_id: str, _: None = Depends(require_auth)) -> str:
    if not valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="job_id inválido")
    job = store.get(job_id)
    if job is None or job.transcript is None:
        raise HTTPException(status_code=404, detail="transcrição indisponível")
    return job.transcript


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str, _: None = Depends(require_auth)) -> dict[str, Any]:
    if not valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="job_id inválido")
    if not store.delete(job_id):
        raise HTTPException(status_code=404, detail="job não encontrado")
    return {"deleted": job_id}
