"""API do Escriba (FastAPI)."""
from __future__ import annotations

import os
import re
import secrets
import shutil
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import json

from . import (config, exports, irmaos, jobs, mailer, media, pdfdocs, profile,
               review, vozes, worker)

app = FastAPI(title="Escriba", docs_url="/api/docs")

# Modo programa (escriba_app.py): só a janela do Escriba, que recebeu o token
# desta sessão, pode usar a API. Outra aba de navegador ou outro programa que
# descubra a porta recebe 403 — evita que um site malicioso aberto no
# navegador dispare ações no Escriba via 127.0.0.1.
_SESSION_TOKEN = os.getenv("ESCRIBA_SESSION_TOKEN", "")
_COOKIE = "escriba_sessao"


@app.middleware("http")
async def _session_guard(request: Request, call_next):
    if not _SESSION_TOKEN or request.url.path == "/api/health":
        return await call_next(request)
    given = request.query_params.get("t")
    if given is not None and secrets.compare_digest(given, _SESSION_TOKEN):
        response = await call_next(request)
        response.set_cookie(_COOKIE, _SESSION_TOKEN, httponly=True, samesite="strict")
        return response
    cookie = request.cookies.get(_COOKIE, "")
    if not secrets.compare_digest(cookie, _SESSION_TOKEN):
        return JSONResponse({"detail": "Sessão inválida."}, status_code=403)
    return await call_next(request)

STATIC_DIR = Path(__file__).parent / "static"
_JOB_ID = re.compile(r"^[0-9a-f]{12}$")

# arquivos que existem em disco; os PDFs são gerados na hora (ver _file_for)
_MIME = {
    "txt": ("transcricao.txt", "text/plain; charset=utf-8", "txt"),
    "srt": ("transcricao.srt", "application/x-subrip", "srt"),
    "vtt": ("transcricao.vtt", "text/vtt", "vtt"),
    "json": ("transcricao.json", "application/json", "json"),
    "ata": ("ata.md", "text/markdown; charset=utf-8", "md"),
}
_PDF_KINDS = {"ata_pdf": "ata", "transcricao_pdf": "transcricao"}
_EMAIL_KINDS = ["ata_pdf", "transcricao_pdf", "ata", "txt", "srt", "vtt", "json"]


@app.on_event("startup")
def _startup() -> None:
    worker.start()
    for job_id in jobs.recover_interrupted():
        worker.enqueue(job_id)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _get_job(job_id: str) -> jobs.Job:
    job = jobs.load(job_id) if _JOB_ID.match(job_id) else None
    if job is None:
        raise HTTPException(status_code=404, detail="Job não encontrado.")
    return job


def _get_done_job(job_id: str) -> jobs.Job:
    job = _get_job(job_id)
    if job.status != jobs.STATUS_DONE:
        raise HTTPException(
            status_code=409,
            detail="A transcrição ainda não terminou — aguarde concluir.")
    return job


def _split_names(raw: str) -> list[str]:
    return review.clean_participants(re.split(r"[\n;,]+", raw or ""))


def _file_for(job: jobs.Job, kind: str) -> tuple[Path, str, str]:
    """(caminho, mime, nome de download) — gera o PDF se for o caso."""
    stem = Path(job.filename).stem
    if kind in _MIME:
        filename, mime, ext = _MIME[kind]
        path = job.path(filename)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Arquivo não gerado.")
        return path, mime, f"{stem}.{ext}"
    if kind in _PDF_KINDS:
        try:
            if kind == "ata_pdf":
                path = pdfdocs.ata_pdf(job)
            else:
                path = pdfdocs.transcript_pdf(job)
        except pdfdocs.PdfError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 — WeasyPrint sem libs, etc.
            raise HTTPException(
                status_code=500,
                detail=f"Não consegui gerar o PDF: {exc}. Confira se as "
                       "bibliotecas do WeasyPrint estão instaladas (README).",
            ) from exc
        return path, "application/pdf", f"{stem}_{_PDF_KINDS[kind]}.pdf"
    raise HTTPException(status_code=404, detail="Formato inválido.")


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/config")
def get_config():
    return {
        "ata_disponivel": bool(config.ANTHROPIC_API_KEY),
        "email_configurado": mailer.configured(),
        "whisperx_instalado": _whisperx_installed(),
    }


def _whisperx_installed() -> bool:
    """WhisperX é opcional (PyTorch + GPU): fica fora do programa empacotado."""
    import importlib.util
    return importlib.util.find_spec("whisperx") is not None


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    engine: str = Form(config.DEFAULT_ENGINE),
    language: str = Form(config.DEFAULT_LANGUAGE),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    want_summary: bool = Form(False),
    participants: str = Form(""),
):
    try:
        ext = media.validate_extension(file.filename or "")
    except media.MediaError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc

    if engine not in ("local", "assemblyai", "whisperx"):
        raise HTTPException(status_code=422, detail="Engine inválido.")
    if engine == "whisperx" and not _whisperx_installed():
        raise HTTPException(
            status_code=422,
            detail="O WhisperX não está instalado neste computador (exige PyTorch "
                   "e GPU NVIDIA de 8 GB+). Use o motor Local, que funciona offline.")

    job = jobs.new_job(
        filename=file.filename or f"arquivo{ext}",
        engine=engine,
        language="" if language == "auto" else language,
        min_speakers=min_speakers or None,
        max_speakers=max_speakers or None,
        want_summary=want_summary,
        participants=_split_names(participants),
    )

    dest = job.path(f"original{ext}")
    size = 0
    with dest.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > config.MAX_UPLOAD_BYTES:
                out.close()
                shutil.rmtree(job.dir, ignore_errors=True)
                raise HTTPException(
                    status_code=413,
                    detail="Arquivo maior que o limite configurado "
                           "(ESCRIBA_MAX_UPLOAD_MB).",
                )
            out.write(chunk)

    job.set_phase("Aguardando na fila")
    worker.enqueue(job.id)
    return {"id": job.id}


@app.get("/api/jobs")
def list_all():
    return [j.__dict__ for j in jobs.list_jobs()]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return _get_job(job_id).__dict__


@app.get("/api/jobs/{job_id}/file/{kind}")
def download(job_id: str, kind: str):
    job = _get_job(job_id)
    path, mime, name = _file_for(job, kind)
    return FileResponse(path, media_type=mime, filename=name)


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    job = _get_job(job_id)
    if job.status == jobs.STATUS_RUNNING or job.ata_status == jobs.ATA_GERANDO:
        raise HTTPException(
            status_code=409,
            detail="Job em processamento — aguarde concluir para excluir.")
    shutil.rmtree(job.dir, ignore_errors=True)
    return {"ok": True}


# --------------------------------------------------------------------------
# Revisão: nomes dos participantes e grafia
# --------------------------------------------------------------------------
class ReviewIn(BaseModel):
    participants: Optional[list[str]] = None
    speaker_names: Optional[dict[str, str]] = None
    corrections: Optional[list[dict]] = None
    meeting: Optional[dict] = None


def _review_payload(job: jobs.Job) -> dict:
    return {
        "participants": job.participants,
        "speaker_names": job.speaker_names,
        "corrections": job.corrections,
        "meeting": {"title": "", "date": "", "place": "", **job.meeting},
        "speakers": review.speakers_overview(job),
        "suggestions": review.suggest_corrections(job),
        "ata_existe": job.path("ata.md").exists(),
        "ata_status": job.ata_status,
        "vozes_info": job.vozes_info,
        "aprende_vozes": job.path("vozes.json").exists(),
        # confirmados na revisão que ainda não estão na lista: o usuário decide
        "nomes_novos": irmaos.ausentes(job.speaker_names.values(),
                                       irmaos.listar_seguro()),
    }


@app.get("/api/jobs/{job_id}/review")
def get_review(job_id: str):
    return _review_payload(_get_done_job(job_id))


@app.put("/api/jobs/{job_id}/review")
def put_review(job_id: str, body: ReviewIn):
    job = _get_done_job(job_id)
    labels = {row["label"] for row in review.speakers_overview(job)}

    names = None
    if body.speaker_names is not None:
        names = {}
        for label, value in body.speaker_names.items():
            clean = review.clean_name(value)
            if label in labels and clean:
                names[label] = clean

    participants = (review.clean_participants(body.participants)
                    if body.participants is not None else None)
    corrections = (review.clean_corrections(body.corrections)
                   if body.corrections is not None else None)
    meeting = None
    if body.meeting is not None:
        meeting = review.clean_meeting(body.meeting)

    changed = {"stale": False}

    def mutate(j: jobs.Job) -> None:
        before = (j.participants, j.speaker_names, j.corrections)
        if participants is not None:
            j.participants = participants
        if names is not None:
            j.speaker_names = names
        if corrections is not None:
            j.corrections = corrections
        if meeting is not None:
            j.meeting = meeting
        changed["stale"] = before != (j.participants, j.speaker_names,
                                      j.corrections)

    jobs.update(job_id, mutate)
    review.render_outputs(job_id)

    job = _get_job(job_id)
    learned = _learn_voices(job) if names is not None else []
    payload = _review_payload(job)
    payload["ata_desatualizada"] = bool(
        changed["stale"] and job.path("ata.md").exists())
    payload["vozes_aprendidas"] = learned
    return payload


def _learn_voices(job: jobs.Job) -> list[str]:
    """Nomes confirmados na revisão viram memória de voz (motor local).

    A fonte é só `job.speaker_names`, o que o usuário digitou ou escolheu no
    campo da revisão. Sugestões (`job.nomes_auto`) nunca entram aqui. Um
    campo vazio não aprende nada e apaga a contribuição desta reunião."""
    vfile = job.path("vozes.json")
    if not vfile.exists():
        return []
    voices = json.loads(vfile.read_text(encoding="utf-8"))
    display = exports.speaker_map(review.load_raw(job))      # "A" -> "Falante 1"
    assignments = {raw: job.speaker_names.get(label, "")
                   for raw, label in display.items() if raw in voices}
    return vozes.learn(job.id, assignments, voices)


# --------------------------------------------------------------------------
# Vozes conhecidas (memória de vozes)
# --------------------------------------------------------------------------
@app.get("/api/vozes")
def list_voices():
    return {"pessoas": vozes.list_people()}


@app.delete("/api/vozes/{nome}")
def forget_voice(nome: str):
    if not vozes.forget(nome):
        raise HTTPException(status_code=404, detail="Voz não encontrada.")
    return {"ok": True}


# --------------------------------------------------------------------------
# Irmãos / nomes conhecidos (lista de nomes, sem biometria)
# --------------------------------------------------------------------------
_IRMAO_ID = re.compile(r"^[0-9a-f]{12}$")


class IrmaoIn(BaseModel):
    nome: str = Field("", max_length=120)
    observacao: str = Field("", max_length=80)
    grafias: list[str] = Field(default_factory=list, max_length=20)


def _irmao_id_ou_404(id_: str) -> None:
    if not _IRMAO_ID.match(id_):
        raise HTTPException(status_code=404, detail="Nome não encontrado.")


@app.get("/api/nomes")
def list_nomes():
    try:
        lista = irmaos.listar()
    except irmaos.IrmaoError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"nomes": [e.to_dict() for e in lista]}


@app.post("/api/nomes", status_code=201)
def create_nome(body: IrmaoIn):
    try:
        return irmaos.criar(body.nome, body.observacao, body.grafias).to_dict()
    except irmaos.IrmaoError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.put("/api/nomes/{id_}")
def update_nome(id_: str, body: IrmaoIn):
    _irmao_id_ou_404(id_)
    try:
        nova = irmaos.atualizar(id_, body.nome, body.observacao, body.grafias)
    except irmaos.IrmaoError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if nova is None:
        raise HTTPException(status_code=404, detail="Nome não encontrado.")
    return nova.to_dict()


@app.delete("/api/nomes/{id_}")
def delete_nome(id_: str):
    _irmao_id_ou_404(id_)
    if not irmaos.apagar(id_):
        raise HTTPException(status_code=404, detail="Nome não encontrado.")
    return {"ok": True}


@app.post("/api/jobs/{job_id}/ata")
def make_ata(job_id: str):
    job = _get_done_job(job_id)
    if not config.ANTHROPIC_API_KEY:
        raise HTTPException(
            status_code=400,
            detail="Para gerar a ata, coloque ANTHROPIC_API_KEY no .env e "
                   "reinicie o Escriba.")
    if job.ata_status == jobs.ATA_GERANDO:
        raise HTTPException(status_code=409, detail="A ata já está sendo gerada.")

    def mutate(j: jobs.Job) -> None:
        j.ata_status = jobs.ATA_GERANDO
        j.ata_error = ""
    jobs.update(job_id, mutate)
    worker.enqueue_ata(job_id)
    return {"ok": True}


# --------------------------------------------------------------------------
# Perfil do cliente + assinatura
# --------------------------------------------------------------------------
class ProfileIn(BaseModel):
    nome: str = ""
    cargo: str = ""
    organizacao: str = ""
    email: str = ""


def _profile_payload() -> dict:
    return {**profile.load(), "tem_assinatura": profile.has_signature()}


@app.get("/api/profile")
def get_profile():
    return _profile_payload()


@app.put("/api/profile")
def put_profile(body: ProfileIn):
    if body.email.strip():
        try:
            mailer.parse_addresses(body.email, "E-mail")
        except mailer.MailError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    profile.save(body.model_dump())
    return _profile_payload()


@app.post("/api/profile/signature")
async def put_signature(file: UploadFile = File(...)):
    raw = await file.read(config.MAX_SIGNATURE_BYTES + 1)
    try:
        profile.set_signature(raw)
    except profile.SignatureError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _profile_payload()


@app.delete("/api/profile/signature")
def delete_signature():
    profile.clear_signature()
    return _profile_payload()


@app.get("/api/profile/signature.png")
def get_signature():
    data = profile.signature_bytes()
    if data is None:
        raise HTTPException(status_code=404, detail="Sem assinatura cadastrada.")
    return Response(data, media_type="image/png",
                    headers={"Cache-Control": "no-store"})


# --------------------------------------------------------------------------
# E-mail
# --------------------------------------------------------------------------
class EmailIn(BaseModel):
    para: str = ""
    cc: str = ""
    assunto: str = ""
    mensagem: str = ""
    anexos: list[str] = ["ata_pdf", "transcricao_pdf"]


_ATTACH_MIME = {
    "ata_pdf": "application/pdf", "transcricao_pdf": "application/pdf",
    "ata": "text/markdown", "txt": "text/plain", "srt": "application/x-subrip",
    "vtt": "text/vtt", "json": "application/json",
}


def _default_body(job: jobs.Job, prof: dict, anexos: list[str]) -> str:
    meet = review.default_title(job)
    data = job.meeting.get("date", "")
    docs = []
    if any(a.startswith("ata") for a in anexos):
        docs.append("a ata")
    if any(a.startswith("transcricao") or a in ("txt", "srt", "vtt", "json")
           for a in anexos):
        docs.append("a transcrição")
    quais = " e ".join(docs) or "os documentos"
    assin = "\n".join(x for x in (prof.get("nome"), prof.get("cargo"),
                                  prof.get("organizacao")) if x)
    return (f"Olá,\n\nSegue em anexo {quais} da reunião “{meet}”"
            f"{f' ({data})' if data else ''}.\n\n"
            + (assin or ""))


@app.get("/api/email/status")
def email_status():
    return {"configurado": mailer.configured(),
            "de": config.SMTP_FROM if mailer.configured() else ""}


@app.post("/api/jobs/{job_id}/email")
def send_email(job_id: str, body: EmailIn):
    job = _get_done_job(job_id)
    prof = profile.load()

    try:
        to = mailer.parse_addresses(body.para, "Para")
        cc = mailer.parse_addresses(body.cc, "Cc")
    except mailer.MailError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not to:
        raise HTTPException(status_code=422,
                            detail="Informe pelo menos um destinatário.")

    kinds = list(dict.fromkeys(body.anexos))
    if not kinds or any(k not in _EMAIL_KINDS for k in kinds):
        raise HTTPException(status_code=422,
                            detail="Escolha pelo menos um anexo válido.")

    attachments = []
    for kind in kinds:
        path, mime, name = _file_for(job, kind)
        attachments.append((name, path.read_bytes(), _ATTACH_MIME[kind]))

    subject = body.assunto.strip() or f"Ata e transcrição — {review.default_title(job)}"
    message = body.mensagem.strip() or _default_body(job, prof, kinds)

    try:
        mailer.send(to=to, cc=cc, subject=subject, body=message,
                    attachments=attachments,
                    sender_name=prof.get("nome") or prof.get("organizacao") or "",
                    reply_to=prof.get("email", ""))
    except mailer.MailError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    entry = {"quando": time.time(), "para": to, "cc": cc, "anexos": kinds,
             "assunto": subject}

    def mutate(j: jobs.Job) -> None:
        j.email_log = (j.email_log + [entry])[-50:]
    jobs.update(job_id, mutate)
    return {"ok": True, "enviado": entry}


# UI (por último, para não engolir as rotas /api)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
