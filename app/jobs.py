"""Jobs de transcrição — um diretório por job em data/jobs/{id}/.

Estrutura de cada job:
    original.<ext>   arquivo enviado
    audio.mp3|wav    áudio normalizado
    segmentos.json   transcrição CRUA (nunca é alterada; as correções de nomes
                     e grafia são aplicadas por cima, na hora de gerar os arquivos)
    vozes.json       assinatura de voz de cada falante (motor local), usada
                     para aprender as vozes quando o usuário confirma os nomes
    meta.json        estado + dados de revisão (participantes, nomes, correções)
    transcricao.txt / .srt / .vtt / .json
    ata.md           (opcional)
    ata.pdf / transcricao.pdf   (gerados sob demanda, com cache)
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from . import config

# RLock: update() chama save() com o lock já adquirido.
_lock = threading.RLock()

STATUS_QUEUED = "na_fila"
STATUS_RUNNING = "processando"
STATUS_DONE = "concluido"
STATUS_ERROR = "erro"

ATA_GERANDO = "gerando"
ATA_PRONTA = "pronta"
ATA_ERRO = "erro"


@dataclass
class Job:
    id: str
    filename: str
    engine: str
    language: str                      # "pt", "en"... ou "" = auto
    min_speakers: Optional[int]
    max_speakers: Optional[int]
    want_summary: bool
    status: str = STATUS_QUEUED
    phase: str = "Aguardando na fila"
    error: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    duration: Optional[float] = None   # duração do áudio (s)
    speakers: list[str] = field(default_factory=list)   # rótulos exibidos
    timeline: list = field(default_factory=list)        # [[ini, fim, idx], ...]
    outputs: list[str] = field(default_factory=list)    # ["txt","srt",...]

    # ---- revisão (nomes, grafia, dados do documento) ------------------------
    participants: list[str] = field(default_factory=list)   # grafia correta
    speaker_names: dict = field(default_factory=dict)       # "Falante 1" -> nome
    corrections: list = field(default_factory=list)         # [{"de","para"}]
    meeting: dict = field(default_factory=dict)             # title/date/place
    ata_status: str = ""                                    # gerando|pronta|erro
    ata_error: str = ""                                     # motivo, se ata_status=erro
    email_log: list = field(default_factory=list)           # envios feitos

    # ---- reconhecimento de vozes e nomes (motor local) ----------------------
    vozes_info: dict = field(default_factory=dict)          # resumo da fase 1
    falantes_estudo: list = field(default_factory=list)     # pessoas do estudo (turnos)
    nomes_auto: dict = field(default_factory=dict)          # "Falante 1" -> {nome, fonte, detalhe}

    # ---- caminhos ----------------------------------------------------------
    @property
    def dir(self) -> Path:
        return config.JOBS_DIR / self.id

    def path(self, name: str) -> Path:
        return self.dir / name

    # ---- persistência ------------------------------------------------------
    def save(self) -> None:
        with _lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp = self.path("meta.json.tmp")
            tmp.write_text(
                json.dumps(asdict(self), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self.path("meta.json"))

    def set_phase(self, phase: str) -> None:
        self.phase = phase
        self.save()


def new_job(**kwargs) -> Job:
    job = Job(id=uuid.uuid4().hex[:12], **kwargs)
    job.save()
    return job


def load(job_id: str) -> Optional[Job]:
    meta = config.JOBS_DIR / job_id / "meta.json"
    if not meta.exists():
        return None
    data = json.loads(meta.read_text(encoding="utf-8"))
    return Job(**data)


def update(job_id: str, mutate: Callable[[Job], None]) -> Optional[Job]:
    """Carrega o job, aplica `mutate` e salva — tudo sob o lock.

    Usado pelas rotas de revisão e pela geração da ata, que rodam em outras
    threads: assim uma não sobrescreve o que a outra acabou de gravar.
    """
    with _lock:
        job = load(job_id)
        if job is None:
            return None
        mutate(job)
        job.save()
        return job


def list_jobs() -> list[Job]:
    jobs = []
    for meta in config.JOBS_DIR.glob("*/meta.json"):
        try:
            jobs.append(Job(**json.loads(meta.read_text(encoding="utf-8"))))
        except Exception:
            continue
    return sorted(jobs, key=lambda j: j.created_at, reverse=True)


def recover_interrupted() -> list[str]:
    """No startup: reenfileira o que ficou pendente de uma sessão anterior."""
    requeued = []
    for job in list_jobs():
        if job.status == STATUS_RUNNING:
            job.status = STATUS_QUEUED
            job.set_phase("Reenfileirado após reinício do servidor")
            requeued.append(job.id)
        elif job.status == STATUS_QUEUED:
            requeued.append(job.id)
        elif job.ata_status == ATA_GERANDO:
            # a ata ficou pela metade: marca como erro para o usuário refazer
            job.ata_status = ATA_ERRO
            job.save()
    return requeued
