"""Fila de processamento.

Uma thread única consome os jobs em ordem — transcrição de horas de áudio não
combina com request HTTP síncrono, e processar um job por vez evita disputa de
GPU/CPU. A UI acompanha por polling no meta.json.

A fila carrega dois tipos de tarefa:
  "processo"  transcrição completa de um arquivo recém-enviado
  "ata"       (re)geração da ata de uma transcrição já pronta
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
from pathlib import Path

from . import config, jobs, media, review
from .engines import get_engine, merge_segments

_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
_started = False


def enqueue(job_id: str) -> None:
    _queue.put(("processo", job_id))


def enqueue_ata(job_id: str) -> None:
    _queue.put(("ata", job_id))


def start() -> None:
    global _started
    if _started:
        return
    _started = True
    threading.Thread(target=_loop, daemon=True, name="escriba-worker").start()


def _loop() -> None:
    while True:
        kind, job_id = _queue.get()
        try:
            if kind == "ata":
                run_ata(job_id)
            else:
                job = jobs.load(job_id)
                if job is not None:
                    _process(job)
        except Exception as exc:  # noqa: BLE001 — worker nunca pode morrer
            traceback.print_exc()
            if kind == "processo":
                _mark_failed(job_id, str(exc))
        finally:
            _queue.task_done()


def _mark_failed(job_id: str, message: str) -> None:
    def mutate(j: jobs.Job) -> None:
        j.status = jobs.STATUS_ERROR
        j.error = message
        j.finished_at = time.time()
        j.phase = "Erro"
    jobs.update(job_id, mutate)


# --------------------------------------------------------------------------
# Ata
# --------------------------------------------------------------------------
def run_ata(job_id: str) -> None:
    """Gera (ou refaz) a ata a partir da transcrição já com nomes e grafia
    corrigidos. Roda na thread do worker; só mexe nos campos da ata."""
    job = jobs.load(job_id)
    if job is None:
        return

    def begin(j: jobs.Job) -> None:
        j.ata_status = jobs.ATA_GERANDO
        j.ata_error = ""
    jobs.update(job_id, begin)

    try:
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY ausente no .env.")
        from . import summary
        ata = summary.generate_ata(review.transcript_txt(job), job.participants)
        job.path("ata.md").write_text(ata, encoding="utf-8")
        for stale in ("ata.pdf", "ata.pdf.key"):
            job.path(stale).unlink(missing_ok=True)

        def ok(j: jobs.Job) -> None:
            j.ata_status = jobs.ATA_PRONTA
            j.ata_error = ""
            j.outputs = review.outputs_for(j)
        jobs.update(job_id, ok)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()

        def fail(j: jobs.Job) -> None:
            j.ata_status = jobs.ATA_ERRO
            j.ata_error = f"Ata não gerada: {exc}"
        jobs.update(job_id, fail)


# --------------------------------------------------------------------------
# Transcrição
# --------------------------------------------------------------------------
def _process(job: jobs.Job) -> None:
    job.status = jobs.STATUS_RUNNING
    job.started_at = time.time()
    job.error = ""
    job.set_phase("Lendo o arquivo")

    original = next(job.dir.glob("original.*"))
    job.duration = media.duration_seconds(original)
    job.save()

    # 1) Normalizar o áudio para o engine escolhido
    engine = get_engine(job.engine)
    if engine.name == "whisperx":
        job.set_phase("Extraindo áudio (WAV 16 kHz)")
        audio = media.to_wav16k(original, job.path("audio.wav"))
    else:
        job.set_phase("Extraindo áudio (MP3 leve para upload)")
        audio = media.to_mp3_asr(original, job.path("audio.mp3"))

    # 2) Transcrever + diarizar (os nomes informados viram vocabulário)
    segments = engine.transcribe(
        audio,
        language=job.language or None,
        min_speakers=job.min_speakers,
        max_speakers=job.max_speakers,
        progress=job.set_phase,
        vocabulary=job.participants or None,
    )
    if not segments:
        raise RuntimeError("O engine não retornou nenhum segmento de fala.")

    segments = merge_segments(segments)

    # 3) Guardar o cru e gerar os arquivos (nomes/correções entram por cima)
    job.set_phase("Gerando arquivos de saída")
    review.save_raw(job, segments)
    meeting = {"title": "", "date": "", "place": "", **job.meeting}
    if not meeting["title"]:
        meeting["title"] = Path(job.filename).stem
    job.meeting = meeting
    job.save()

    review.render_outputs(job.id)
    job = jobs.load(job.id)            # render_outputs gravou falantes/timeline

    # 4) Ata automática (opcional)
    if job.want_summary:
        if config.ANTHROPIC_API_KEY:
            job.set_phase("Gerando ata da reunião (IA)")
            run_ata(job.id)
        else:
            def no_key(j: jobs.Job) -> None:
                j.ata_status = jobs.ATA_ERRO
                j.ata_error = ("Ata não gerada: ANTHROPIC_API_KEY ausente no "
                               ".env (a transcrição foi concluída normalmente).")
            jobs.update(job.id, no_key)

    def finish(j: jobs.Job) -> None:
        j.status = jobs.STATUS_DONE
        j.finished_at = time.time()
        j.phase = "Concluído"
    jobs.update(job.id, finish)
