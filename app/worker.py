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
from pathlib import Path

import json
import logging

from . import (atribuicao, config, fim_de_semana, irmaos, jobs, media, nomes,
               perfis, review, vozes)
from .engines import get_engine, merge_segments
from .engines.base import CORO_LABEL

log = logging.getLogger("escriba.worker")
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
            # vai para o escriba.log (no programa empacotado não há console)
            log.exception("Falha no job %s (%s)", job_id, kind)
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
        if config.ANTHROPIC_API_KEY:
            from . import summary
            ata = summary.generate_ata(review.transcript_txt(job), job.participants)
            origem = jobs.ORIGEM_IA
        else:
            # sem chave: a ata sai da estrutura da reunião, offline
            from . import ata_local
            ata = ata_local.gera(job)
            origem = jobs.ORIGEM_LOCAL
        job.path("ata.md").write_text(ata, encoding="utf-8")
        for stale in ("ata.pdf", "ata.pdf.key"):
            job.path(stale).unlink(missing_ok=True)

        def ok(j: jobs.Job) -> None:
            j.ata_status = jobs.ATA_PRONTA
            j.ata_error = ""
            j.ata_origem = origem
            j.outputs = review.outputs_for(j)
        jobs.update(job_id, ok)
    except Exception as exc:  # noqa: BLE001
        log.exception("Falha ao gerar a ata do job %s", job_id)

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
    if engine.name in ("local", "whisperx"):
        job.set_phase("Extraindo áudio (WAV 16 kHz)")
        audio = media.to_wav16k(original, job.path("audio.wav"))
    else:
        job.set_phase("Extraindo áudio (MP3 leve para upload)")
        audio = media.to_mp3_asr(original, job.path("audio.mp3"))

    # 2) Transcrever + diarizar (os nomes informados viram vocabulário)
    # participante digitado com erro vale pelo nome da lista ("José Edaurdo" -> "José Eduardo")
    _entradas = irmaos.listar_seguro(log)
    participantes = irmaos.corrige_participantes(job.participants, _entradas)
    # A lista de irmãos é OBRIGATÓRIA (regra do Eduardo): sem participantes
    # digitados, o vocabulário vem da lista toda — o programa nunca ignora a lista.
    vocabulario = participantes or [e.nome for e in _entradas]
    # Os papéis do cadastro (presidente/orador/dirigente/leitor) são os nomes de
    # maior valor para o reconhecimento. O hotwords do Whisper tem orçamento
    # apertado (~224 tokens) e corta o fim da lista; então esses nomes LIDERAM o
    # vocabulário, para nunca caírem fora do corte. Corrige grafia pela lista.
    papeis_nomes = irmaos.corrige_participantes(
        [n for n in ((job.meeting or {}).get("papeis") or {}).values() if n], _entradas)
    if papeis_nomes:
        vocabulario = list(dict.fromkeys([*papeis_nomes, *vocabulario]))
    # Dica de quantidade de falantes para a AssemblyAI. Sem ela, a API funde
    # todo mundo (reunião de 34 virou 6 clusters). Se o usuário não digitou o
    # nº, usa o tamanho da lista de participantes da reunião; sem ela, o
    # fallback de config. Não mexe no motor local (usa o job.max_speakers cru).
    max_sp = job.max_speakers
    if engine.name == "assemblyai" and not max_sp:
        max_sp = len(participantes) or config.ASSEMBLYAI_FALLBACK_SPEAKERS
    segments = engine.transcribe(
        audio,
        language=job.language or None,
        min_speakers=job.min_speakers,
        max_speakers=max_sp,
        progress=job.set_phase,
        vocabulary=vocabulario or None,
    )
    if not segments:
        raise RuntimeError("O engine não retornou nenhum segmento de fala.")

    raw_segments = list(segments)     # finos, ANTES do merge: o motor precisa da fronteira entre pessoas
    segments = merge_segments(segments)

    # Assinaturas de voz desta reunião (motor local): base do aprendizado
    voices = getattr(engine, "voices", None) or {}
    if voices:
        job.path("vozes.json").write_text(json.dumps(voices), encoding="utf-8")
    job.vozes_info = getattr(engine, "scan", None) or {}
    job.falantes_estudo = getattr(engine, "estudo", None) or []
    job.estudo_info = getattr(engine, "estudo_info", None) or {}
    job.save()

    # 3) Quem fala cada trecho.
    # Reunião RELIGIOSA: o MOTOR DE ESTRUTURA (app.estrutura) decide trecho a
    # trecho e grava atribuicao.json; segmentos.json segue cru. Os demais tipos
    # (negócios, conversa, monólogo) seguem pelos perfis (nomeação por cluster).
    job.set_phase("Fase 3/3 · Identificando quem é quem" if voices
                  else "Identificando quem é quem")
    perfil = perfis.detecta(segments, job)
    log.info("perfil de reunião detectado: %s", perfil)
    usou_motor = False
    if perfil == "religiosa":
        try:
            atrib = atribuicao.rodar(job, raw_segments)
        except Exception:        # noqa: BLE001 — o motor nunca derruba o job
            log.exception("motor de estrutura falhou; caindo para os perfis")
            atrib = None
        if atrib is not None:
            job.path(atribuicao.ATRIB_FILE).write_text(
                json.dumps(atrib, ensure_ascii=False), encoding="utf-8")
            auto = atribuicao.nomes_auto(atrib)

            def _set(j: jobs.Job) -> None:
                j.nomes_auto = auto
            jobs.update(job.id, _set)
            usou_motor = True

    if not usou_motor:
        # Backchannel do dirigente de volta nele: em reunião com estudo, o
        # dirigente agradece/chama entre os comentários e o diarizador às vezes
        # separa esses turnos num cluster à parte. Nos engines de nuvem, funde
        # pela estrutura. (sem atribuicao.json: as saídas usam os clusters)
        job.path(atribuicao.ATRIB_FILE).unlink(missing_ok=True)
        if engine.name != "local":
            segments = merge_segments(fim_de_semana.funde_dirigente(segments))
        _auto_names(job, segments, voices)

    job = jobs.load(job.id)            # nomes_auto gravado

    # 4) Guardar o cru e gerar os arquivos (nomes/correções entram por cima)
    job.set_phase("Gerando arquivos de saída")
    review.save_raw(job, segments)
    meeting = {"title": "", "date": "", "place": "", **job.meeting}
    if not meeting["title"]:
        meeting["title"] = Path(job.filename).stem
    job.meeting = meeting
    job.save()

    review.render_outputs(job.id)
    job = jobs.load(job.id)            # render_outputs gravou falantes/timeline

    # 5) Ata automática (opcional)
    if job.want_summary:
        job.set_phase("Gerando ata da reunião (IA)" if config.ANTHROPIC_API_KEY
                      else "Gerando ata da reunião")
        run_ata(job.id)

    def finish(j: jobs.Job) -> None:
        j.status = jobs.STATUS_DONE
        j.finished_at = time.time()
        j.phase = "Concluído"
    jobs.update(job.id, finish)


def _conhecido(nome: str, entradas, participantes) -> bool:
    """Nome que consta na lista de irmãos (inclusive por grafia) ou nos participantes."""
    if irmaos.corrige_participante(nome, entradas) is not None:
        return True
    return nomes._participante_de(nome, list(participantes)) is not None


def _auto_names(job: jobs.Job, segments: list, voices: dict) -> None:
    """Sugere, para cada "Falante N", o nome que der para deduzir:
    1º a memória de vozes (quem já foi confirmado em outras reuniões),
    2º quem é chamado pelo nome / se apresenta na conversa,
    3º por eliminação.

    As sugestões vão SÓ para `job.nomes_auto`. Não preenchem `speaker_names`
    nem `participants`: esses campos são o que o usuário confirma na revisão,
    e são a única fonte para a memória de vozes aprender. Um nome deduzido
    que vira "fato" aqui seria aprendido na próxima salva e se reforçaria."""
    # A estratégia de nomeação vem do PERFIL da reunião (religiosa,
    # negócios, conversa, monólogo): um módulo por tipo, escolhido por
    # detecção — assim um tipo novo é um módulo, não uma emenda no núcleo.
    auto, perfil = perfis.nomeia(job, segments, voices)
    log.info("perfil de reunião detectado: %s", perfil)


    def mutate(j: jobs.Job) -> None:
        j.nomes_auto = auto
    jobs.update(job.id, mutate)
