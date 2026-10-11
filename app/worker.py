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

from . import config, exports, irmaos, jobs, media, nomes, review, vozes
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
    participantes = irmaos.corrige_participantes(job.participants, irmaos.listar_seguro(log))
    segments = engine.transcribe(
        audio,
        language=job.language or None,
        min_speakers=job.min_speakers,
        max_speakers=job.max_speakers,
        progress=job.set_phase,
        vocabulary=participantes or None,
    )
    if not segments:
        raise RuntimeError("O engine não retornou nenhum segmento de fala.")

    segments = merge_segments(segments)

    # Assinaturas de voz desta reunião (motor local): base do aprendizado
    voices = getattr(engine, "voices", None) or {}
    if voices:
        job.path("vozes.json").write_text(json.dumps(voices), encoding="utf-8")
    job.vozes_info = getattr(engine, "scan", None) or {}
    job.falantes_estudo = getattr(engine, "estudo", None) or []
    job.estudo_info = getattr(engine, "estudo_info", None) or {}
    job.save()

    # 3) Dar nome às vozes (sugestões; o usuário confirma na revisão)
    job.set_phase("Fase 3/3 · Identificando quem é quem" if voices
                  else "Identificando quem é quem")
    _auto_names(job, segments, voices)
    job = jobs.load(job.id)            # _auto_names gravou as sugestões (nomes_auto)

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
    segments = [s for s in segments if s.speaker != CORO_LABEL]  # coro não é pessoa
    display = exports.speaker_map(segments)                  # "A" -> "Falante 1"
    by_voice = vozes.match({lab: v["vetor"] for lab, v in voices.items()})
    strong = {lab for lab, (_, sim) in by_voice.items() if sim >= vozes.MATCH_STRONG}
    entradas = irmaos.listar_seguro(log)                     # lista de irmãos (só nomes)
    participantes = irmaos.corrige_participantes(job.participants, entradas)
    candidates = [*participantes, *(p["nome"] for p in vozes.list_people())]
    by_talk = nomes.infer(segments, candidates, exclude=strong, irmaos=entradas)

    auto: dict = {}
    for raw, label in display.items():
        if raw in by_voice and (raw in strong or raw not in by_talk):
            name, sim = by_voice[raw]
            auto[label] = {"nome": name, "fonte": "voz",
                           "detalhe": f"voz reconhecida ({round(sim * 100)}% parecida)"}
        elif raw in by_talk:
            name, _, why = by_talk[raw]
            auto[label] = {"nome": name, "fonte": "conversa", "detalhe": why}

    # nomes dos turnos do estudo (chamada/agradecimento do dirigente): sugestão
    # acima de conversa e eliminação, abaixo de voz reconhecida
    for pessoa in job.falantes_estudo or []:
        if not pessoa.get("nomeada") or pessoa["rotulo"] not in display:
            continue
        label = display[pessoa["rotulo"]]
        if auto.get(label, {}).get("fonte") == "voz":
            continue
        detalhe = pessoa["origem"] or "estrutura do estudo"
        if not _conhecido(pessoa["nome"], entradas, participantes):
            # nome fora da lista e dos participantes: sugestão de confiança menor
            detalhe += " · fora da lista e dos participantes: confirme"
        auto[label] = {"nome": pessoa["nome"], "fonte": "turno", "detalhe": detalhe}

    # homônimos na lista: o nome é só sugestão, a observação diz para confirmar pela voz
    for label, info in auto.items():
        aviso = irmaos.corrige(info["nome"], entradas)
        if aviso.ambiguo:
            info["detalhe"] = f"{info['detalhe']} · {aviso.observacao}"

    # mesmo nome não pode ir para duas vozes
    seen: set = set()
    for label in list(auto):
        if auto[label]["nome"] in seen:
            del auto[label]
        else:
            seen.add(auto[label]["nome"])

    # 3º por eliminação: 1 falante sem nome + 1 participante livre (sugestão)
    nomeados = {**job.speaker_names, **{lab: i["nome"] for lab, i in auto.items()}}
    for label, nome in nomes.por_eliminacao(display.values(), participantes,
                                            nomeados).items():
        auto[label] = {"nome": nome, "fonte": "eliminacao",
                       "detalhe": nomes.ELIMINACAO_DETALHE}

    def mutate(j: jobs.Job) -> None:
        j.nomes_auto = auto
    jobs.update(job.id, mutate)
