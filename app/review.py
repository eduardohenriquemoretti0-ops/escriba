"""Revisão da transcrição: nomes dos participantes e grafia correta.

A transcrição crua (segmentos.json) NUNCA é alterada. Tudo que o cliente
informa na tela de revisão — quem é cada "Falante N", correções de grafia —
fica no meta.json do job e é aplicado POR CIMA toda vez que os arquivos são
gerados. Assim dá para refazer a revisão quantas vezes quiser, sem perder o
original nem aplicar uma correção duas vezes.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict
from difflib import SequenceMatcher

from . import exports, jobs
from .engines.base import Segment

SEG_FILE = "segmentos.json"

MAX_PARTICIPANTS = 80
MAX_CORRECTIONS = 300
MAX_NAME_LEN = 80
MAX_MEETING_FIELD = 160

_CTRL = re.compile(r"[\x00-\x1f\x7f]")
_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’-]*")


# --------------------------------------------------------------------------
# Limpeza das entradas do usuário
# --------------------------------------------------------------------------
def clean_name(value: object, limit: int = MAX_NAME_LEN) -> str:
    text = _CTRL.sub(" ", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()[:limit]


def clean_participants(values: list) -> list[str]:
    seen, out = set(), []
    for v in values or []:
        name = clean_name(v)
        key = name.casefold()
        if name and key not in seen:
            seen.add(key)
            out.append(name)
    return out[:MAX_PARTICIPANTS]


def clean_corrections(values: list) -> list[dict]:
    out = []
    for c in values or []:
        de = clean_name(c.get("de") if isinstance(c, dict) else "")
        para = clean_name(c.get("para") if isinstance(c, dict) else "")
        if de and para and de != para:
            out.append({"de": de, "para": para})
    return out[:MAX_CORRECTIONS]


def clean_meeting(values: dict) -> dict:
    values = values if isinstance(values, dict) else {}
    return {k: clean_name(values.get(k), MAX_MEETING_FIELD)
            for k in ("title", "date", "place")}


# --------------------------------------------------------------------------
# Segmentos
# --------------------------------------------------------------------------
def save_raw(job: jobs.Job, segments: list[Segment]) -> None:
    job.path(SEG_FILE).write_text(
        json.dumps([asdict(s) for s in segments], ensure_ascii=False),
        encoding="utf-8")


def load_raw(job: jobs.Job) -> list[Segment]:
    path = job.path(SEG_FILE)
    if not path.exists():
        return []
    return [Segment(**d) for d in json.loads(path.read_text(encoding="utf-8"))]


def apply_corrections(text: str, corrections: list[dict]) -> str:
    """Troca palavras inteiras, respeitando maiúsculas/minúsculas do original."""
    for c in corrections or []:
        de, para = c.get("de", ""), c.get("para", "")
        if not de or not para:
            continue
        pattern = r"(?<![\wÀ-ÿ])" + re.escape(de) + r"(?![\wÀ-ÿ])"
        text = re.sub(pattern, lambda _m, p=para: p, text)
    return text


def segments_for(job: jobs.Job) -> list[Segment]:
    """Segmentos já com as correções de grafia aplicadas."""
    return [
        Segment(s.start, s.end, s.speaker,
                apply_corrections(s.text, job.corrections))
        for s in load_raw(job)
    ]


def names_for(job: jobs.Job, segments: list[Segment]) -> dict[str, str]:
    """Nome de cada falante nas saídas (TXT/SRT/VTT/JSON/PDF/ata).

    Usa só `speaker_names`, que é o que o usuário confirmou na revisão. A
    sugestão (`nomes_auto`) não entra aqui: sem confirmação, o falante sai
    como "Falante N". Assim o que aparece no arquivo é sempre dado confirmado,
    e apagar um nome na revisão de fato tira o nome dos arquivos."""
    return exports.apply_names(exports.speaker_map(segments), job.speaker_names)


def default_title(job: jobs.Job) -> str:
    return job.meeting.get("title") or job.filename.rsplit(".", 1)[0]


# --------------------------------------------------------------------------
# Geração dos arquivos
# --------------------------------------------------------------------------
def transcript_txt(job: jobs.Job) -> str:
    segs = segments_for(job)
    return exports.to_txt(segs, names_for(job, segs),
                          f"Transcrição — {job.filename}", job.duration)


def outputs_for(job: jobs.Job) -> list[str]:
    outs = ["txt", "srt", "vtt", "json", "transcricao_pdf"]
    if job.path("ata.md").exists():
        outs += ["ata", "ata_pdf"]
    return outs


def render_outputs(job_id: str) -> None:
    """(Re)gera TXT/SRT/VTT/JSON, a lista de falantes e a timeline."""
    job = jobs.load(job_id)
    if job is None:
        return
    segs = segments_for(job)
    names = names_for(job, segs)

    job.path("transcricao.txt").write_text(
        exports.to_txt(segs, names, f"Transcrição — {job.filename}",
                       job.duration), encoding="utf-8")
    job.path("transcricao.srt").write_text(
        exports.to_srt(segs, names), encoding="utf-8")
    job.path("transcricao.vtt").write_text(
        exports.to_vtt(segs, names), encoding="utf-8")
    job.path("transcricao.json").write_text(
        exports.to_json(segs, names, {
            "arquivo": job.filename,
            "engine": job.engine,
            "duracao_segundos": job.duration,
            "participantes": job.participants,
        }), encoding="utf-8")

    def mutate(j: jobs.Job) -> None:
        j.speakers = list(dict.fromkeys(names.values()))
        j.timeline = exports.build_timeline(segs, names)
        j.outputs = outputs_for(j)

    jobs.update(job_id, mutate)


# --------------------------------------------------------------------------
# Dados para a tela de revisão
# --------------------------------------------------------------------------
def speakers_overview(job: jobs.Job) -> list[dict]:
    """Um item por 'Falante N': quantas falas, palavras e trechos de amostra."""
    raw_segs = load_raw(job)
    default = exports.speaker_map(raw_segs)
    rows: dict[str, dict] = {}
    for seg in raw_segs:
        label = default[seg.speaker]
        row = rows.setdefault(label, {
            "label": label,
            "nome": job.speaker_names.get(label, ""),
            "auto": job.nomes_auto.get(label),
            "falas": 0, "palavras": 0, "primeira_fala": seg.start,
            "amostras": [],
        })
        row["falas"] += 1
        row["palavras"] += len(seg.text.split())
        if len(row["amostras"]) < 3 and len(seg.text.split()) >= 6:
            txt = apply_corrections(seg.text, job.corrections)
            row["amostras"].append(txt if len(txt) <= 140 else txt[:137] + "…")
    return list(rows.values())


def _fold(text: str) -> str:
    """Minúsculas e sem acentos, só para comparar."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).casefold()


def suggest_corrections(job: jobs.Job, limit: int = 40,
                        threshold: float = 0.8) -> list[dict]:
    """Palavras da transcrição parecidas com um nome informado, mas escritas
    de outro jeito (ex.: 'Rosilaine' x 'Rosilene'). Só sugere; o cliente decide."""
    if not job.participants:
        return []

    targets: dict[str, str] = {}          # forma sem acento -> grafia certa
    for full in job.participants:
        for part in full.split():
            part = part.strip(".,;:()")
            if len(part) >= 4:
                targets.setdefault(_fold(part), part)
    if not targets:
        return []

    already = {c["de"] for c in job.corrections}
    counts: Counter = Counter()
    for seg in load_raw(job):
        counts.update(_WORD.findall(seg.text))

    found: list[dict] = []
    for word, n in counts.items():
        if len(word) < 4 or word in already:
            continue
        w = _fold(word)
        best, best_ratio = None, 0.0
        for folded, proper in targets.items():
            if folded[0] != w[0] or abs(len(folded) - len(w)) > 2:
                continue
            if w == folded:                       # só difere no acento/caixa
                if word != proper:
                    best, best_ratio = proper, 1.0
                break
            sm = SequenceMatcher(None, w, folded)
            if sm.quick_ratio() < threshold:
                continue
            ratio = sm.ratio()
            if ratio >= threshold and ratio > best_ratio:
                best, best_ratio = proper, ratio
        if best and word != best:
            found.append({"de": word, "para": best, "ocorrencias": n,
                          "similaridade": round(best_ratio, 2)})

    found.sort(key=lambda d: (-d["ocorrencias"], -d["similaridade"], d["de"]))
    return found[:limit]
