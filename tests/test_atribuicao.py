"""Integração do motor de estrutura no Escriba (app.atribuicao + review).

Garante que a cola funciona: o resultado do motor vira segmentos rotulados por
pessoa, o nomes_auto casa com os rótulos de exibição, e review.segments_for
prefere a atribuição quando ela existe. O acerto da NOMEAÇÃO em si é coberto por
tests/estrutura (gabaritos do Eduardo).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import atribuicao, config, jobs, review  # noqa: E402
from app.engines.base import CORO_LABEL, Segment  # noqa: E402
from app.estrutura import analisar  # noqa: E402

DADOS = ROOT / "tests/estrutura/dados"
sys.path.insert(0, str(ROOT / "tests/estrutura"))
from ler_srt_motor import ler_srt  # noqa: E402

LISTA = json.loads((DADOS / "lista_irmaos.json").read_text(encoding="utf-8"))


@pytest.fixture
def dados(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    (tmp_path / "jobs").mkdir()
    (tmp_path / "nomes").mkdir()
    (tmp_path / "nomes" / "nomes.json").write_text(
        json.dumps(LISTA, ensure_ascii=False), encoding="utf-8")
    return tmp_path


def _atrib():
    segs = ler_srt(DADOS / "10-10-2026_transcricao.srt")
    papeis = {"presidente": "Rinaldo Santos", "orador": "Ademir Baldino",
              "dirigente": "José Eduardo Silveira", "leitor": "Wellington Passos"}
    return analisar(segs, LISTA, papeis=papeis).como_dict()


def test_segmentos_e_nomes_auto_casam():
    atrib = _atrib()
    segs = atribuicao.segmentos(atrib)
    auto = atribuicao.nomes_auto(atrib, segs)
    assert len(segs) == len(atrib["falas"])
    pessoas = {s.speaker for s in segs if s.speaker != CORO_LABEL}
    # um rótulo "Falante N" por pessoa, todos com nome e fonte
    assert len(auto) == len(pessoas)
    assert all(k.startswith("Falante ") for k in auto)
    assert all(v["nome"] and v["fonte"] for v in auto.values())
    # os papéis principais foram achados
    fontes = {v["fonte"] for v in auto.values()}
    assert {"presidente", "orador", "dirigente", "leitor"} <= fontes


def test_segments_for_prefere_atribuicao(dados):
    atrib = _atrib()
    job = jobs.new_job(filename="10-10.mp3", engine="assemblyai", language="pt",
                       min_speakers=None, max_speakers=None, want_summary=False,
                       participants=[])
    job.nomes_auto = atribuicao.nomes_auto(atrib)
    job.save()
    job.path(atribuicao.ATRIB_FILE).write_text(
        json.dumps(atrib, ensure_ascii=False), encoding="utf-8")
    # segmentos.json cru diferente (um cluster só): a atribuição deve vencer
    review.save_raw(job, [Segment(0.0, 1.0, "A", "qualquer coisa")])

    job = jobs.load(job.id)
    segs = review.segments_for(job)
    assert len(segs) == len(atrib["falas"])          # veio da atribuição, não do cru
    names = review.names_for(job, segs)
    assert "José Eduardo Silveira" in names.values()
    assert "Ademir Baldino" in names.values()


def test_rodar_sem_lista_retorna_none(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)       # sem nomes/nomes.json
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    (tmp_path / "jobs").mkdir()

    class J:
        meeting: dict = {}
    assert atribuicao.rodar(J(), [Segment(0.0, 1.0, "A", "oi")]) is None


def test_trava_descarta_reuniao_sem_estrutura():
    """17-10-2026: nenhuma parte achada -> a reunião inteira saía "[?]". Agora o
    resultado é descartado e o job segue pelos perfis."""
    falas = [{"ini": 0.0, "fim": 60.0, "quem": "[?]", "parte": "sem estrutura"}]
    assert atribuicao._estrutura_falhou({"blocos": [], "falas": falas})
    assert "sem estrutura" in atribuicao._estrutura_falhou(
        {"blocos": [{"parte": "abertura"}], "falas": falas})


def test_trava_aceita_reuniao_com_estrutura():
    falas = [{"ini": 0.0, "fim": 600.0, "quem": "William Brandão", "parte": "abertura"},
             {"ini": 600.0, "fim": 660.0, "quem": "[?]", "parte": "sem estrutura"},
             {"ini": 660.0, "fim": 900.0, "quem": "Em coro", "parte": "sem estrutura"}]
    assert atribuicao._estrutura_falhou({"blocos": [{"parte": "abertura"}], "falas": falas}) == ""


def test_rodar_17_10_acha_a_estrutura(dados):
    """Reunião real de 17-10 (segmentos crus do motor local + pyannote) com o cadastro."""
    segs = [Segment(s["inicio"], s["fim"], s["voz"], s["texto"])
            for s in json.loads((DADOS / "17-10-2026_cru_local.json").read_text(encoding="utf-8"))]

    class J:
        meeting = {"tipo_reuniao": "fim_de_semana",
                   "papeis": {"presidente": "William Brandão", "orador": "Henrique do Vale",
                              "dirigente": "Marcelo Almeida", "leitor": "Fernando Santos"}}
    atrib = atribuicao.rodar(J(), segs)
    assert atrib is not None
    partes = {(b["parte"], b["quem"]) for b in atrib["blocos"]}
    assert ("discurso", "Henrique do Vale") in partes
    assert ("estudo de A Sentinela", "Marcelo Almeida") in partes


PARTICIPANTES_17_10 = [
    "Jose Eduardo Silveira", "Elza Almeida", "William Brandão", "Alessandra Almeida",
    "Rinaldo Santos", "Fabio Vizignani", "Ines Vizignani", "Antonio Carlos", "Lucas Silva",
    "Rafael Silva", "Talita Silva", "Matheus Santos", "Alex Mota", "Luiz", "Ninfa",
    "Deana Santos", "Flavia Santos", "Henrique do Vale", "Otavio", "Milena", "Alicia", "Isaac",
    "Arlete Hoenne", "Arlete Oliveira", "Joelma Pinheiro", "Mercedes", "Vivian Manara",
    "Carmen", "Rita", "Fatima"]


def test_participantes_informados_entram_na_atribuicao(dados):
    """Bug de 17-10-2026: os nomes digitados em 'participaram da reunião' não chegavam
    ao motor de estrutura (só viravam vocabulário do Whisper)."""
    segs = [Segment(s["inicio"], s["fim"], s["voz"], s["texto"])
            for s in json.loads((DADOS / "17-10-2026_cru_local.json").read_text(encoding="utf-8"))]

    class J:
        participants = PARTICIPANTES_17_10
        meeting = {"tipo_reuniao": "fim_de_semana",
                   "papeis": {"presidente": "William Brandão", "orador": "Henrique do Vale",
                              "dirigente": "Marcelo Almeida", "leitor": "Fernando Santos"}}
    quem = {f["quem"] for f in atribuicao.rodar(J(), segs)["falas"]}
    assert "José Eduardo Silveira" in quem        # "irmão Zé Eduardo" (não Eduardo Moretti)
    assert "Eduardo Moretti" not in quem          # não estava na reunião
    assert {"Rafael Silva", "Lucas Silva", "Otavio"} <= quem     # "Rafa?", "Lucas?", "Otávio?"
    # duas Arletes na reunião: continua dúvida, mas só entre as duas
    arletes = [q for q in quem if "Arlete" in q]
    assert arletes and all(q.startswith("[?]") and "Hoenne" in q and "Oliveira" in q for q in arletes)
