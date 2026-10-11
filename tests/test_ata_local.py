"""Ata local (sem API): conteúdo, origem e geração pelo worker sem ANTHROPIC_API_KEY.

Rodar: python -m pytest tests/test_ata_local.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import ata_local, config, irmaos, jobs, review, worker  # noqa: E402
from app.engines.base import Segment  # noqa: E402


@pytest.fixture
def dados(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")   # sem chave: ata local
    (tmp_path / "jobs").mkdir()
    irmaos.criar("José Eduardo", "dirigente")     # para corrigir "José Edaurdo"
    return tmp_path


def _job() -> jobs.Job:
    job = jobs.new_job(filename="10-10-2026.mp3", engine="local", language="pt",
                       min_speakers=None, max_speakers=None, want_summary=True,
                       participants=["Eduardo", "José Edaurdo", "Ademir"])
    job.status = jobs.STATUS_DONE
    job.duration = 6339.0
    review.save_raw(job, [
        Segment(10.0, 30.0, "A", "Bem-vindos, vamos começar. Vamos cantar o Cântica de número 56. "
                                 "Faça da verdade a sua vida. Tem como base provérbios 3."),
        Segment(2244.0, 2269.0, "B", "Queremos louvar com o cântico de número 55, que tem por tema, "
                                     "Nada Temo, Meus Amados. Os"),
        Segment(5992.0, 6035.0, "B", "Vamos cantar o cântico número 150. Busquem a Deus para "
                                     "obter livramento."),
    ])
    job.estudo_info = {"parte1": {
        "presidente": "Rinaldo", "orador": "Ademir", "orador_cru": "Módemir",
        "orador_confianca": "media", "tema": TEMA, "orador_inicio": 427.4,
        "orador_fim": 2190.0}}
    job.falantes_estudo = [
        {"rotulo": "P1", "nome": "José Eduardo", "papel": "dirigente", "nomeada": True,
         "comentarios": 85},
        {"rotulo": "P2", "nome": "Wellington", "papel": "leitor", "nomeada": True,
         "comentarios": 43},
        {"rotulo": "P3", "nome": "Rinaldo", "papel": "presidente + comentarista",
         "nomeada": True, "comentarios": 2},
        {"rotulo": "P4", "nome": "Carmen", "papel": "comentarista", "nomeada": True,
         "comentarios": 1},
        {"rotulo": "P5", "nome": "Comentarista 1", "papel": "comentarista",
         "nomeada": False, "comentarios": 3},
    ]
    job.save()
    return job


TEMA = "como sempre tomar as melhores decisões"


def test_ata_local_traz_as_secoes_pedidas(dados):
    texto = ata_local.gera(_job())
    assert "# Ata da reunião" in texto
    assert "10/10/2026" in texto and "01:45:39" in texto           # data e duração
    assert "**Presidente:** Rinaldo" in texto
    assert "**Dirigente de A Sentinela:** José Eduardo" in texto
    assert "**Leitor de A Sentinela:** Wellington" in texto
    assert "**Orador:** Ademir (nome pela grafia do anúncio" in texto
    assert f"**Tema:** {TEMA}" in texto
    assert "Trecho:** 00:07:07 a 00:36:30" in texto
    assert "Cântico de número 56. Faça da verdade a sua vida" in texto
    assert "Cântico de número 55. Nada Temo, Meus Amados" in texto
    assert "Cântico de número 150. Busquem a Deus para obter livramento" in texto
    assert "- Rinaldo: 2 comentários" in texto
    assert "- Carmen: 1 comentário" in texto
    assert "- Comentarista 1: 3 comentários" in texto
    assert "- Eduardo" in texto and "- José Eduardo" in texto      # participante corrigido


def test_ata_local_nao_resume_comentarios_e_avisa(dados):
    texto = ata_local.gera(_job())
    assert "Ata gerada localmente, sem IA" in texto
    assert "não resume o conteúdo dos comentários" in texto


def test_canticos_sem_repetir_numero():
    textos = ["Cântico de número 56. Faça da verdade a sua vida.",
              "o cântico 56, Faça da verdade a sua vida"]
    assert ata_local.canticos(textos) == ["Cântico de número 56. Faça da verdade a sua vida"]


def test_worker_gera_ata_local_sem_chave_e_marca_a_origem(dados):
    job = _job()
    worker.run_ata(job.id)
    job = jobs.load(job.id)
    assert job.ata_status == jobs.ATA_PRONTA
    assert job.ata_origem == jobs.ORIGEM_LOCAL
    assert job.ata_error == ""
    assert "ata" in job.outputs and "ata_pdf" in job.outputs
    assert "Ata da reunião" in job.path("ata.md").read_text(encoding="utf-8")


def test_ata_com_chave_continua_pela_api(dados, monkeypatch):
    chamadas = []

    def falsa(texto, participantes=None):
        chamadas.append(texto)
        return "# Ata da reunião\n\nvia API"

    from app import summary
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "chave-de-teste")
    monkeypatch.setattr(summary, "generate_ata", falsa)
    job = _job()
    worker.run_ata(job.id)
    job = jobs.load(job.id)
    assert chamadas and job.ata_origem == jobs.ORIGEM_IA
    assert "via API" in job.path("ata.md").read_text(encoding="utf-8")


def test_canticos_com_as_formas_da_fala():
    """A fala real vem como "o cântico é de número 56" e "o cântico é o número 150"."""
    textos = ["Ainda o cântico é de número 56, faça da verdade a sua vida, tem como base",
              "Vamos cantar, o cântico é o número 150. Busquem a Deus para obter livramento."]
    assert ata_local.canticos(textos) == [
        "Cântico de número 56. Faça da verdade a sua vida",
        "Cântico de número 150. Busquem a Deus para obter livramento"]
