"""Nomes nas saídas: sugestão aparece, confirmado vence, apagar vale, participante corrigido.

Rodar: python -m pytest tests/test_nomes_exibicao.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import config, exports, irmaos, jobs, review  # noqa: E402
from app.engines.base import Segment  # noqa: E402


@pytest.fixture
def dados(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    (tmp_path / "jobs").mkdir()
    return tmp_path


def _job(**campos) -> jobs.Job:
    job = jobs.new_job(filename="10-10-2026.mp3", engine="local", language="pt",
                       min_speakers=None, max_speakers=None, want_summary=False,
                       participants=[])
    job.status = jobs.STATUS_DONE
    review.save_raw(job, [
        Segment(0.0, 4.0, "A", "Bom dia a todos, vamos começar a reunião."),
        Segment(4.5, 9.0, "B", "Perfeito, obrigado por reservar a sala."),
    ])
    for k, v in campos.items():
        setattr(job, k, v)
    job.save()
    return job


def _sugestao(nome: str) -> dict:
    return {"nome": nome, "fonte": "turno", "detalhe": "chamado pelo dirigente 1×"}


def test_sugestao_aparece_no_txt_quando_nao_ha_confirmacao(dados):
    job = _job(nomes_auto={"Falante 1": _sugestao("Ademir")})
    segs = review.segments_for(job)
    names = review.names_for(job, segs)
    assert names == {"A": "Ademir", "B": "Falante 2"}
    txt = review.transcript_txt(job)
    assert "Ademir · 00:00:00" in txt and "Falante 2 · 00:00:04" in txt


def test_confirmado_vence_a_sugestao(dados):
    job = _job(nomes_auto={"Falante 1": _sugestao("Ademir")},
               speaker_names={"Falante 1": "Ademir Balbino"})
    names = review.names_for(job, review.segments_for(job))
    assert names["A"] == "Ademir Balbino"


def test_apagar_o_nome_na_revisao_nao_deixa_a_sugestao_voltar(dados):
    # o usuário tinha confirmado "Ademir" e apagou o campo: o marcador "" fica gravado
    job = _job(nomes_auto={"Falante 1": _sugestao("Ademir")},
               speaker_names={"Falante 1": "Ademir"})
    gravado = review.marca_sem_nome({"Falante 1": ""}, set(job.speaker_names))
    assert gravado == {"Falante 1": ""}
    job.speaker_names = gravado
    names = review.names_for(job, review.segments_for(job))
    assert names["A"] == "Falante 1"
    assert "Ademir" not in review.transcript_txt(job)


def test_campo_vazio_de_quem_nunca_teve_nome_nao_grava_marcador():
    # a sugestão continua valendo: nada foi apagado de propósito
    gravado = review.marca_sem_nome({"Falante 1": "", "Falante 2": "Carmen"}, set())
    assert gravado == {"Falante 2": "Carmen"}


def test_marcador_sem_nome_aparece_na_revisao_como_sem_nome(dados):
    job = _job(nomes_auto={"Falante 1": _sugestao("Ademir")},
               speaker_names={"Falante 1": ""})
    linha = next(r for r in review.speakers_overview(job) if r["label"] == "Falante 1")
    assert linha["sem_nome"] is True and linha["nome"] == ""


def test_legenda_e_linha_do_tempo_usam_o_mesmo_nome(dados):
    job = _job(nomes_auto={"Falante 1": _sugestao("Ademir")})
    segs = review.segments_for(job)
    names = review.names_for(job, segs)
    srt = exports.to_srt(segs, names)
    vtt = exports.to_vtt(segs, names)
    assert "Ademir: Bom dia" in srt and "<v Ademir>" in vtt
    assert exports.build_timeline(segs, names)[0][2] == 0


def test_participante_com_erro_de_digitacao_vale_pelo_nome_da_lista():
    lista = [irmaos.Entrada(id="j", nome="José Eduardo", observacao="dirigente",
                            grafias=(), atualizado=0.0),
             irmaos.Entrada(id="m", nome="Marines", observacao="",
                            grafias=("Marinês", "Marilenez"), atualizado=0.0)]
    assert irmaos.corrige_participantes(
        ["José Edaurdo", "Marinez", "Eduardo", "Caon"], lista) == [
        "José Eduardo", "Marines", "Eduardo", "Caon"]


def test_participante_so_com_parte_do_nome_nao_e_trocado():
    lista = [irmaos.Entrada(id="j", nome="José Eduardo", observacao="",
                            grafias=(), atualizado=0.0)]
    assert irmaos.corrige_participante("Eduardo", lista) is None


def test_participantes_sem_repetir_depois_da_correcao():
    lista = [irmaos.Entrada(id="j", nome="José Eduardo", observacao="",
                            grafias=(), atualizado=0.0)]
    assert irmaos.corrige_participantes(["José Eduardo", "José Edaurdo"], lista) == [
        "José Eduardo"]


def test_sugestao_fora_da_lista_e_dos_participantes_tem_confianca_menor():
    from app import worker
    lista = [irmaos.Entrada(id="a", nome="Ademir", observacao="", grafias=(),
                            atualizado=0.0)]
    assert worker._conhecido("Ademir", lista, []) is True
    assert worker._conhecido("Vera", lista, ["Eduardo"]) is False
    assert worker._conhecido("Eduardo", [], ["Eduardo"]) is True
