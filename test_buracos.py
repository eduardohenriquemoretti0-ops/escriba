"""Testes da retranscrição dos trechos pulados (app/buracos.py), sem modelos.

Rodar:  python -m pytest test_buracos.py -v
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from app import buracos
from app.engines import local_engine
from app.engines.diarize import SAMPLE_RATE, Sentence, Word

SR = SAMPLE_RATE


def _seno(segundos: float, dbfs: float) -> np.ndarray:
    t = np.arange(int(segundos * SR)) / SR
    amp = 10 ** (dbfs / 20) * np.sqrt(2)  # RMS do seno = amp/√2
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _seg(start, end, text, palavras, logprob=-0.3, no_speech=0.1):
    return SimpleNamespace(
        start=start,
        end=end,
        text=text,
        avg_logprob=logprob,
        no_speech_prob=no_speech,
        words=[SimpleNamespace(start=a, end=b, word=w) for a, b, w in palavras],
    )


# --------------------------------------------------------------------------
# Achar buracos
# --------------------------------------------------------------------------
def test_acha_buraco_com_fala_no_vad():
    palavras = [(0.0, 1.0), (1.2, 2.0), (12.0, 13.0)]
    falas = [(0.0, 2.0), (4.0, 9.0), (12.0, 13.0)]
    achados = buracos.acha_buracos(20.0, palavras, [], falas)
    assert len(achados) == 1
    b = achados[0]
    assert 2.0 < b.inicio < 2.1 and 11.9 < b.fim < 12.0 and b.fala_s == 5.0


def test_coro_e_pausa_sem_fala_nao_sao_buraco():
    palavras = [(0.0, 1.0), (12.0, 13.0), (30.0, 31.0)]
    falas = [(3.0, 9.0), (14.0, 14.5)]
    # 1º intervalo: dentro do coro; 2º: só 0,5 s de fala (< FALA_MIN)
    achados = buracos.acha_buracos(31.0, palavras, [(2.0, 10.0)], falas)
    assert achados == []


def test_intervalo_curto_nao_e_buraco():
    assert buracos.acha_buracos(5.0, [(0.0, 1.0), (2.5, 3.0)], [], [(1.0, 2.5)]) == []


# --------------------------------------------------------------------------
# Ganho
# --------------------------------------------------------------------------
def test_ganho_leva_fala_baixa_ao_alvo():
    trecho = _seno(4.0, -35.0)
    y, g = buracos.aplica_ganho(trecho, [(10.0, 14.0)], 10.0)
    assert abs(g - 12.0) < 0.2
    assert abs(buracos.dbfs(y) - buracos.RMS_ALVO) < 0.3


def test_ganho_tem_teto_e_nao_estoura():
    y, g = buracos.aplica_ganho(_seno(2.0, -60.0), [(0.0, 2.0)], 0.0)
    assert g == buracos.GANHO_MAX
    alto = np.concatenate([_seno(1.0, -50.0), np.full(10, 0.2, np.float32)])
    y, g = buracos.aplica_ganho(alto, [(0.0, 1.0)], 0.0)
    assert float(np.max(np.abs(y))) <= 0.99 + 1e-6 and g < buracos.GANHO_MAX


def test_fala_ja_alta_nao_recebe_ganho():
    _y, g = buracos.aplica_ganho(_seno(2.0, -15.0), [(0.0, 2.0)], 0.0)
    assert g == 0.0


# --------------------------------------------------------------------------
# Retranscrição
# --------------------------------------------------------------------------
def _audio_com_buraco():
    audio = np.zeros(int(20 * SR), dtype=np.float32)
    audio[int(4 * SR) : int(9 * SR)] = _seno(5.0, -34.0)
    return audio, [(0.0, 1.0), (12.0, 13.0)], [(4.0, 9.0)]


def test_retranscreve_poe_palavras_no_tempo_absoluto_e_filtra():
    audio, palavras, falas = _audio_com_buraco()
    recebidos = []

    def transcrever(y):
        recebidos.append(buracos.dbfs(y))
        return [
            _seg(
                2.9,
                4.0,
                " Bom dia irmãos.",
                [(2.9, 3.3, " Bom"), (3.3, 3.6, " dia"), (3.6, 4.0, " irmãos.")],
            ),
            _seg(5.0, 6.0, " Obrigado por assistir.", [(5.0, 6.0, " Obrigado")]),
            _seg(6.0, 7.0, " ruído", [(6.0, 7.0, " ruído")], logprob=-1.5),
            _seg(7.0, 8.0, " nada", [(7.0, 8.0, " nada")], no_speech=0.9),
            _seg(30.0, 31.0, " fora", [(30.0, 31.0, " fora")]),
        ]

    novos, rel = buracos.retranscreve(
        audio,
        palavras,
        [],
        transcrever,
        teto_cpu_s=60,
        e_alucinacao=local_engine._is_hallucination,
        falas=falas,
    )
    assert recebidos and recebidos[0] > -30, "o trecho chega com ganho"
    assert len(novos) == 1
    assert [w.text for w in novos[0]] == [" Bom", " dia", " irmãos."]
    assert abs(novos[0][0].start - (rel.detalhes[0]["inicio"] + 2.9)) < 1e-6
    motivos = [r["motivo"] for r in rel.detalhes[0]["recusados"]]
    assert motivos[0] == "alucinação conhecida"
    assert motivos[1].startswith("avg_logprob") and motivos[2].startswith("no_speech")
    assert motivos[3] == "sem palavras"
    assert rel.buracos == 1 and rel.retranscritos == 1 and rel.palavras == 3


def test_teto_zero_desliga():
    audio, palavras, falas = _audio_com_buraco()

    def nunca(_y):
        raise AssertionError("não deveria transcrever")

    novos, rel = buracos.retranscreve(
        audio,
        palavras,
        [],
        nunca,
        teto_cpu_s=0,
        e_alucinacao=lambda _t: False,
        falas=falas,
    )
    assert novos == [] and rel.retranscritos == 0


def test_teto_de_cpu_corta_e_avisa_no_progresso():
    audio = np.zeros(int(60 * SR), dtype=np.float32)
    palavras = [(0.0, 1.0), (10.0, 11.0), (20.0, 21.0), (30.0, 31.0)]
    falas = [(2.0, 8.0), (12.0, 15.0), (22.0, 29.0), (32.0, 50.0)]
    fases = []

    def caro(_y):
        t0 = time.process_time()
        while time.process_time() - t0 < 0.05:
            pass  # gasta CPU de verdade
        return []

    _novos, rel = buracos.retranscreve(
        audio,
        palavras,
        [],
        caro,
        teto_cpu_s=0.04,
        e_alucinacao=lambda _t: False,
        progress=fases.append,
        falas=falas,
    )
    assert rel.buracos == 4 and rel.retranscritos == 1 and rel.pulados_teto == 3
    # o de mais fala vai primeiro (32–50 s)
    assert rel.detalhes[0]["inicio"] > 31
    assert any("Recuperando trechos pulados" in f for f in fases)
    assert any("teto de CPU" in f for f in fases)


# --------------------------------------------------------------------------
# Motor local: frases dos buracos vão para a voz mais parecida
# --------------------------------------------------------------------------
class _ExtratorFalso:
    def __init__(self, voz):
        self.voz = voz

    def embed(self, samples):
        return self.voz


def test_frases_dos_buracos_vao_para_o_centroide_mais_parecido(monkeypatch):
    v0, v1 = np.eye(4, dtype=np.float32)[0], np.eye(4, dtype=np.float32)[1]
    centroides = {0: v0, 1: v1}
    antigas = [
        Sentence([Word(0.0, 2.0, " Olá.")], 0),
        Sentence([Word(20.0, 22.0, " Tchau.")], 1),
    ]
    novas_palavras = [[Word(10.0, 10.5, " Bom"), Word(10.5, 11.6, " dia.")]]
    monkeypatch.setattr(local_engine.diarize, "_embed_span", lambda ex, a, i, f: v1)
    out = local_engine._junta_buracos(
        antigas,
        centroides,
        novas_palavras,
        _ExtratorFalso(v1),
        np.zeros(SR * 30, np.float32),
    )
    assert [s.text for s in out] == ["Olá.", "Bom dia.", "Tchau."]
    assert out[1].speaker == 1 and out[1].embedding is not None


def test_parametros_do_whisper_sem_initial_prompt():
    kw = local_engine._whisper_kwargs("pt")
    assert "initial_prompt" not in kw
    assert kw["condition_on_previous_text"] is False
    assert kw["word_timestamps"] is True
