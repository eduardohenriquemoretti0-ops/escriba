"""Testes da memória de vozes e da dedução de nomes pela conversa.

Rodar:  python -m pytest test_vozes_nomes.py -v
"""
import json

import numpy as np
import pytest

from app import config, vozes
from app.engines.base import Segment
from app.nomes import infer


@pytest.fixture(autouse=True)
def dados_temporarios(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)


def _voz(seed: int, ruido: float = 0.0) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=192)
    if ruido:
        v = v + ruido * np.random.default_rng(seed + 100).normal(size=192)
    return list(v / np.linalg.norm(v))


# --------------------------------------------------------------------------
# Memória de vozes
# --------------------------------------------------------------------------
def test_reconhece_voz_confirmada_em_outra_reuniao():
    vozes.learn("job1", {"A": "Fernando", "B": "Eduardo"},
                {"A": {"vetor": _voz(1), "segundos": 600}, "B": {"vetor": _voz(2), "segundos": 400}})
    # reunião nova: mesmas pessoas, com variação de timbre/microfone, ordem trocada
    achou = vozes.match({"A": _voz(2, 0.4), "B": _voz(1, 0.4)})
    assert achou["A"][0] == "Eduardo" and achou["B"][0] == "Fernando"


def test_voz_desconhecida_nao_vira_ninguem():
    vozes.learn("job1", {"A": "Fernando"}, {"A": {"vetor": _voz(1), "segundos": 60}})
    assert vozes.match({"A": _voz(99)}) == {}


def test_confirmar_de_novo_substitui_e_renomear_move():
    voices = {"A": {"vetor": _voz(1), "segundos": 60}}
    vozes.learn("job1", {"A": "Fernando"}, voices)
    vozes.learn("job1", {"A": "Fernando"}, voices)          # salvou 2x a revisão
    assert vozes.list_people()[0]["reunioes"] == 1
    vozes.learn("job1", {"A": "Fernando Souza"}, voices)    # corrigiu o nome
    assert [p["nome"] for p in vozes.list_people()] == ["Fernando Souza"]
    vozes.learn("job1", {"A": ""}, voices)                   # apagou o nome
    assert vozes.list_people() == []


def test_aprende_com_varias_reunioes_e_esquece():
    for i in range(3):
        vozes.learn(f"job{i}", {"A": "Ana"}, {"A": {"vetor": _voz(5, 0.3 + i / 10), "segundos": 300}})
    pessoa = vozes.list_people()[0]
    assert pessoa["reunioes"] == 3 and pessoa["minutos"] == 15.0
    assert vozes.forget("Ana") and vozes.list_people() == []
    assert not vozes.forget("Ana")


def test_memoria_de_outro_modelo_e_guardada_a_parte(monkeypatch):
    vozes.learn("job1", {"A": "Fernando"}, {"A": {"vetor": _voz(1), "segundos": 60}})
    monkeypatch.setattr(config, "DIARIZATION_MODEL", "/x/outro_modelo.onnx")
    assert vozes.list_people() == []
    assert (config.DATA_DIR / "vozes" / "vozes.nemo_en_titanet_small.json").exists()


def test_arquivo_e_json_legivel():
    vozes.learn("job1", {"A": "Bia"}, {"A": {"vetor": _voz(3), "segundos": 10}})
    data = json.loads((config.DATA_DIR / "vozes" / "vozes.json").read_text())
    assert "Bia" in data["pessoas"]


# --------------------------------------------------------------------------
# Nomes pela conversa
# --------------------------------------------------------------------------
def _seg(t, spk, txt):
    return Segment(t, t + 3, spk, txt)


def test_quem_e_chamado_pelo_nome_e_o_outro():
    segs = [_seg(0, "A", "Oi, tudo bem?"), _seg(4, "B", "Tudo, e você?"),
            _seg(8, "A", "Então fechou, Fernando."), _seg(12, "B", "Fechou."),
            _seg(16, "A", "Fernando, me liga amanhã."), _seg(20, "B", "Ligo sim.")]
    assert infer(segs)["B"][0] == "Fernando"
    assert "A" not in infer(segs)


def test_fala_citada_nao_conta():
    segs = [_seg(0, "A", "Ele disse: você demorou demais, Fernando."), _seg(4, "B", "Pois é."),
            _seg(8, "A", "Ele falou, Fernando."), _seg(12, "B", "Sim.")]
    assert infer(segs) == {}


def test_apresentacao_e_grafia_parecida_com_participante():
    segs = [_seg(0, "A", "Bom dia, aqui é a Rosilene."), _seg(4, "B", "Bom dia.")]
    assert infer(segs, ["Rosilaine Andrade"])["A"][0] == "Rosilaine Andrade"


def test_palavras_comuns_nao_viram_nome():
    segs = [_seg(0, "A", "Então, Beleza."), _seg(4, "B", "Exatamente."),
            _seg(8, "A", "Perfeito, Combinado."), _seg(12, "B", "Valeu, Obrigado.")]
    assert infer(segs) == {}


def test_nao_sugere_para_quem_ja_tem_nome_pela_voz():
    segs = [_seg(0, "A", "Valeu, Fernando."), _seg(4, "B", "Valeu."),
            _seg(8, "A", "Até mais, Fernando."), _seg(12, "B", "Tchau.")]
    assert infer(segs, exclude={"B"}) == {}


# --------------------------------------------------------------------------
# Sugestão por eliminação
# --------------------------------------------------------------------------
from app.nomes import ELIMINACAO_DETALHE, por_eliminacao  # noqa: E402


def test_eliminacao_dois_participantes_um_deduzido():
    # (a) 2 participantes, 1 falante já deduzido -> o outro recebe o restante
    sugerido = por_eliminacao(["Falante 1", "Falante 2"],
                              ["Eduardo Moretti", "Fernando TopVigia"],
                              {"Falante 2": "Fernando TopVigia"})
    assert sugerido == {"Falante 1": "Eduardo Moretti"}
    assert ELIMINACAO_DETALHE == "único participante restante"


def test_eliminacao_tolerante_a_caixa_acento_e_primeiro_nome():
    # o nome deduzido pode ser só o primeiro nome / outra caixa
    sugerido = por_eliminacao(["Falante 1", "Falante 2"],
                              ["Eduardo Moretti", "José Fernando"],
                              {"Falante 1": "FERNANDO"})
    assert sugerido == {"Falante 2": "Eduardo Moretti"}


def test_eliminacao_nao_sugere_com_3_participantes_e_2_vozes():
    # (b) nº de falantes diferente do nº de participantes -> não sugere
    assert por_eliminacao(["Falante 1", "Falante 2"],
                          ["Ana", "Bia", "Carla"], {"Falante 2": "Bia"}) == {}


def test_eliminacao_sem_participantes_nao_sugere():
    # (c) participantes vazios
    assert por_eliminacao(["Falante 1", "Falante 2"], [], {}) == {}
    assert por_eliminacao(["Falante 1", "Falante 2"], ["", "  "], {}) == {}


def test_eliminacao_dois_sem_nome_e_dois_livres_nao_sugere():
    # (d) 2 falantes sem nome e 2 participantes livres -> não sugere
    assert por_eliminacao(["Falante 1", "Falante 2"],
                          ["Eduardo Moretti", "Fernando TopVigia"], {}) == {}


def test_eliminacao_nunca_vai_para_a_memoria_de_vozes():
    # a sugestão é só sugestão: learn() não é chamado com ela
    sugerido = por_eliminacao(["Falante 1", "Falante 2"],
                              ["Eduardo Moretti", "Fernando TopVigia"],
                              {"Falante 2": "Fernando TopVigia"})
    assert sugerido and vozes.list_people() == []
