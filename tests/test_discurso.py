"""Parte 1 da reunião: presidente, orador, fragmentos e Rinaldo nos dois papéis.

Sem modelos: a voz é um vetor por cluster, e a estrutura vem do texto.
Rodar: python -m pytest tests/test_discurso.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import discurso, irmaos, turnos  # noqa: E402
from app.engines.base import Segment  # noqa: E402

RINALDO = irmaos.Entrada(id="r1", nome="Rinaldo", observacao="presidente",
                         grafias=("Reinaldo",), atualizado=0.0)
ADEMIR = irmaos.Entrada(id="a1", nome="Ademir", observacao="orador visitante",
                        grafias=("Admir",), atualizado=0.0)
ENTRADAS = [RINALDO, ADEMIR]
PARTICIPANTES = ["Ademir", "Eduardo"]
TEMA = "como sempre tomar as melhores decisões"


def _voz(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=64)
    return v / np.linalg.norm(v)


SEED_POR_CLUSTER = {"A": 1, "B": 2, "C": 5, "E": 6, "F": 7}


def _embed(a: float, b: float) -> np.ndarray:
    # o presidente e o orador usam a MESMA voz (mesmo microfone), como na reunião real
    return _voz(9)


def _seg(ini, fim, falante, texto):
    return Segment(float(ini), float(fim), falante, texto)


def _reuniao(com_agradecimento: bool = True):
    segs = [
        _seg(0, 30, "A", "Vimos aos nossos irmãos, visitantes, pessoas que nos visitam ali."),
        _seg(31, 33, "B", "Amém."),
        _seg(87, 120, "A", "Vamos boas-vindas também ao irmão Ademir Balbino, ele fará o discurso."),
        _seg(121, 125, "C", "Concordo com isso mesmo, é muito bom."),
        _seg(387, 405.5, "A", "Novamente damos boas-vindas a todos, irmão orador, Módemir, "
                              "esposa. Temos um discurso com o seguinte tema, "
                              "como sempre tomar as melhores decisões, como sempre tomar as "
                              "melhores decisões."),
        _seg(405.6, 407.4, "B", "Ele é"),
        _seg(409.3, 411.3, "A", "da congregação Nova Cidade Jardim."),
        _seg(427.4, 700, "A", "É muito bom quando nós recebemos um presente de alguém, "
                              "nós nos sentimos amados e acolhidos."),
        _seg(700.5, 701, "B", "É"),
        _seg(701.5, 800, "A", "O seguinte, os planos do dirigente certamente dão bons resultados."),
        _seg(800.5, 801, "C", "Jeová?"),
        _seg(801.5, 1000, "A", "Provérbios 27, 11. Olha só o que nos diz. Seja sábio, meu filho."),
    ]
    if com_agradecimento:
        segs.append(_seg(1002, 1010, "A", "Muito obrigado irmão Ademir, excelente proferimento, "
                                          "nos ajudou bastante."))
    segs += [
        _seg(1011, 1020, "B", "Vamos continuar com o estudo da Sentinela. "
                              "Sim, irmão Reinaldo, por favor."),
        _seg(1021, 1040, "C", "Eu acho que a oração ajuda muito na família. Ela é importante."),
        _seg(1041, 1044, "B", "Obrigado, irmão Reinaldo. Irmã Beltrana, por favor."),
        _seg(1045, 1060, "E", "Concordo, a oração é importante para a família inteira."),
        _seg(1061, 1063, "B", "Muito bem, irmã Beltrana."),
        _seg(1064, 1070, "F", "Eu lembro de uma experiência com a oração."),
    ]
    return segs


def _pessoa(pessoas, nome):
    achadas = [p for p in pessoas if p.nome == nome]
    assert len(achadas) == 1, f"{nome}: {[p.nome for p in pessoas]}"
    return achadas[0]


# --------------------------------------------------------------------------
# Absorção de fragmentos
# --------------------------------------------------------------------------
def test_fragmento_entre_dois_blocos_longos_do_mesmo_falante_e_absorvido():
    segs = [_seg(0, 20, "A", "Um discurso longo com muitas palavras."),
            _seg(21, 21.5, "B", "É"),
            _seg(22, 40, "A", "Continua o discurso com mais palavras aqui.")]
    out, n = discurso.absorve_fragmentos(segs)
    assert n == 1 and out[1].speaker == "A"


def test_fragmento_entre_falantes_diferentes_nao_e_absorvido():
    segs = [_seg(0, 20, "A", "Um discurso longo com muitas palavras."),
            _seg(21, 21.5, "B", "É"),
            _seg(22, 40, "C", "Outra pessoa falando bastante coisa aqui.")]
    out, n = discurso.absorve_fragmentos(segs)
    assert n == 0 and out[1].speaker == "B"


def test_fragmento_entre_blocos_curtos_nao_e_absorvido():
    segs = [_seg(0, 3, "A", "Bom dia a todos."),
            _seg(4, 4.5, "B", "Sim"),
            _seg(5, 8, "A", "Vamos começar o estudo agora.")]
    out, n = discurso.absorve_fragmentos(segs)
    assert n == 0 and out[1].speaker == "B"


# --------------------------------------------------------------------------
# Parte 1: presidente, orador e agradecimento
# --------------------------------------------------------------------------
def test_orador_e_presidente_separados_mesmo_com_a_mesma_voz():
    segs = _reuniao()
    out, pessoas, stats = turnos.reatribui(
        segs, _embed, irmaos=ENTRADAS, participantes=PARTICIPANTES)
    rot = {s.start: s.speaker for s in out}
    pres = rot[0.0]
    ora = rot[427.4]
    assert pres != ora
    assert rot[387.0] == pres          # o anúncio é do presidente
    assert rot[1002.0] == pres         # o agradecimento é do presidente
    assert stats["parte1"]["tema"] == TEMA


def test_discurso_vira_um_bloco_so_do_orador():
    out, _, _ = turnos.reatribui(_reuniao(), _embed, irmaos=ENTRADAS,
                                 participantes=PARTICIPANTES)
    ora = next(s.speaker for s in out if s.start == 427.4)
    blocos = [s for s in out if s.speaker == ora]
    assert len(blocos) == 1
    assert blocos[0].start == 427.4 and blocos[0].end == 1000
    # os fragmentos de outros falantes dentro do discurso sumiram
    assert not [s for s in out if 427.4 <= s.start < 1000 and s.speaker != ora]


def test_anuncio_mais_antigo_nao_e_o_do_discurso():
    """O anúncio das 87 s ("ele fará o discurso") não tem agradecimento depois
    com o mesmo nome: o discurso começa no anúncio das 387 s."""
    _, _, stats = turnos.reatribui(_reuniao(), _embed, irmaos=ENTRADAS,
                                   participantes=PARTICIPANTES)
    p1 = stats["parte1"]
    assert p1["anuncio_inicio"] == 387.0
    assert p1["orador_inicio"] == 427.4


def test_nome_do_orador_vem_dos_participantes_quando_o_anuncio_erra():
    _, _, stats = turnos.reatribui(_reuniao(), _embed, irmaos=ENTRADAS,
                                   participantes=PARTICIPANTES)
    p1 = stats["parte1"]
    assert p1["orador_cru"] == "Módemir"       # o que o Whisper escreveu
    assert p1["orador"] == "Ademir"            # pela lista/participantes
    assert p1["orador_confianca"] == "media"


def test_sem_agradecimento_nao_ha_estrutura_de_discurso():
    _, pessoas, stats = turnos.reatribui(_reuniao(com_agradecimento=False), _embed,
                                         irmaos=ENTRADAS, participantes=PARTICIPANTES)
    assert stats["parte1"] is None
    assert all(p.papel != "orador" for p in pessoas)


# --------------------------------------------------------------------------
# Rinaldo: presidente e comentarista são UMA pessoa na lista de falantes
# --------------------------------------------------------------------------
def test_rinaldo_presidente_e_comentarista_viram_um_falante_com_os_dois_papeis():
    out, pessoas, stats = turnos.reatribui(_reuniao(), _embed, irmaos=ENTRADAS,
                                           participantes=PARTICIPANTES)
    rinaldo = _pessoa(pessoas, "Rinaldo")
    assert "presidente" in rinaldo.papel and "comentarista" in rinaldo.papel
    assert rinaldo.nomeada
    # "Reinaldo" (grafia) chamado no estudo é o mesmo rótulo do presidente
    comentario = next(s for s in out if s.start == 1021.0)
    assert comentario.speaker == next(s.speaker for s in out if s.start == 0.0)
    # e o mesmo falante aparece uma vez só na lista
    assert [p.rotulo for p in pessoas].count(rinaldo.rotulo) == 1


def test_orador_aparece_como_orador_e_nao_como_comentarista():
    _, pessoas, _ = turnos.reatribui(_reuniao(), _embed, irmaos=ENTRADAS,
                                     participantes=PARTICIPANTES)
    ademir = _pessoa(pessoas, "Ademir")
    assert ademir.papel == "orador"
    assert "tema:" in ademir.origem


def test_presidente_sem_entrada_na_lista_fica_sem_nome():
    _, pessoas, _ = turnos.reatribui(_reuniao(), _embed, irmaos=(ADEMIR,),
                                     participantes=PARTICIPANTES)
    pres = next(p for p in pessoas if p.papel.startswith("presidente"))
    assert pres.nome == "Presidente" and not pres.nomeada


def test_conversa_de_duas_pessoas_nao_perde_o_sim_do_outro():
    """Sem discurso e sem estudo, "Sim." de B entre dois blocos longos de A é fala real."""
    segs = [_seg(0, 20, "A", "Vou te contar sobre a viagem que fizemos no ano passado."),
            _seg(20.2, 20.6, "B", "Sim."),
            _seg(21, 40, "A", "Então fomos para a praia e ficamos uma semana inteira.")]
    out, pessoas, stats = turnos.reatribui(segs, _embed, irmaos=ENTRADAS,
                                           participantes=PARTICIPANTES)
    assert [s.speaker for s in out] == ["A", "B", "A"]
    assert stats["ativo"] is False


def test_tema_nao_engole_a_apresentacao_do_orador():
    texto = ("Temos um discurso com o seguinte tema, como sempre tomar as melhores decisões, "
             "como sempre tomar as melhores decisões, ele é da congregação Nova Cidade Jardim.")
    assert discurso._tema(texto) == TEMA


def test_participante_com_grafia_do_whisper_vale_no_estudo():
    """"Modemir" (chamada no estudo) é o participante Ademir digitado: vale pelo nome."""
    segs = _reuniao()
    segs.append(_seg(1080, 1090, "C", "Ainda uma palavra sobre a oração de hoje."))
    segs.append(_seg(1091, 1093, "B", "Obrigado, Modemir. Irmã Beltrana, por favor."))
    out, pessoas, _ = turnos.reatribui(segs, _embed, irmaos=ENTRADAS,
                                       participantes=PARTICIPANTES)
    assert any(p.nome == "Ademir" and "Modemir" not in p.nome for p in pessoas)
    assert not any(p.nome == "Modemir" for p in pessoas)
