"""Detecção de perfil de reunião."""
from app.engines.base import Segment
from app import perfis


def _seg(a, b, spk, txt):
    return Segment(start=a, end=b, speaker=spk, text=txt)


def test_detecta_religiosa():
    segs = [
        _seg(0, 5, "A", "Vamos entoar o cântico de número 56 a Jeová."),
        _seg(6, 40, "B", "Parágrafo 2: em uma guerra os soldados precisam confiar."),
        _seg(41, 45, "A", "Pergunta 2: por que confiar em Jeová? Irmão orador."),
    ]
    assert perfis.detecta(segs) == "religiosa"


def test_detecta_negocios():
    segs = [
        _seg(0, 5, "A", "Vamos à pauta da reunião, ordem do dia."),
        _seg(6, 10, "B", "Primeiro ponto da pauta: o faturamento do cliente."),
        _seg(11, 15, "A", "O contrato e o orçamento do projeto."),
    ]
    assert perfis.detecta(segs) == "negocios"


def test_detecta_monologo():
    segs = [_seg(0, 60, "A", "Este é um áudio de um só falante do começo ao fim.")]
    assert perfis.detecta(segs) == "monologo"


def test_detecta_conversa_padrao():
    segs = [_seg(0, 5, "A", "E aí, beleza?"), _seg(6, 10, "B", "Tudo e você?")]
    assert perfis.detecta(segs) == "conversa"


def test_perfil_forcado_pela_reuniao():
    segs = [_seg(0, 5, "A", "qualquer coisa"), _seg(6, 10, "B", "outra coisa")]

    class J:
        meeting = {"perfil": "negocios"}

    assert perfis.detecta(segs, J()) == "negocios"
