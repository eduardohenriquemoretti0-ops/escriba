"""Chamada fora da lista ("Entendeu?") não substitui um nome já firme."""
from app.perfis import base


def _ctx():
    return base.Ctx(job=None, segments=[], voices={}, display={"B": "Falante 2"},
                    inicio={"Falante 2": 5.0}, entradas=[], participantes=[],
                    by_voice={}, strong=set(), by_talk={})


def test_chamada_em_duvida_nao_apaga_nome_da_conversa(monkeypatch):
    monkeypatch.setattr(base.nomes, "por_chamada",
                        lambda *a: {"B": ("Entendeu", "anunciado por quem preside", False)})
    auto = {"Falante 2": {"nome": "Fernando TopVigia", "fonte": "conversa",
                          "detalhe": "foi chamado pelo nome 5×"}}
    _ctx().passo_chamada(auto)
    assert auto["Falante 2"]["nome"] == "Fernando TopVigia"


def test_chamada_em_duvida_ainda_marca_quem_nao_tem_nome(monkeypatch):
    monkeypatch.setattr(base.nomes, "por_chamada",
                        lambda *a: {"B": ("Entendeu", "anunciado por quem preside", False)})
    auto: dict = {}
    _ctx().passo_chamada(auto)
    assert auto["Falante 2"]["fonte"] == "duvida"


def test_hotwords_usa_nome_falado():
    from app.engines.local_engine import _nome_falado
    todos = ["Carmem Carlotti", "José Eduardo Silveira", "José Carlos Caon", "Caio"]
    assert _nome_falado("Carmem Carlotti", todos) == "Carmem"
    assert _nome_falado("José Eduardo Silveira", todos) == "José Eduardo"
    assert _nome_falado("Caio", todos) == "Caio"
