"""Testes pequenos de nomes e eventos do motor (app.estrutura).
Rodar: PYTHONPATH=. python -m pytest tests/estrutura -q"""
import json
from pathlib import Path

from app.estrutura import Lista, Regras, fonetica

AQUI = Path(__file__).resolve().parent
LISTA = Lista.de_json(json.loads((AQUI / "dados/lista_irmaos.json").read_text(encoding="utf-8")))
R = Regras.carregar()


def nome(ouvido, genero=None):
    return LISTA.resolver(ouvido, genero)


def test_fonetica():
    assert fonetica("Matheus") == fonetica("Mateus")
    assert fonetica("Caom") == fonetica("Caon")
    assert fonetica("Kátya") == fonetica("Katia")
    assert fonetica("Marinez") == fonetica("Marines")


def test_resolve_unico_e_duvida():
    assert nome("Matheus Santos").nome == "Matheus Santos"
    assert nome("Elza").nome == "Elza Almeida"
    assert nome("Caom", "m").nome == "José Carlos Caon"
    assert nome("Fernando").duvida
    assert nome("Marcos").duvida
    assert nome("Fernando Croci").nome == "Fernando Croce"
    assert nome("Ademir Balbino").duvida
    assert nome("Toma Denise").nome == "Denise Bezerra"


def _tipos(texto, tipo):
    return [(e.tipo, e.nome) for e in R.eventos(texto) if e.tipo == tipo]


def test_chamadas():
    assert ("chamada", "Fátima") in _tipos("Jeremias 23:26-27. Irmã Fátima, por gentileza.", "chamada")
    assert ("chamada", "Matheus Santos") in _tipos("Agora vamos pedir para o Matheus Santos.", "chamada")
    assert ("chamada", "Jailson") in _tipos("Qual que é, Jailson?", "chamada")
    assert not _tipos("Meio que a Irmã Rita comentou, né?", "chamada")
    assert not _tipos("muito importante a parte do irmão Berto, nesses estados", "chamada")


def test_agradecimento_e_paragrafo():
    assert ("agradecimento", "Carmem Carlotti") in _tipos("Muito obrigado, irmã Carmem Carlotti. Irmã Edna.", "agradecimento")
    assert _tipos("Vamos ao 12.", "paragrafo")
    assert _tipos("8.", "paragrafo")
    assert R.e_retorno("Isso mesmo.") and R.e_retorno("Como?") and not R.e_retorno("O que me pode fazer?")
