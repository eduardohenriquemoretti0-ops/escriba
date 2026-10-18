"""Programa da reunião vindo do formulário (tipo + partes)."""
import json

from app import main


def test_programa_limpa_e_descarta_vazios():
    raw = json.dumps([
        {"papel": "tesouros_discurso", "rotulo": "1. Discurso", "nomes": ["Ademir Baldino"]},
        {"papel": "ministerio_parte_4", "nomes": ["Ana Paula", " ", "Maria Silva"]},
        {"papel": "vida_crista_parte_7", "nomes": []},
        {"papel": "x<script>", "nomes": ["Fulano"]},
        "lixo",
    ])
    partes = main._programa(raw)
    assert [p["papel"] for p in partes] == ["tesouros_discurso", "ministerio_parte_4", "xscript"]
    assert partes[1]["nomes"] == ["Ana Paula", "Maria Silva"]


def test_programa_invalido_vira_vazio():
    assert main._programa("{nao é json") == []
    assert main._programa('{"a": 1}') == []
    assert main._programa("") == []


def test_tipos_do_formulario():
    assert main._PERFIS_FORM["religiosa_meio_de_semana"] == ("religiosa", "meio_de_semana")
    assert main._PERFIS_FORM["religiosa_fim_de_semana"] == ("religiosa", "fim_de_semana")
    assert main._PERFIS_FORM["negocios"] == ("negocios", None)
