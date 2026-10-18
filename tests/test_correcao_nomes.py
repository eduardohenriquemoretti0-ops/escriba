"""Correção de nome resolvido no texto (app/correcao_nomes).

Garante que garble de nome vira o nome resolvido por fonética, sem atropelar
outro irmão real nem mexer em garble severo (que é do passe de IA)."""
from app.correcao_nomes import corrige

ALVOS = ["Ademir Baldino", "Rinaldo Santos", "José Eduardo Silveira", "Wellington Passos"]


def test_corrige_sobrenome_parecido():
    assert corrige("irmão Ademir Balbino, que", ALVOS) == "irmão Ademir Baldino, que"
    assert corrige("ao Ademir Baulino hoje", ALVOS) == "ao Ademir Baldino hoje"


def test_nao_atropela_outro_irmao_real():
    # "Ademir Duarte" é outra pessoa da lista; não pode virar Baldino
    assert corrige("irmão Ademir Duarte comentou", ALVOS) == "irmão Ademir Duarte comentou"


def test_garble_severo_fica():
    # "Palpino"/"Módulo Vir" não casam por fonética; ficam para o passe de IA
    assert "Palpino" in corrige("Módulo Vir, Palpino, esposa", ALVOS)


def test_nome_ja_certo_mantem():
    assert corrige("o irmão Ademir Baldino, sua esposa", ALVOS) == "o irmão Ademir Baldino, sua esposa"


def test_sem_alvos_nao_mexe():
    assert corrige("qualquer texto Fulano Beltrano", []) == "qualquer texto Fulano Beltrano"
