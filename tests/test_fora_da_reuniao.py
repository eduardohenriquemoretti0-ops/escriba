"""Fala depois do encerramento: fica no texto, mas não vira participante."""
from app import atribuicao, exports
from app.engines.base import FORA_LABEL, FORA_NOME


def _atrib():
    return {"falas": [
        {"ini": 0, "fim": 5, "texto": "Bom dia.", "quem": "Rinaldo Santos", "papel": "presidente"},
        {"ini": 5, "fim": 6, "texto": "Amém.", "quem": "Em coro", "papel": "amém"},
        {"ini": 7, "fim": 8, "texto": "Obrigado.", "quem": "(fora da reunião)", "papel": "fora"},
    ]}


def test_fora_tem_rotulo_proprio_e_nao_consome_falante():
    segs = atribuicao.segmentos(_atrib())
    assert [s.speaker for s in segs] == ["P01", "CORO", FORA_LABEL]
    mapa = exports.speaker_map(segs)
    assert mapa == {"P01": "Falante 1", "CORO": "Em coro", FORA_LABEL: FORA_NOME}


def test_fora_nao_entra_nos_nomes():
    assert list(atribuicao.nomes_auto(_atrib())) == ["Falante 1"]
