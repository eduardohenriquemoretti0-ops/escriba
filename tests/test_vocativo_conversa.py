"""Nomes em conversa/negócios pela chamada no começo da frase (regra do Eduardo,
2026-10-18): "ah Fulano", "então Siclano", "mas Beltrano" -> quem responde logo
depois é o chamado, também com mais de 2 pessoas."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import nomes  # noqa: E402
from app.engines.base import Segment  # noqa: E402


def _conversa(falas):
    t, segs = 0.0, []
    for quem, texto in falas:
        segs.append(Segment(t, t + 3.0, quem, texto))
        t += 3.5
    return segs


def test_entao_e_mas_com_tres_pessoas():
    segs = _conversa([
        ("A", "Bom dia a todos, vamos começar a reunião de vendas."),
        ("A", "Então Carlos, como fechou o mês?"),
        ("B", "Fechou bem, batemos a meta com folga."),
        ("C", "Eu acho que dá para crescer mais no interior."),
        ("A", "Mas Carlos você acha que o estoque aguenta?"),
        ("B", "Aguenta sim, já pedi reposição."),
    ])
    assert nomes.infer(segs).get("B", ("",))[0] == "Carlos"


def test_ah_fulano_no_fim():
    segs = _conversa([
        ("A", "Quem trouxe o relatório?"),
        ("C", "Fui eu."),
        ("A", "Ah, Renata."),
        ("C", "Está tudo na pasta compartilhada."),
        ("B", "Vou olhar depois."),
        ("A", "Então Renata, manda para o diretor."),
        ("C", "Mando hoje ainda."),
    ])
    assert nomes.infer(segs).get("C", ("",))[0] == "Renata"


def test_narracao_nao_e_chamada():
    """'Mas Pedro foi lá' é narração: Pedro não está na conversa."""
    segs = _conversa([
        ("A", "Mas Pedro foi lá e resolveu tudo sozinho."),
        ("B", "Que bom, ele sempre resolve."),
        ("C", "Então Pedro viajou de novo?"),
        ("B", "Viajou, volta segunda."),
        ("A", "Mas Pedro disse que voltava antes."),
        ("B", "Mudou de ideia."),
    ])
    assert nomes.infer(segs).get("B", ("",))[0] != "Pedro"
