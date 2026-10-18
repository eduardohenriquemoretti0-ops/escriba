"""Presidente da reunião de meio de semana e palavras que não são nome.

Regressão dos arranjos de 2026-10-15:
- o presidente (abre e anuncia as partes) é reconhecido pelo cluster da 1ª
  fala que também mais chama/agradece;
- "Que", "Egito" e afins nunca viram nome de falante.

Rodar: python -m pytest tests/test_presidente_reuniao.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import nomes  # noqa: E402
from app.engines.base import Segment  # noqa: E402


def _reuniao() -> list[Segment]:
    # "A" abre, anuncia e agradece as partes (presidente); "B" e "C" comentam.
    return [
        Segment(0.0, 30.0, "A", "Vamos começar. Irmão Bruno, por favor."),
        Segment(31.0, 90.0, "B", "A leitura desta semana fala de fugir para o Egito."),
        Segment(91.0, 95.0, "A", "Muito obrigado, irmão Bruno. Agora joias."),
        Segment(96.0, 150.0, "C", "Uma joia que encontrei nesta semana."),
        Segment(151.0, 156.0, "A", "Isso mesmo, irmão Carlos. Encerramos assim."),
    ]


def test_presidente_e_o_cluster_que_abre_e_anuncia():
    assert nomes.presidente_reuniao(_reuniao()) == "A"


def test_sem_chamadas_nao_arrisca_presidente():
    segs = [Segment(0.0, 60.0, "A", "Leitura corrida sem chamar ninguém."),
            Segment(61.0, 120.0, "B", "Outro trecho qualquer.")]
    assert nomes.presidente_reuniao(segs) is None


def test_que_e_egito_nao_sao_nome():
    assert nomes._nao_e_nome("Que")
    assert nomes._nao_e_nome("Egito")
    assert nomes._nao_e_nome("Judá")
    # nome de pessoa de verdade continua passando
    assert not nomes._nao_e_nome("Fernando")
