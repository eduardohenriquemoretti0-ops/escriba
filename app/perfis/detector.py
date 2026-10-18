"""Detecção do perfil da reunião pelo conteúdo e pelo nº de falantes.

Sem acerto de nenhum sinal forte, o padrão é ``conversa`` (o perfil de reserva).
Pode-se forçar o perfil pela reunião do job (``job.meeting['perfil']``).
"""
from __future__ import annotations

import re
from collections import Counter

from ..engines.base import CORO_LABEL
from .. import nomes as _nomes

_RELIGIOSA = [
    "cantico", "cantica", "jeova", "paragrafo", "sentinela", "congregacao",
    "versiculo", "biblia", "reino", "irmao orador", "discurso publico",
    "pergunta", "estudo biblico",
]
_NEGOCIOS = [
    "pauta", "ordem do dia", "ata da reuniao", "ponto da pauta", "diretoria",
    "faturamento", "cnpj", "balanco", "orçamento", "orcamento", "cronograma",
    "entrega do projeto", "cliente", "contrato", "metas do",
]


def _fold(s: str) -> str:
    return _nomes._fold(s)


def _conta(texto: str, termos: list[str]) -> int:
    return sum(len(re.findall(re.escape(t), texto)) for t in termos)


def detecta(segments, job=None) -> str:
    """Nome do perfil: 'religiosa' | 'negocios' | 'conversa' | 'monologo'."""
    forcado = (getattr(job, "meeting", None) or {}).get("perfil")
    if forcado:
        return forcado
    segs = [s for s in segments if getattr(s, "speaker", None) != CORO_LABEL
            and s.text.strip()]
    if not segs:
        return "conversa"
    falantes = len({s.speaker for s in segs})
    if falantes <= 1:
        return "monologo"
    texto = _fold(" ".join(s.text for s in segs))
    rel = _conta(texto, _RELIGIOSA)
    neg = _conta(texto, _NEGOCIOS)
    # religiosa tem sinais muito característicos (cântico, Jeová, parágrafo):
    # 3+ ocorrências já decidem. Negócios precisa superar e ter sinal claro.
    if rel >= 3 and rel >= neg:
        return "religiosa"
    if neg >= 3 and neg > rel:
        return "negocios"
    return "conversa"
