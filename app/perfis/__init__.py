"""Perfis de reunião: um módulo por tipo, escolhido por detecção.

Em vez de mexer no núcleo a cada tipo de reunião, o `worker` detecta o perfil
(`detecta`) e chama a estratégia de nomeação daquele perfil (`nomeia`). Cada
perfil é um módulo:

* ``religiosa``  — congregação: presidente, orador, dirigente, leitor, estudo.
* ``negocios``   — reunião de empresa: quem preside + chamadas/apresentações.
* ``conversa``   — conversa comum: chamadas e apresentações, sem papéis fixos.
* ``monologo``   — um só falante.

O AssemblyAI é ponte; a nomeação por estrutura (chamada, presidente, papéis)
vale também para o motor local próprio (meta de janeiro).
"""
from __future__ import annotations

from . import conversa, monologo, negocios, religiosa
from .base import Ctx, Perfil
from .detector import detecta

_PERFIS: dict[str, Perfil] = {
    p.nome: p for p in (religiosa.PERFIL, negocios.PERFIL,
                        conversa.PERFIL, monologo.PERFIL)
}


def nomeia(job, segments, voices) -> tuple[dict, str]:
    """Detecta o perfil e devolve (`nomes_auto`, nome_do_perfil).

    `nomes_auto` é {label: {nome, fonte, detalhe}} — o worker grava direto."""
    nome = detecta(segments, job)
    perfil = _PERFIS.get(nome, conversa.PERFIL)
    ctx = Ctx.montar(job, segments, voices)
    return perfil.nomeia(ctx), perfil.nome


__all__ = ["detecta", "nomeia", "Ctx", "Perfil"]
