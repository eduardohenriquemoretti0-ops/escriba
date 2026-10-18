"""Perfil reunião de negócios.

Base: nomeia por voz, apresentação/chamada e eliminação (como a conversa). A
mais: marca quem conduz a reunião ("[?] Presidente da reunião") quando dá para
ver pela estrutura — numa reunião de empresa o nome costuma vir da apresentação,
não de uma lista fixa. A expandir: pauta/ata, decisões, responsáveis por tarefa.
"""
from __future__ import annotations

from .. import nomes as _nomes
from .base import Ctx, Perfil


def nomeia(ctx: Ctx) -> dict:
    auto: dict = {}
    ctx.passo_voz_e_conversa(auto)
    ctx.passo_chamada(auto)
    # quem conduz: a 1ª fala que também é dos que mais chamam/agradecem. Se já
    # tem nome (apresentou-se / foi chamado), mantém; senão marca o papel.
    chefe = _nomes.presidente_reuniao(ctx.segments)
    if chefe in ctx.display:
        label = ctx.display[chefe]
        if label not in auto:
            auto[label] = {"nome": "[?] Presidente da reunião (ouça)",
                           "fonte": "duvida",
                           "detalhe": ctx.com_tempo(label,
                           "abre e conduz a reunião · ouça e digite o nome")}
    ctx.passo_homonimo_e_dedup(auto)
    ctx.passo_eliminacao(auto)
    return auto


PERFIL = Perfil("negocios", "reunião de empresa (quem conduz + apresentações)", nomeia)
