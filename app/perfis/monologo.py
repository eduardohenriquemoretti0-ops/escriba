"""Perfil monólogo: um só falante.

Um falante só: tenta o nome pela apresentação/chamada e pelo único participante
informado. Sem evidência, fica "[?]" — nunca inventa.
"""
from __future__ import annotations

from .base import Ctx, Perfil


def nomeia(ctx: Ctx) -> dict:
    auto: dict = {}
    ctx.passo_voz_e_conversa(auto)
    ctx.passo_chamada(auto)
    labels = list(ctx.display.values())
    # único falante sem nome e um só participante informado: é ele.
    if len(labels) == 1:
        label = labels[0]
        if label not in auto and len(ctx.participantes) == 1:
            auto[label] = {"nome": ctx.participantes[0], "fonte": "eliminacao",
                           "detalhe": "único falante e único participante informado"}
        elif label not in auto:
            auto[label] = {"nome": "[?] Orador (ouça)", "fonte": "duvida",
                           "detalhe": ctx.com_tempo(label,
                           "monólogo sem nome na fala nem participante informado")}
    ctx.passo_homonimo_e_dedup(auto)
    return auto


PERFIL = Perfil("monologo", "um só falante", nomeia)
