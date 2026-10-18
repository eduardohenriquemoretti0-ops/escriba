"""Perfil conversa comum: sem papéis fixos.

Nomeia por voz reconhecida, apresentação/chamada na conversa e eliminação. É o
perfil de reserva quando nenhum outro bate.
"""
from __future__ import annotations

from .base import Ctx, Perfil


def nomeia(ctx: Ctx) -> dict:
    auto: dict = {}
    ctx.passo_voz_e_conversa(auto)
    ctx.passo_chamada(auto)
    ctx.passo_homonimo_e_dedup(auto)
    ctx.passo_eliminacao(auto)
    return auto


PERFIL = Perfil("conversa", "conversa comum, sem papéis fixos", nomeia)
