"""Perfil reunião religiosa (congregação).

Estrutura manda, voz confirma (regra do Eduardo). Ordem das sugestões, da mais
forte para a mais fraca: voz reconhecida > presidente/papel do fim de semana >
turno do estudo > chamada > conversa > eliminação. Na dúvida, "[?]" com o tempo.
"""
from __future__ import annotations

from .. import fim_de_semana
from .base import Ctx, Perfil


def nomeia(ctx: Ctx) -> dict:
    auto: dict = {}
    ctx.passo_voz_e_conversa(auto)

    # turnos do estudo (chamada/agradecimento do dirigente), só motor local
    for pessoa in ctx.job.falantes_estudo or []:
        if pessoa.get("duvida") and pessoa["rotulo"] in ctx.display:
            dlabel = ctx.display[pessoa["rotulo"]]
            auto[dlabel] = {
                "nome": pessoa["nome"], "fonte": "duvida",
                "detalhe": ctx.com_tempo(dlabel,
                (pessoa.get("origem") or "sem chamada/agradecimento")
                + " · ouça o trecho e digite o nome")}
            continue
        if not pessoa.get("nomeada") or pessoa["rotulo"] not in ctx.display:
            continue
        label = ctx.display[pessoa["rotulo"]]
        if auto.get(label, {}).get("fonte") == "voz":
            continue
        detalhe = pessoa["origem"] or "estrutura do estudo"
        if not ctx.conhecido(pessoa["nome"]):
            detalhe += " · fora da lista e dos participantes: confirme"
        auto[label] = {"nome": pessoa["nome"], "fonte": "turno", "detalhe": detalhe}

    ctx.passo_chamada(auto)

    # presidente: abre e fecha a reunião, mas NUNCA é fixo (pode ser qualquer
    # irmão, no meio ou no fim de semana). Por isso não se crava a observação
    # "presidente" da lista como nome. Identifica o cluster que preside pela
    # estrutura (abre/fecha + conduz) e, se o nome não veio por voz/chamada,
    # marca o papel para o usuário ouvir e confirmar.
    from .. import nomes as _nomes
    pres_cluster = _nomes.presidente_reuniao(ctx.segments)
    if pres_cluster in ctx.display:
        label = ctx.display[pres_cluster]
        fonte = auto.get(label, {}).get("fonte")
        if fonte in (None, "duvida", "conversa", "eliminacao"):
            auto[label] = {"nome": "[?] Presidente (abre e fecha a reunião)",
                           "fonte": "presidente",
                           "detalhe": ctx.com_tempo(label,
                           "abre e encerra a reunião · ouça e confirme")}

    # papéis do fim de semana (orador, dirigente, leitor): cluster pela estrutura,
    # nome pelo anúncio do presidente. Vence chamada/conversa, perde só p/ voz.
    for raw, p in fim_de_semana.papeis(ctx.segments, ctx.entradas,
                                       ctx.participantes).items():
        label = ctx.display.get(raw)
        if not label or auto.get(label, {}).get("fonte") == "voz":
            continue
        if p.nome and p.na_lista:
            auto[label] = {"nome": p.nome, "fonte": "papel", "detalhe": p.detalhe}
        elif p.nome:
            auto[label] = {"nome": f"[?] {p.nome}?", "fonte": "duvida",
                           "detalhe": ctx.com_tempo(label, f"{p.detalhe} · fora da "
                           "lista · ouça o trecho e confirme a grafia")}
        else:
            auto[label] = {"nome": f"[?] {p.papel.capitalize()} (ouça)",
                           "fonte": "duvida",
                           "detalhe": ctx.com_tempo(label, p.detalhe)}

    ctx.passo_homonimo_e_dedup(auto)
    ctx.passo_eliminacao(auto)
    return auto


PERFIL = Perfil("religiosa", "reunião de congregação "
                "(presidente, orador, dirigente, leitor, estudo)", nomeia)
