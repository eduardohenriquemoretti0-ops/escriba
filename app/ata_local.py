"""Ata local: a estrutura da reunião em Markdown, sem API e sem IA.

Usada quando não há ANTHROPIC_API_KEY. Não resume o conteúdo dos comentários.
Traz o que a estrutura já identificou: data e duração, presidente, orador e
tema, dirigente e leitor de A Sentinela, cânticos citados com número, os
comentaristas com o nº de comentários e os participantes informados. Os nomes
são sugestões do motor: a revisão é que confirma.
"""
from __future__ import annotations

import re
from datetime import datetime

from . import exports, irmaos, jobs, review

# "Cântico de número 56. Faça da verdade a sua vida" (aceita "Cântica", "nº 56")
# aceita as formas da fala: "cântico de número 56", "o cântico é de número 56",
# "o cântico é o número 150" (até duas palavras de ligação entre as partes)
_CANTICO = re.compile(
    r"(?i)c[âa]ntic[oa]\s+(?:(?:é|e|de|do|da|o|a|no|na)\s+){0,2}"
    r"(?:n[úu]mero|n[º°o]\.?)\s*(\d{1,3})\b[.,:]?"
    r"\s*(?:que tem por tema\s*,?\s*)?([^.!?\n]{0,100})")
_DATA_ARQUIVO = re.compile(r"(\d{1,2})[-./](\d{1,2})[-./](\d{4})")


def _data(job: jobs.Job) -> str:
    informada = (job.meeting or {}).get("date", "").strip()
    if informada:
        return informada
    m = _DATA_ARQUIVO.search(job.filename or "")
    if m:
        return f"{int(m.group(1)):02d}/{int(m.group(2)):02d}/{m.group(3)}"
    return datetime.fromtimestamp(job.created_at).strftime("%d/%m/%Y")


def canticos(texto_segmentos: list[str]) -> list[str]:
    """Cânticos citados com número, sem repetir o mesmo número."""
    achados: dict[int, str] = {}
    for texto in texto_segmentos:
        for m in _CANTICO.finditer(texto):
            numero = int(m.group(1))
            titulo = re.sub(r"\s+", " ", m.group(2))
            titulo = re.split(r",?\s*tem como base", titulo, flags=re.IGNORECASE)[0]
            titulo = titulo.strip(" ,;")
            titulo = titulo[:1].upper() + titulo[1:]
            if numero not in achados or (not achados[numero] and titulo):
                achados[numero] = titulo
    return [f"Cântico de número {n}" + (f". {t}" if t else "")
            for n, t in sorted(achados.items())]


def _pessoa_nomeada(p: dict) -> str:
    return p.get("nome") if p.get("nomeada") else "não identificado"


def gera(job: jobs.Job) -> str:
    """Markdown da ata local. Não chama API nem modelo."""
    segs = review.segments_for(job)
    info = job.estudo_info or {}
    p1 = info.get("parte1") or {}
    pessoas = job.falantes_estudo or []

    dirigente = next((p for p in pessoas if p.get("papel") == "dirigente"
                      and not str(p.get("nome", "")).startswith("Co-dirigente")), None)
    leitor = next((p for p in pessoas if p.get("papel") == "leitor"), None)
    comentaristas = [p for p in pessoas if "comentarista" in p.get("papel", "")]

    titulo = review.default_title(job)
    linhas = [f"# Ata da reunião — {titulo}", "",
              f"- **Data:** {_data(job)}",
              f"- **Duração:** {exports.hms(job.duration) if job.duration else 'não informada'}",
              f"- **Arquivo:** {job.filename}", "",
              "## Quem conduziu", ""]
    presidente = p1.get("presidente")
    linhas.append(f"- **Presidente:** {presidente or 'não identificado (sem nome na lista de irmãos)'}")
    linhas.append("- **Dirigente de A Sentinela:** "
                  + (_pessoa_nomeada(dirigente) if dirigente else "não identificado"))
    linhas.append("- **Leitor de A Sentinela:** "
                  + (_pessoa_nomeada(leitor) if leitor else "não identificado"))
    linhas.append("")

    if p1:
        orador = p1.get("orador") or "não identificado"
        if p1.get("orador_confianca") not in (None, "", "alta"):
            orador += " (nome pela grafia do anúncio: confira na revisão)"
        linhas += ["## Discurso público", "",
                   f"- **Orador:** {orador}",
                   f"- **Tema:** {p1.get('tema') or 'não identificado'}"]
        if p1.get("orador_inicio") is not None and p1.get("orador_fim") is not None:
            linhas.append(f"- **Trecho:** {exports.hms(p1['orador_inicio'])} a "
                          f"{exports.hms(p1['orador_fim'])}")
        linhas.append("")

    linhas += ["## Cânticos citados", ""]
    cantos = canticos([s.text for s in segs])
    linhas += [f"- {c}" for c in cantos] or ["- Nenhum cântico com número na transcrição."]
    linhas.append("")

    linhas += ["## Comentaristas", ""]
    duvidas = 0
    for p in comentaristas:
        n = int(p.get("comentarios") or 0)
        if p.get("duvida"):
            duvidas += n          # sem evidência de nome: um total só, no fim
            continue
        unidade = "comentário" if n == 1 else "comentários"
        linhas.append(f"- {p.get('nome')}: {n} {unidade}")
    if duvidas:
        unidade = "trecho" if duvidas == 1 else "trechos"
        linhas.append(f"- [?] sem nome confirmado: {duvidas} {unidade} "
                      "(confira na revisão)")
    if not comentaristas:
        linhas.append("- Nenhum comentário identificado.")
    linhas.append("")

    participantes = irmaos.corrige_participantes(job.participants or [],
                                                 irmaos.listar_seguro())
    linhas += ["## Participantes informados", ""]
    linhas += [f"- {p}" for p in participantes] or ["- Nenhum participante informado."]
    linhas += ["", "---", "",
               "_Ata gerada localmente, sem IA: traz a estrutura da reunião e não "
               "resume o conteúdo dos comentários. Os nomes são sugestões, confira "
               "na revisão._", ""]
    return "\n".join(linhas)
