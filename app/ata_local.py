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
from typing import Optional

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


def _comentaristas_dos_segmentos(job: jobs.Job, segs: list,
                                 presidente: Optional[str]
                                 ) -> list[tuple[str, int, float]]:
    """(nome, nº de falas, início da 1ª fala) por falante, direto dos segmentos.

    Para motores sem análise de estudo (AssemblyAI): agrupa por falante, usa o
    nome já resolvido (confirmado na revisão ou sugestão), e marca o tempo da
    1ª fala. Fora: quem preside (já consta em "Quem conduziu")."""
    nomes = review.names_for(job, segs)              # rótulo cru -> nome exibido
    default = exports.speaker_map(segs)              # rótulo cru -> "Falante N"
    por_falante: dict[str, dict] = {}
    for s in segs:
        row = por_falante.setdefault(s.speaker, {"n": 0, "ini": s.start})
        row["n"] += 1
    saida: list[tuple[str, int, float]] = []
    for raw, row in por_falante.items():
        nome = nomes.get(raw) or default.get(raw, raw)
        if presidente and nome == presidente:
            continue                                 # já está em "Quem conduziu"
        if nome == default.get(raw):                 # sem nome: marca para ouvir
            nome = f"[?] sem nome (ouça) — {nome}"
        saida.append((nome, row["n"], row["ini"]))
    saida.sort(key=lambda t: t[2])
    return saida


def _gera_do_motor(job: jobs.Job, atrib: dict) -> str:
    """Ata a partir do motor de estrutura: papéis vêm de `blocos`, comentaristas
    das falas com papel 'comentarista' (agrupadas por quem, com o tempo da 1ª),
    e as pendências viram a seção "Ouvir e confirmar" (tempo + motivo)."""
    blocos = atrib.get("blocos", [])
    falas = atrib.get("falas", [])
    pend = atrib.get("pendencias", [])

    def por_papel(papel: str) -> Optional[str]:
        return next((b["quem"] for b in blocos if b.get("papel") == papel and b.get("quem")), None)

    presidente = por_papel("presidente")
    orador = por_papel("orador")
    dirigente = por_papel("dirigente")
    leitor = por_papel("leitor")
    papeis_nomes = {n for n in (presidente, orador, dirigente, leitor) if n}

    titulo = review.default_title(job)
    linhas = [f"# Ata da reunião — {titulo}", "",
              f"- **Data:** {_data(job)}",
              f"- **Duração:** {exports.hms(job.duration) if job.duration else 'não informada'}",
              f"- **Arquivo:** {job.filename}", "",
              "## Quem conduziu", "",
              f"- **Presidente:** {presidente or 'não identificado (ouça e confirme)'}",
              f"- **Dirigente de A Sentinela:** {dirigente or 'não identificado'}",
              f"- **Leitor de A Sentinela:** {leitor or 'não identificado'}", ""]

    if orador:
        linhas += ["## Discurso público", "", f"- **Orador:** {orador}"]
        disc = next((b for b in blocos if b.get("papel") == "orador"), None)
        if disc:
            linhas.append(f"- **Trecho:** {exports.hms(disc['ini'])} a {exports.hms(disc['fim'])}")
        linhas.append("")

    linhas += ["## Cânticos citados", ""]
    cantos = canticos([s.text for s in review.segments_for(job)])
    linhas += [f"- {c}" for c in cantos] or ["- Nenhum cântico com número na transcrição."]
    linhas.append("")

    # comentaristas: falas com papel 'comentarista', por pessoa, tempo da 1ª fala.
    # presidente/orador/dirigente/leitor NÃO entram aqui.
    com: dict[str, list] = {}
    for f in falas:
        if f.get("papel") != "comentarista":
            continue
        nome = f.get("quem") or "[?] sem nome (ouça)"
        if nome in papeis_nomes:
            continue
        row = com.setdefault(nome, [0, f.get("ini", 0.0)])
        row[0] += 1
    linhas += ["## Comentaristas", ""]
    rows = []
    for nome, (n, ini) in sorted(com.items(), key=lambda kv: kv[1][1]):
        unidade = "comentário" if n == 1 else "comentários"
        rows.append(f"- {nome}: {n} {unidade} (em {exports.hms(ini)})")
    linhas += rows or ["- Nenhum comentário identificado."]
    linhas.append("")

    linhas += ["## Ouvir e confirmar", ""]
    pend_rows = [f"- {p['inicio']}–{p['fim']} · {p.get('nivel', 'conferir')} · "
                 f"**{p['quem']}** ({p.get('papel', '')}) — {p.get('motivo', '')}"
                 for p in pend]
    linhas += pend_rows or ["- Nada a confirmar."]
    linhas.append("")

    participantes = irmaos.corrige_participantes(job.participants or [], irmaos.listar_seguro())
    linhas += ["## Participantes informados", ""]
    linhas += [f"- {p}" for p in participantes] or ["- Nenhum participante informado."]
    linhas += ["", "---", "",
               "_Ata gerada localmente, sem IA, a partir do motor de estrutura: traz "
               "a estrutura da reunião e não resume os comentários. Os nomes são "
               "sugestões; confirme na revisão e nos trechos de 'Ouvir e confirmar'._", ""]
    return "\n".join(linhas)


def gera(job: jobs.Job) -> str:
    """Markdown da ata local. Não chama API nem modelo."""
    atrib = review.load_atribuicao(job)
    if atrib is not None:
        return _gera_do_motor(job, atrib)
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
    if not presidente:
        # reunião de meio de semana (sem discurso público): o presidente vem
        # da sugestão do motor (fonte "presidente" — abre e anuncia as partes)
        presidente = next((i.get("nome") for i in (job.nomes_auto or {}).values()
                           if i.get("fonte") == "presidente"), None)
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
                   f"- **Tema:** {review.corrige_texto(p1.get('tema') or '', job) or 'não identificado'}"]
        if p1.get("orador_inicio") is not None and p1.get("orador_fim") is not None:
            linhas.append(f"- **Trecho:** {exports.hms(p1['orador_inicio'])} a "
                          f"{exports.hms(p1['orador_fim'])}")
        linhas.append("")

    linhas += ["## Cânticos citados", ""]
    cantos = canticos([s.text for s in segs])
    linhas += [f"- {c}" for c in cantos] or ["- Nenhum cântico com número na transcrição."]
    linhas.append("")

    # início da 1ª fala de cada rótulo: todo comentário leva o tempo, para o
    # Eduardo achar o trecho no áudio (regra: "marque o tempo dos comentários").
    ini_rotulo: dict[str, float] = {}
    for s in segs:
        if s.speaker not in ini_rotulo:
            ini_rotulo[s.speaker] = s.start

    def _tempo(p: dict) -> str:
        t = ini_rotulo.get(p.get("rotulo"))
        return f" (em {exports.hms(t)})" if t is not None else ""

    linhas += ["## Comentaristas", ""]
    rows: list[str] = []
    if comentaristas:
        for p in comentaristas:
            n = int(p.get("comentarios") or 0)
            unidade = "comentário" if n == 1 else "comentários"
            nome = "[?] sem nome (ouça)" if p.get("duvida") else p.get("nome")
            rows.append(f"- {nome}: {n} {unidade}{_tempo(p)}")
    else:
        # Motor sem análise de estudo (ex.: AssemblyAI): a lista sai direto dos
        # segmentos — cada falante com nº de comentários e o tempo da 1ª fala,
        # em ordem de aparição. Quem preside não entra (já está em "Quem conduziu").
        for nome, n, ini in _comentaristas_dos_segmentos(job, segs, presidente):
            unidade = "comentário" if n == 1 else "comentários"
            rows.append(f"- {nome}: {n} {unidade} (em {exports.hms(ini)})")
    linhas += rows or ["- Nenhum comentário identificado."]
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
