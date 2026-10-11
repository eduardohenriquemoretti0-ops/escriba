"""Estudo bíblico (A Sentinela): comentários separados por turno.

O agrupamento de vozes do estudo fica coarse: os comentaristas caem em poucos
clusters. Aqui a separação segue a estrutura da reunião:

1. estrutura(): dirigente (quem mais chama e agradece), leitor (quem fala logo
   depois de "parágrafo"/"leitura" do dirigente) e início do estudo. Só ativa
   com dirigente detectado e MIN_EVENTOS chamadas/agradecimentos posicionais.
2. Turnos: cada trecho contínuo de um mesmo cluster que não é do dirigente nem
   do leitor, entre falas do dirigente. Não usa ABSORB_SECONDS.
3. Nome do turno: chamada do dirigente logo antes ("X, por favor") e/ou
   agradecimento logo depois ("Obrigado, X"). Se os dois discordam, o turno
   fica sem nome (sinal de erro de corte).
4. Mesmo nome = mesma pessoa. Turno sem nome só se junta a uma pessoa com voz
   parecida (cosseno >= LIMIAR_VOZ); senão vira "Comentarista N".

Só sugere: os nomes vão para `nomes_auto` pelo worker e nunca para a memória
de vozes sem confirmação do usuário.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from typing import Callable, Optional

import numpy as np

from . import nomes
from .engines.base import CORO_LABEL, Segment

MIN_EVENTOS = 4            # chamadas/agradecimentos do dirigente para ativar
TURNO_CURTO_SECONDS = 3.0  # turno mais curto que isso não tem voz confiável
LEITURA_MIN_SECONDS = 8.0  # bloco do leitor precisa ser de leitura de verdade
LIMIAR_VOZ = 0.80          # medido no salão: pares de nomes diferentes até ~0,75; mesmo nome a partir de ~0,73 (ver docs)
_ESTUDO = re.compile(r"(?i)sentinela|\bestudo\b")
_LEITURA = re.compile(r"(?i)par[aá]grafo|leitura")
_AGRADECE = re.compile(r"(?i)obrigad")
_ZOOM = re.compile(r"(?i)zoom")
_DIRIGENTE_NOME = re.compile(
    rf"(?i:irm[ãa]o\s+)?(?i:dirigente),?\s+({nomes._NC})")


@dataclass
class Estrutura:
    dirigente: str                 # rótulo cru do dirigente
    leitor: Optional[str]          # rótulo cru do leitor (se detectado)
    inicio: float                  # quando o estudo começa
    eventos: int                   # chamadas + agradecimentos posicionais
    nome_dirigente: Optional[str] = None


@dataclass
class Turno:
    rotulo: str                    # cluster cru (A, B, C...)
    inicio: float
    fim: float
    idx: list[int] = field(default_factory=list)   # índices dos blocos
    chamado: Optional[str] = None
    agradecido: Optional[str] = None
    nome: Optional[str] = None
    origem: str = ""
    discordancia: bool = False
    voz: Optional[np.ndarray] = None
    pessoa: str = ""               # rótulo da pessoa (P1, P2...)
    apos_limite: bool = False      # há fala do dirigente/leitor antes deste turno
    zoom: bool = False             # comentário pelo Zoom (som diferente do salão)

    @property
    def segundos(self) -> float:
        return self.fim - self.inicio


@dataclass
class Pessoa:
    rotulo: str                    # rótulo cru usado nos blocos ("P1"...)
    nome: str                      # nome, ou "Comentarista N"
    papel: str                     # "dirigente", "leitor", "comentarista"
    origem: str                    # de onde veio o nome
    comentarios: int               # nº de turnos (ou blocos, p/ dirigente/leitor)
    segundos: float
    nomeada: bool


# --------------------------------------------------------------------------
# 1) Estrutura da reunião
# --------------------------------------------------------------------------
def estrutura(segs: list[Segment]) -> Optional[Estrutura]:
    """Dirigente, leitor e início do estudo. None se a reunião não tem a estrutura."""
    segs = [s for s in segs if s.speaker != CORO_LABEL]
    eventos = nomes._eventos_posicionais(segs)
    if len(eventos) < MIN_EVENTOS:
        return None
    d = nomes.dirigente_de(segs)
    if d is None:
        return None
    inicio = next((s.start for s in segs if s.speaker == d and _ESTUDO.search(s.text)), None)
    if inicio is None:
        inicio = min(segs[i].start for _, i, _ in eventos if segs[i].speaker == d)
    leitor = None
    for i, s in enumerate(segs):
        if s.start < inicio or s.speaker != d:
            continue
        m = _LEITURA.search(s.text)
        if m is None:
            continue
        # "Muito obrigado, irmã X, pela leitura" agradece; "Vamos iniciar lendo o
        # parágrafo 1" anuncia. Vale só a frase que contém a palavra-chave.
        if _AGRADECE.search(re.split(r"[.!?…]", s.text[:m.start()])[-1]):
            continue
        prox = next((x for x in segs[i + 1:] if x.speaker != d), None)
        if prox is not None and prox.end - prox.start >= LEITURA_MIN_SECONDS:
            leitor = prox.speaker
        break
    nome_d = None
    for s in segs:
        m = _DIRIGENTE_NOME.search(s.text)
        if m:
            nome_d = m.group(1)
            break
    return Estrutura(d, leitor, inicio, len(eventos), nome_d)


# --------------------------------------------------------------------------
# 2) Turnos e nomes
# --------------------------------------------------------------------------
def _mesmo_nome(a: str, b: str) -> bool:
    return SequenceMatcher(None, nomes._fold(a), nomes._fold(b)).ratio() >= nomes.UNIFICA_MIN


def _turnos(segs: list[Segment], est: Estrutura) -> tuple[list[Turno], list[int]]:
    """Turnos de comentário. Interjeição curta (< TURNO_CURTO_SECONDS) que não
    vem depois de fala do dirigente é parte do turno anterior: não tem voz
    própria e, se recebesse o nome, o agradecimento cairia nela, não no turno
    de quem comentou de verdade."""
    idx = [i for i, s in enumerate(segs)
           if s.start >= est.inicio and s.speaker != CORO_LABEL]
    brutos: list[Turno] = []
    cur: Optional[Turno] = None
    limite = False
    for i in idx:
        s = segs[i]
        if s.speaker in (est.dirigente, est.leitor):
            cur = None
            limite = True
            continue
        if cur is None or cur.rotulo != s.speaker:
            cur = Turno(rotulo=s.speaker, inicio=s.start, fim=s.end, apos_limite=limite)
            brutos.append(cur)
            limite = False
        cur.idx.append(i)
        cur.fim = s.end
    turnos: list[Turno] = []
    for t in brutos:
        if turnos and not t.apos_limite and t.segundos < TURNO_CURTO_SECONDS:
            turnos[-1].idx += t.idx
            turnos[-1].fim = t.fim
            continue
        turnos.append(t)
    return turnos, idx


def _nomeia(turnos: list[Turno], segs: list[Segment], idx: list[int],
            est: Estrutura) -> None:
    pos = {i: k for k, i in enumerate(idx)}
    for t in turnos:
        k = pos[t.idx[0]]
        antes = segs[idx[k - 1]] if k > 0 else None
        j = pos[t.idx[-1]] + 1
        depois = segs[idx[j]] if j < len(idx) else None
        if antes is not None and antes.speaker == est.dirigente:
            chamadas = [n for tp, n in nomes.eventos_do_texto(antes.text) if tp == "chamada"]
            t.chamado = chamadas[0] if chamadas else None   # a 1ª chamada é a do turno seguinte
        if depois is not None and depois.speaker == est.dirigente:
            agr = [n for tp, n in nomes.eventos_do_texto(depois.text) if tp == "agradecimento"]
            t.agradecido = agr[0] if agr else None
        c, a = t.chamado, t.agradecido
        if antes is not None and antes.speaker == est.dirigente and _ZOOM.search(antes.text):
            t.zoom = True
        if c and a and not _mesmo_nome(c, a):
            # a chamada costuma ser mais clara e repetida: ela vence; o conflito
            # fica registrado na origem
            t.discordancia = True
            t.nome = c
            t.origem = f"chamado pelo dirigente; agradecimento disse {a}"
        elif c or a:
            t.nome = c or a
            if c and a:
                t.origem = "chamado e agradecido pelo dirigente"
            else:
                t.origem = "chamado pelo dirigente" if c else "agradecido pelo dirigente"
        if t.zoom and t.origem:
            t.origem += " (pelo Zoom)"


def _centroide(vetores: list[np.ndarray]) -> np.ndarray:
    v = np.mean(vetores, axis=0)
    return v / (np.linalg.norm(v) + 1e-9)


def medir_separacao(turnos: list[Turno]) -> tuple[list[float], list[float]]:
    """Cosseno entre turnos com nome e voz: (mesmo nome, nomes diferentes).

    Usado para calibrar LIMIAR_VOZ no job real. Só pares com voz."""
    mesmo, diferente = [], []
    com_voz = [t for t in turnos if t.nome and t.voz is not None]
    for i, a in enumerate(com_voz):
        for b in com_voz[i + 1:]:
            s = float(a.voz @ b.voz)
            (mesmo if a.nome == b.nome else diferente).append(s)
    return mesmo, diferente


def reatribui(segs: list[Segment],
              embed: Callable[[float, float], np.ndarray],
              limiar: float = LIMIAR_VOZ) -> tuple[list[Segment], list[Pessoa], dict]:
    """Troca o rótulo dos blocos do estudo pela pessoa de cada turno.

    `embed(inicio, fim)` devolve a voz (vetor normalizado) de um trecho.
    Devolve (blocos, pessoas, resumo). Sem estrutura: devolve tudo igual.
    """
    est = estrutura(segs)
    if est is None:
        return segs, [], {"ativo": False}

    turnos, idx = _turnos(segs, est)
    _nomeia(turnos, segs, idx, est)

    # nomes canônicos (Admir/Ademir viram um só)
    nomes_brutos = [t.nome for t in turnos if t.nome]
    mapa = nomes._unifica(nomes_brutos) if nomes_brutos else {}
    for t in turnos:
        if t.nome:
            t.nome = mapa.get(t.nome, t.nome)

    for t in turnos:
        if t.segundos >= TURNO_CURTO_SECONDS:
            t.voz = embed(t.inicio, t.fim)

    # pessoas com nome: uma por nome canônico
    pessoas: dict[str, dict] = {}
    for t in turnos:
        if t.nome:
            p = pessoas.setdefault(t.nome, {"turnos": [], "origens": []})
            p["turnos"].append(t)
            p["origens"].append(t.origem)
            t.pessoa = f"nome:{t.nome}"

    # turnos sem nome: junta à pessoa de voz parecida; senão, Comentarista N
    grupos: dict[str, list[np.ndarray]] = {}
    zoom_do_grupo: dict[str, bool] = {}
    for n, p in pessoas.items():
        vozes = [t.voz for t in p["turnos"] if t.voz is not None]
        if vozes:
            grupos[f"nome:{n}"] = vozes
            zoom_do_grupo[f"nome:{n}"] = any(t.zoom for t in p["turnos"])
    comentaristas = 0
    anterior: Optional[str] = None
    for t in turnos:
        if t.nome:
            anterior = t.pessoa
            continue
        if t.voz is None:
            if anterior is None:
                comentaristas += 1
                anterior = f"novo:{comentaristas}"
            t.pessoa = anterior
            continue
        melhor, sim = None, -1.0
        for chave, vozes in grupos.items():
            if zoom_do_grupo.get(chave, False) != t.zoom:
                continue                       # Zoom e salão não se fundem
            s = float(_centroide(vozes) @ t.voz)
            if s > sim:
                melhor, sim = chave, s
        if melhor is not None and sim >= limiar:
            t.pessoa = melhor
            grupos[melhor].append(t.voz)
        else:
            comentaristas += 1
            t.pessoa = f"novo:{comentaristas}"
            grupos[t.pessoa] = [t.voz]
            zoom_do_grupo[t.pessoa] = t.zoom
        anterior = t.pessoa

    # rótulos crus: P1, P2... por ordem de aparição
    ordem: dict[str, str] = {}
    for t in turnos:
        if t.pessoa not in ordem:
            ordem[t.pessoa] = f"P{len(ordem) + 1}"

    out = list(segs)
    for t in turnos:
        for i in t.idx:
            out[i] = replace(segs[i], speaker=ordem[t.pessoa])

    resumo_pessoas = _pessoas(turnos, segs, idx, est, ordem, pessoas)
    stats = {
        "ativo": True,
        "dirigente": est.dirigente,
        "leitor": est.leitor,
        "inicio": est.inicio,
        "turnos": len(turnos),
        "turnos_com_nome": sum(1 for t in turnos if t.nome),
        "discordancias": sum(1 for t in turnos if t.discordancia),
        "comentaristas": sum(1 for p in resumo_pessoas if p.papel == "comentarista"),
        "limiar_voz": limiar,
    }
    return out, resumo_pessoas, stats


def _pessoas(turnos: list[Turno], segs: list[Segment], idx: list[int],
             est: Estrutura, ordem: dict[str, str],
             nomeados: dict) -> list[Pessoa]:
    res: list[Pessoa] = []
    # dirigente
    seg_d = [segs[i] for i in idx if segs[i].speaker == est.dirigente]
    res.append(Pessoa(est.dirigente, est.nome_dirigente or "Dirigente",
                      "dirigente",
                      "anúncio no texto" if est.nome_dirigente else "",
                      len(seg_d), round(sum(s.end - s.start for s in seg_d), 1),
                      bool(est.nome_dirigente)))
    # leitor: nome da chamada que antecede o primeiro bloco de leitura
    if est.leitor is not None:
        seg_l = [segs[i] for i in idx if segs[i].speaker == est.leitor]
        nome_l = None
        k = next((k for k, i in enumerate(idx) if segs[i].speaker == est.leitor), None)
        if k is not None and k > 0:
            antes = segs[idx[k - 1]]
            if antes.speaker == est.dirigente:
                ch = [n for tp, n in nomes.eventos_do_texto(antes.text) if tp == "chamada"]
                nome_l = ch[-1] if ch else None
        res.append(Pessoa(est.leitor, nome_l or "Leitor", "leitor",
                          "chamado pelo dirigente antes da leitura" if nome_l else "",
                          len(seg_l), round(sum(s.end - s.start for s in seg_l), 1),
                          bool(nome_l)))
    # comentaristas (nome ou Comentarista N)
    por_pessoa: dict[str, list[Turno]] = {}
    for t in turnos:
        por_pessoa.setdefault(t.pessoa, []).append(t)
    for chave, ts in por_pessoa.items():
        nome = chave.split(":", 1)[1] if chave.startswith("nome:") else None
        if nome is None:
            nome = f"Comentarista {chave.split(':', 1)[1]}"
        origens: dict[str, int] = {}
        for t in ts:
            if t.origem:
                origens[t.origem] = origens.get(t.origem, 0) + 1
        origem = ", ".join(f"{o} {n}×" for o, n in origens.items())
        res.append(Pessoa(ordem[chave], nome, "comentarista", origem, len(ts),
                          round(sum(t.segundos for t in ts), 1), chave.startswith("nome:")))
    return res
