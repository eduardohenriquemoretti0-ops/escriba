"""Papéis da reunião de fim de semana (discurso público + A Sentinela).

A estrutura é fixa e vale para TODA reunião de fim de semana:

* o presidente abre e ANUNCIA os nomes — "orador irmão X", "dirigente Y",
  "leitor Z";
* o ORADOR faz o discurso público: um bloco longo e contínuo de um só falante,
  antes de A Sentinela começar;
* em A Sentinela o DIRIGENTE dá o número do parágrafo/pergunta ("vamos ao 12",
  "parágrafo 9", "pergunta 2") e o LEITOR (sempre outra pessoa) lê o parágrafo
  logo em seguida.

O dirigente e o leitor mudam a cada semana, então os CLUSTERS saem da estrutura
(quem dá o número / quem lê em seguida) e os NOMES saem do anúncio do presidente.
Regra do Eduardo: estrutura manda, voz confirma; na dúvida, "[?]". Só sugere —
nada vai para a memória de vozes sem o usuário confirmar.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional

from . import irmaos as _irmaos
from . import nomes
from .engines.base import CORO_LABEL, Segment

# Anúncio de parágrafo/leitura do dirigente: "parágrafo 9", "vamos ao 12",
# "pergunta 2", "16." (número isolado). Mesmo padrão provado em estudo.py.
_CUE = re.compile(
    r"(?i)par[aá]grafo|pergunta\s*\d|leitura|\b(?:vamos|prosseguir|seguir|ir)\b"
    r"[^.?!]*\b\d+|^\s*\d+\s*[º°]?\s*[.!]?\s*$"
)
# Nomes anunciados pelo presidente. O nome vem logo depois do papel.
_ORADOR = re.compile(rf"(?i:irm[ãa]o\s+orador|\borador\b)\s*,?\s*(?:irm[ãa]o\s+)?({nomes._NC})")
_ORADOR2 = re.compile(rf"(?i:discurso)[^.!?]*?(?i:irm[ãa]o\s+)({nomes._NC})")
_DIRIGENTE = re.compile(rf"(?i:dirigente)\s*,?\s*(?:irm[ãa]o\s+)?({nomes._NC})")
_LEITOR = re.compile(rf"(?i:\bleitor\b)\s*,?\s*(?:irm[ãa]o\s+)?({nomes._NC})")

CUE_MAX_PALAVRAS = 30       # a dica do dirigente é curta; a leitura é longa
CUE_MAX_SEGUNDOS = 30.0
LEITURA_MIN_SEGUNDOS = 15.0  # o leitor lê um bloco longo
LEITURA_JANELA = 60.0        # o leitor lê até isso depois da dica
ORADOR_MIN_SEGUNDOS = 300.0  # discurso público: pelo menos 5 min de um falante


def _fold(s: str) -> str:
    return nomes._fold(s)


def _resolve_lista(cru: str, entradas: list, parts: list[str]
                   ) -> tuple[str, bool]:
    """(nome, na_lista) do nome anunciado, resolvido pela lista de irmãos.

    Primeiro tenta a grafia exata da lista/participantes. Senão, um nome com
    2+ tokens casa por PREFIXO único ("José Eduardo" -> "José Eduardo Silveira").
    Um nome de 1 token (só o primeiro) é fraco demais: fica em dúvida mesmo que
    haja um único irmão com esse primeiro nome, porque o leitor pode nem estar
    na lista (regra do Eduardo: na dúvida, "[?]")."""
    c = _irmaos.corrige_participante(cru, entradas)
    if c:
        return c, True
    if nomes._participante_de(cru, parts) is not None:
        return cru, True
    toks = _fold(cru).split()
    if len(toks) >= 2:
        cand = [e.nome for e in entradas
                if _fold(e.nome).split()[:len(toks)] == toks]
        if len(cand) == 1:
            return cand[0], True
    return cru, False


@dataclass
class Papel:
    cluster: str              # rótulo cru do falante
    papel: str                # "orador" | "dirigente" | "leitor"
    nome: Optional[str]       # nome anunciado (grafia da lista quando casa)
    na_lista: bool            # o nome casa com a lista/participantes?
    detalhe: str


def _nome_anunciado(segs: list[Segment], pres: Optional[str]) -> dict[str, str]:
    """Nomes crus que o presidente anunciou para cada papel.

    Lê só as falas do presidente (ou, sem presidente, as primeiras falas) e
    pega o nome logo depois de "orador"/"dirigente"/"leitor"."""
    texto = " ".join(
        s.text for s in segs
        if (pres and s.speaker == pres) or (not pres)
    )
    if pres is None:                       # sem presidente: só a abertura
        texto = " ".join(s.text for s in segs[:12])
    achados: dict[str, str] = {}
    for papel, rx in (("orador", _ORADOR), ("orador", _ORADOR2),
                      ("dirigente", _DIRIGENTE), ("leitor", _LEITOR)):
        if papel in achados:
            continue
        m = rx.search(texto)
        if m:
            nome = nomes._limpa_nome(m.group(1))
            if nome:
                achados[papel] = nome
    return achados


def _curto(seg: Segment) -> bool:
    return (len(seg.text.split()) <= CUE_MAX_PALAVRAS
            and (seg.end - seg.start) <= CUE_MAX_SEGUNDOS)


def _inicio_estudo(segs: list[Segment]) -> Optional[float]:
    """Começo de A Sentinela: a primeira dica curta de parágrafo/pergunta."""
    for s in segs:
        if _CUE.search(s.text) and _curto(s):
            return s.start
    return None


def _orador(segs: list[Segment], pres: Optional[str],
            estudo_ini: Optional[float]) -> Optional[str]:
    """Cluster do discurso público: maior tempo de fala antes do estudo,
    sem ser o presidente."""
    teto = estudo_ini if estudo_ini is not None else float("inf")
    dur: dict[str, float] = defaultdict(float)
    for s in segs:
        if s.start >= teto or s.speaker == pres:
            continue
        dur[s.speaker] += s.end - s.start
    if not dur:
        return None
    spk, tot = max(dur.items(), key=lambda kv: kv[1])
    return spk if tot >= ORADOR_MIN_SEGUNDOS else None


def _dirigente_leitor(segs: list[Segment], estudo_ini: Optional[float]
                      ) -> tuple[Optional[str], Optional[str]]:
    """(dirigente, leitor) pela estrutura do estudo.

    Dirigente = quem mais dá a dica curta de parágrafo/pergunta. Leitor = quem
    lê o bloco longo logo depois da dica (e quem mais lê "parágrafo N: ...")."""
    piso = estudo_ini if estudo_ini is not None else 0.0
    estudo = [s for s in segs if s.start >= piso]
    dicas: list[int] = []
    cue_por_falante: Counter = Counter()
    for i, s in enumerate(estudo):
        if _CUE.search(s.text) and _curto(s):
            dicas.append(i)
            cue_por_falante[s.speaker] += 1
    if not cue_por_falante:
        return None, None
    dirigente, n = cue_por_falante.most_common(1)[0]
    if n < 2:
        return None, None

    leitor_votos: Counter = Counter()
    # (a) quem lê logo depois de uma dica do dirigente
    for i in dicas:
        if estudo[i].speaker != dirigente:
            continue
        for j in range(i + 1, len(estudo)):
            if estudo[j].start - estudo[i].end > LEITURA_JANELA:
                break
            if estudo[j].speaker != dirigente:
                if (estudo[j].end - estudo[j].start) >= LEITURA_MIN_SEGUNDOS:
                    leitor_votos[estudo[j].speaker] += 1
                break
    # (b) quem lê blocos longos começando por "parágrafo N:" / "vamos ao N:"
    for s in estudo:
        if (s.speaker != dirigente
                and (s.end - s.start) >= LEITURA_MIN_SEGUNDOS
                and _CUE.search(s.text[:40])):
            leitor_votos[s.speaker] += 1
    leitor = leitor_votos.most_common(1)[0][0] if leitor_votos else None
    return dirigente, leitor


# Fala típica do dirigente entre comentários: agradece/chama/devolve a palavra.
_RETORNO_DIR = re.compile(
    r"(?i)^\W*(isso|muito bem|muito bom|exatamente|perfeito|correto|t[aá] bom|"
    r"bem lembrad|com certeza|boa|verdade|certo|[óo]timo|pois (?:é|e)|amém|amem)\b")
_PEDE_PALAVRA = re.compile(
    r"(?i)por favor|por gentileza|pode falar|quem (?:gostaria|quer)|"
    r"vamos (?:pedir|ouvir|para)|algu[ée]m (?:mais|gostaria)|mais algu")


def _e_fala_dirigente(texto: str) -> bool:
    """O texto é fala de quem conduz: agradece, chama, devolve ou pergunta curto."""
    t = texto.strip()
    if not t:
        return False
    if _RETORNO_DIR.search(t) or _PEDE_PALAVRA.search(t):
        return True
    if nomes.eventos_do_texto(t):          # chamada/agradecimento com nome
        return True
    # pergunta curta dirigida ("O que que é?", "Qual é, irmão Antônio?")
    return t.endswith("?") and len(t.split()) <= 12


def funde_dirigente(segments: list[Segment], frac: float = 0.6
                    ) -> list[Segment]:
    """Devolve os segmentos com o backchannel do dirigente de volta nele.

    Padrão de toda reunião com estudo: o dirigente fala algo depois de cada
    comentário (agradece, confirma) e chama o próximo irmão. O diarizador às
    vezes joga esses turnos curtos do dirigente num cluster à parte. Aqui, um
    cluster (que não é dirigente/leitor/presidente) cujas falas são, na maioria
    (>= `frac`), fala de quem conduz, volta para o cluster do dirigente. Os
    comentaristas de verdade (respostas com conteúdo) ficam onde estão.
    """
    segs = [s for s in segments if s.text.strip() and s.speaker != CORO_LABEL]
    if not segs:
        return segments
    estudo_ini = _inicio_estudo(segs)
    if estudo_ini is None:                 # sem estudo, não há esse padrão
        return segments
    pres = nomes.presidente_reuniao(segs)
    dirigente, leitor = _dirigente_leitor(segs, estudo_ini)
    if not dirigente:
        return segments
    porcluster: dict[str, list[Segment]] = defaultdict(list)
    for s in segs:
        porcluster[s.speaker].append(s)
    alvo: set[str] = set()
    for cl, ss in porcluster.items():
        if cl in (dirigente, leitor, pres) or len(ss) < 2:
            continue
        # só clusters do ESTUDO: a abertura e o discurso (antes do estudo) têm
        # fala parecida com a do dirigente (anúncio, agradecimento) e NÃO podem
        # ser fundidos — senão o presidente/orador some dentro do dirigente.
        no_estudo = sum(1 for s in ss if s.start >= estudo_ini)
        if no_estudo / len(ss) < 0.7:
            continue
        dir_falas = sum(1 for s in ss if _e_fala_dirigente(s.text))
        if dir_falas / len(ss) >= frac:
            alvo.add(cl)
    if not alvo:
        return segments
    from dataclasses import replace
    return [replace(s, speaker=dirigente) if s.speaker in alvo else s
            for s in segments]


def papeis(segments: list[Segment], entradas: Iterable = (),
           participantes: Iterable[str] = ()) -> dict[str, Papel]:
    """{rótulo_cru: Papel} para orador, dirigente e leitor do fim de semana.

    Os clusters saem da estrutura; os nomes saem do anúncio do presidente. Um
    nome que não casa com a lista nem com os participantes fica como sugestão
    (`na_lista=False`): o usuário confirma na revisão."""
    segs = [s for s in segments if s.text.strip() and s.speaker != CORO_LABEL]
    if not segs:
        return {}
    entradas = list(entradas)
    parts = [p for p in participantes if p and p.strip()]

    pres = nomes.presidente_reuniao(segs)
    estudo_ini = _inicio_estudo(segs)
    anunciado = _nome_anunciado(segs, pres)

    clusters = {
        "orador": _orador(segs, pres, estudo_ini),
        "dirigente": None,
        "leitor": None,
    }
    clusters["dirigente"], clusters["leitor"] = _dirigente_leitor(segs, estudo_ini)

    out: dict[str, Papel] = {}
    for papel, cluster in clusters.items():
        if not cluster or cluster == pres or cluster in out:
            continue
        cru = anunciado.get(papel)
        if cru:
            nome, na_lista = _resolve_lista(cru, entradas, parts)
            detalhe = {
                "orador": "faz o discurso público (anunciado pelo presidente)",
                "dirigente": "dá o número do parágrafo (anunciado pelo presidente)",
                "leitor": "lê o parágrafo depois da dica do dirigente "
                          "(anunciado pelo presidente)",
            }[papel]
            out[cluster] = Papel(cluster, papel, nome, bool(na_lista), detalhe)
        else:
            # cluster do papel achado, mas o presidente não anunciou o nome:
            # na dúvida, "[?]" com o papel — nunca um nome inventado.
            detalhe = {
                "orador": "faz o discurso público · sem nome no anúncio",
                "dirigente": "dá o número do parágrafo · sem nome no anúncio",
                "leitor": "lê o parágrafo após a dica · sem nome no anúncio",
            }[papel]
            out[cluster] = Papel(cluster, papel, None, False, detalhe)
    return out
