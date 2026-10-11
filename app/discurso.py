"""Parte 1 da reunião (discurso público): presidente, orador e fragmentos.

Numa reunião de congregação o presidente abre, anuncia o orador ("irmão
orador X", "temos um discurso com o seguinte tema ...") e agradece no fim
("Muito obrigado irmão X, excelente ..."). Presidente e orador costumam usar
o mesmo microfone, então a voz não separa os dois. Aqui a separação segue a
estrutura: antes do anúncio e depois do agradecimento é o presidente; entre o
anúncio e o agradecimento é o orador.

Também absorve fragmentos curtos (até 2 palavras ou menos de 1,5 s) que caem
no meio de um bloco longo do mesmo falante: "É", "Jeová?", "Ele é" atribuídos
a outro cluster por engano. Vale para a reunião inteira.

Só sugere: os nomes saem como sugestão (`nomes_auto`). Nunca vão para a
memória de vozes sem confirmação do usuário.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from typing import Optional, Sequence

from . import irmaos as _irmaos
from . import nomes
from .engines.base import CORO_LABEL, Segment

FRAGMENTO_PALAVRAS = 2          # até isso: fragmento
FRAGMENTO_SECONDS = 1.5         # ou menos que isso
BLOCO_LONGO_SECONDS = 8.0       # o bloco vizinho precisa ter pelo menos isso
PRESIDENTE_APOS_SECONDS = 180.0  # sem estudo: presidente fala até 3 min após o agradecimento

# "irmão orador X" / "discurso ... irmão X" (o nome vem logo depois)
_ANUNCIO = re.compile(r"(?i)\b(?:irm[ãa]o\s+orador|discurso)\b")
_NOME_ORADOR = re.compile(rf"(?i:irm[ãa]o\s+orador\s*,?\s*)({nomes._NC})")
_NOME_IRMAO = re.compile(rf"(?i:irm[ãa]o\s+)({nomes._NC})")
_TEMA = re.compile(r"(?i)seguinte tema\s*,?\s*([^.!?]+)")
_AGRADECE = re.compile(rf"(?i:obrigad[oa])\s*,?\s*(?i:irm[ãa]o\s+)({nomes._NC})")
# "damos boas-vindas ao irmão X": o anúncio do discurso costuma não repetir o nome
_BOAS_VINDAS = re.compile(rf"(?i:boas[- ]vindas)[^.!?]*?(?i:irm[ãa]o\s+)({nomes._NC})")
NOME_ANTERIOR_SECONDS = 400.0   # até quanto antes do anúncio o nome ainda vale


@dataclass
class Parte1:
    presidente: Optional[str]          # da lista de irmãos (observação "presidente")
    presidente_cluster: str            # rótulo cru de quem preside
    orador: str                        # nome do orador (pode ser sugestão baixa)
    orador_cru: str                    # como o anúncio escreveu
    orador_confianca: str              # alta (lista) | media (participantes) | baixa
    tema: str
    anuncio_inicio: float
    agradecimento_inicio: Optional[float]
    orador_cluster: str
    orador_inicio: float
    orador_fim: float
    indices_presidente: list[int] = field(default_factory=list)
    indices_orador: list[int] = field(default_factory=list)


def _dur(s: Segment) -> float:
    return s.end - s.start


def _fragmento(s: Segment) -> bool:
    return len(s.text.split()) <= FRAGMENTO_PALAVRAS or _dur(s) < FRAGMENTO_SECONDS


def absorve_fragmentos(segs: list[Segment],
                       janela: Optional[tuple[float, float]] = None
                       ) -> tuple[list[Segment], int]:
    """Fragmento curto entre dois blocos do mesmo falante vira desse falante.

    Vale quando pelo menos um dos dois blocos vizinhos é longo (>= 8 s). `janela`
    (início, fim) limita onde vale; sem ela, vale na reunião inteira. Devolve
    (blocos, quantos foram absorvidos). Coro não conta como vizinho.

    Numa conversa de duas pessoas, um "Sim." curto entre dois blocos longos do
    outro é fala real: por isso o uso no job é só dentro da janela da parte 1."""
    fala = [i for i, s in enumerate(segs) if s.speaker != CORO_LABEL]
    out = list(segs)
    n = 0
    for k in range(1, len(fala) - 1):
        i, a, b = fala[k], fala[k - 1], fala[k + 1]
        s = segs[i]
        if janela is not None and not (janela[0] <= s.start <= janela[1]):
            continue
        if not _fragmento(s):
            continue
        if segs[a].speaker != segs[b].speaker or segs[a].speaker == s.speaker:
            continue
        if max(_dur(segs[a]), _dur(segs[b])) < BLOCO_LONGO_SECONDS:
            continue
        out[i] = replace(s, speaker=segs[a].speaker)
        n += 1
    return out, n


def _mesma_pessoa(a: str, b: str) -> bool:
    fa, fb = _irmaos.fold(a), _irmaos.fold(b)
    return fa == fb or SequenceMatcher(None, fa, fb).ratio() >= nomes.FUZZY_MIN


def _resolve(cru: str, entradas: Sequence, participantes: Sequence[str]) -> tuple[str, str]:
    """(nome, confiança). Tenta o nome inteiro e depois o primeiro nome:
    "Ademir Balbino" vira "Ademir" pela lista. "Módemir" vira "Ademir" pelos
    participantes (semelhança 0,77). Sem casamento, fica a grafia crua, com
    confiança baixa."""
    candidatos = [cru] + ([cru.split()[0]] if " " in cru else [])
    for c in candidatos:
        if entradas:
            achado = _irmaos.corrige(c, entradas)
            if achado.casou:
                return achado.nome, "media" if achado.ambiguo else "alta"
        if participantes:
            known = {}
            for p in participantes:
                known.setdefault(_irmaos.fold(p), p)
            nome = nomes._canonical(c, {}, known)
            if nome:
                return nome, "media"
    return cru, "baixa"


# o anúncio segue com a apresentação do orador ("..., ele é da congregação X")
_APOS_TEMA = re.compile(r",\s*(?i:ele|ela)\s+(?i:é|e)\b.*$|,\s*(?i:da|do)\s+congrega.*$")


def _tema(texto: str) -> str:
    m = _TEMA.search(texto)
    if not m:
        return ""
    partes: list[str] = []
    for p in (x.strip() for x in _APOS_TEMA.sub("", m.group(1)).split(",")):
        # "como sempre X, como sempre X" (repetição do Whisper): uma vez só
        if p and (not partes or partes[-1].casefold() != p.casefold()):
            partes.append(p)
    return ", ".join(partes)[:200]


def nome_presidente(entradas: Sequence) -> Optional[str]:
    """Nome da lista com a observação "presidente". Mais de um: não escolhe."""
    achados = [e.nome for e in entradas
               if "presidente" in _irmaos.fold(e.observacao)]
    return achados[0] if len(achados) == 1 else None


def _nome_anterior(segs: list[Segment], fala: list[int], i: int) -> Optional[str]:
    """Último "boas-vindas ao irmão X" pouco antes do bloco i (o anúncio não repete o nome)."""
    for j in reversed([k for k in fala if k < i]):
        if segs[i].start - segs[j].end > NOME_ANTERIOR_SECONDS:
            break
        m = _BOAS_VINDAS.search(segs[j].text)
        if m:
            return m.group(1)
    return None


def _anuncios(segs: list[Segment], fala: list[int], entradas, participantes,
              est_inicio: Optional[float]):
    """Candidatos a anúncio: (índice, nome cru, nome resolvido, confiança, tema).

    O nome vem do próprio bloco ("irmão orador X") ou, se o bloco não traz,
    da boas-vindas anterior. Sem nome nenhum, não é candidato."""
    out = []
    for i in fala:
        s = segs[i]
        if est_inicio is not None and s.start >= est_inicio:
            break
        if not _ANUNCIO.search(s.text):
            continue
        m = _NOME_ORADOR.search(s.text) or _NOME_IRMAO.search(s.text)
        cru = m.group(1) if m else _nome_anterior(segs, fala, i)
        if cru is None:
            continue
        nome, conf = _resolve(cru, entradas, participantes)
        tema = _tema(s.text)
        if not tema:
            # o tema pode vir no bloco seguinte, se a frase quebrou
            prox = next((segs[j].text for j in fala if segs[j].start >= s.end), "")
            tema = _tema(prox) if prox else ""
        out.append((i, cru, nome, conf, tema))
    return out


def parte1(segs: list[Segment], entradas: Sequence = (),
           participantes: Sequence[str] = (),
           est_inicio: Optional[float] = None) -> Optional[Parte1]:
    """Estrutura do discurso público, ou None se não houver anúncio e agradecimento.

    Escolhe o ÚLTIMO anúncio que tem agradecimento depois, com o mesmo nome:
    o presidente pode citar o orador antes (ex.: abertura) sem que o discurso
    comece ali. O orador é o falante que mais fala entre o anúncio e o
    agradecimento. O presidente é o falante que mais fala antes do anúncio.
    """
    fala = [i for i, s in enumerate(segs) if s.speaker != CORO_LABEL]
    candidatos = _anuncios(segs, fala, entradas, participantes, est_inicio)
    for ia, cru, orador, conf, tema in reversed(candidatos):
        t_fim_anuncio = segs[ia].end
        agrad = None
        for i in fala:
            if segs[i].start < t_fim_anuncio:
                continue
            if est_inicio is not None and segs[i].start >= est_inicio:
                break
            m = _AGRADECE.search(segs[i].text)
            if m and _mesma_pessoa(_resolve(m.group(1), entradas, participantes)[0], orador):
                agrad = i
                break
        if agrad is None:
            continue
        # início do discurso: primeiro bloco longo depois do anúncio
        ini_ora = next((segs[i].start for i in fala
                        if segs[i].start >= t_fim_anuncio and _dur(segs[i]) >= BLOCO_LONGO_SECONDS
                        and i < agrad), None)
        if ini_ora is None:
            continue
        fim_ora = segs[agrad].start
        tempo_ora: Counter = Counter()
        for i in fala:
            s = segs[i]
            if ini_ora <= s.start < fim_ora:
                tempo_ora[s.speaker] += _dur(s)
        if not tempo_ora:
            continue
        cl_ora = tempo_ora.most_common(1)[0][0]
        pre = [i for i in fala if segs[i].start < segs[ia].start]
        base = pre or [i for i in fala if segs[i].start < ini_ora]
        tempo_pre: Counter = Counter()
        for i in base:
            tempo_pre[segs[i].speaker] += _dur(segs[i])
        if not tempo_pre:
            continue
        cl_pres = tempo_pre.most_common(1)[0][0]
        limite = est_inicio if est_inicio is not None else \
            segs[agrad].end + PRESIDENTE_APOS_SECONDS
        ind_pres = [i for i in fala if segs[i].speaker == cl_pres and (
            segs[i].start < ini_ora
            or (segs[i].start >= segs[agrad].start and segs[i].start < limite))]
        ind_ora = [i for i in fala if segs[i].speaker == cl_ora
                   and ini_ora <= segs[i].start < fim_ora]
        return Parte1(
            presidente=nome_presidente(entradas),
            presidente_cluster=cl_pres,
            orador=orador,
            orador_cru=cru,
            orador_confianca=conf,
            tema=tema,
            anuncio_inicio=segs[ia].start,
            agradecimento_inicio=segs[agrad].start,
            orador_cluster=cl_ora,
            orador_inicio=ini_ora,
            orador_fim=fim_ora,
            indices_presidente=ind_pres,
            indices_orador=ind_ora,
        )
    return None


def resumo(p: Parte1, blocos_orador: int, blocos_presidente: int, absorvidos: int) -> dict:
    """Dados da parte 1 para o estudo_info (tela, ata e testes)."""
    return {
        "presidente": p.presidente,
        "presidente_cluster": p.presidente_cluster,
        "orador": p.orador,
        "orador_cru": p.orador_cru,
        "orador_confianca": p.orador_confianca,
        "tema": p.tema,
        "anuncio_inicio": round(p.anuncio_inicio, 1),
        "agradecimento_inicio": (None if p.agradecimento_inicio is None
                                 else round(p.agradecimento_inicio, 1)),
        "orador_inicio": round(p.orador_inicio, 1),
        "orador_fim": round(p.orador_fim, 1),
        "blocos_orador": blocos_orador,
        "blocos_presidente": blocos_presidente,
        "fragmentos_absorvidos": absorvidos,
    }
