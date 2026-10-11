"""Descobre quem é quem pelo que as pessoas falam.

Duas pistas na transcrição:

1. Vocativo: "Então fechou, Fernando." Quem fala NÃO é o Fernando; quem
   responde (ou quem falou logo antes) provavelmente é.
2. Apresentação: "Aqui é o Fernando", "eu sou a Ana", "meu nome é Bia".
   Quem fala É essa pessoa (pista forte).

As pistas viram votos por falante; cada nome vai para no máximo um falante.
Só é sugerido quem tiver votos suficientes e folga sobre o segundo colocado.
É uma SUGESTÃO: aparece preenchida na revisão para o usuário confirmar.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Iterable, Optional

from .engines.base import Segment

MIN_VOTES = 2.0          # votos mínimos para sugerir um nome
NEXT_TURN_SECONDS = 15.0  # até quanto tempo depois a "resposta" ainda conta

_NAME = r"[A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][a-záâãàéêíóôõúç]{2,}"
# nome isolado entre pontuação/borda da frase: ", Fernando." / "Fernando, ..."
# O nome precisa estar com maiúscula; só as palavras ao redor ignoram caixa.
_VOCATIVE = re.compile(
    rf"(?:^|[,.!?…]\s+|^(?i:ô|ó|oi|olá|e aí|valeu|obrigad[oa]),?\s+)({_NAME})"
    rf"(?=\s*[,.!?…]|\s*$)", re.MULTILINE)
_INTRO = re.compile(
    rf"\b(?i:aqui (?:é|quem fala é) (?:o|a)|eu sou (?:o|a)|meu nome é|"
    rf"quem fala é (?:o|a)|sou (?:o|a))\s+({_NAME})")
# "Eduardo falando", "Fernando aqui" (apresentação no começo da frase)
_INTRO_AFTER = re.compile(rf"(?:^|[,.!?…]\s+)({_NAME})\s+(?i:falando|aqui)\b", re.MULTILINE)
# Fala citada ("ele disse: você demorou, Fernando") — o vocativo é de
# outra pessoa, não de quem está falando agora.
_QUOTE = re.compile(r"\b(?i:disse|falou|falei|perguntou|perguntei|respondeu|"
                    r"respondi|mandou|escreveu|me disse|ele disse|ela disse)\b")

# Palavras capitalizadas que não são nomes de pessoa (início de frase etc.)
_STOP = {
    "então", "sim", "não", "nao", "beleza", "perfeito", "ótimo", "otimo", "fechou",
    "exatamente", "isso", "tchau", "obrigado", "obrigada", "valeu", "bom", "boa",
    "olha", "cara", "mano", "gente", "pessoal", "senhor", "senhora", "deus",
    "nossa", "pois", "certo", "claro", "verdade", "combinado", "combinadíssimo",
    "tranquilo", "show", "legal", "ah", "oi", "olá", "ola", "tá", "ta", "ok",
    "amém", "amem", "irmão", "irmã", "irmao", "irma", "irmãos", "irmãs",
}


def _fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text.casefold())
                   if unicodedata.category(c) != "Mn")


# A stoplist é guardada JÁ normalizada. Antes ela tinha "então" com acento
# e a consulta usava _fold (sem acento): "Então" nunca era barrado e virou
# sugestão de nome (13 falas no salão).
_STOP_FOLD = {_fold(w) for w in _STOP}
# Palavras comuns que aparecem com maiúscula no começo de fala ou logo depois
# de vírgula ("Como vai", "Nós vamos"). Não são nomes de pessoa.
# Nomes sagrados e títulos que o Whisper escreve com maiúscula: não são pessoas
# da reunião ("Jeová" virou sugestão de falante no salão).
_SAGRADOS = {_fold(w) for w in ("jeová", "jeova", "jesus", "cristo", "satanás",
                                "satanas", "espírito", "espirito")}
_COMUNS = {_fold(w) for w in (
    "como", "nós", "nos", "aqui", "agora", "depois", "também", "mas", "porque",
    "quando", "onde", "pra", "para", "uma", "uns", "isso", "esse", "essa", "aí",
    "vamos", "vou", "né", "daí", "assim", "então", "entao", "cada", "todo",
    "toda", "todos", "todas", "outro", "outra", "nada", "algo", "hoje", "ontem",
)}


def _nao_e_nome(palavra: str) -> bool:
    """Palavra que nunca é nome de pessoa (stoplist, comuns, sagrados)."""
    k = _fold(palavra)
    return k in _STOP_FOLD or k in _COMUNS or k in _SAGRADOS


def _proper_nouns(segments: Iterable[Segment]) -> set[str]:
    """Palavras que aparecem com maiúscula NO MEIO da frase (não por serem
    a primeira palavra) — sinal de nome próprio.

    Heurística extra: se a mesma palavra aparece em minúscula no meio de
    alguma frase da transcrição ("... e então vamos"), não é nome, mesmo que
    o Whisper a tenha escrito com maiúscula em outro trecho.
    """
    seen: dict[str, int] = defaultdict(int)
    minusculas: set[str] = set()
    for seg in segments:
        for m in re.finditer(rf"(?<=[a-záâãàéêíóôõúç,]\s)({_NAME})\b", seg.text):
            seen[m.group(1)] += 1
        for m in re.finditer(r"(?<=[a-záâãàéêíóôõúç,]\s)([a-záâãàéêíóôõúç]{2,})\b",
                             seg.text):
            minusculas.add(_fold(m.group(1)))
    return {w for w, n in seen.items()
            if n >= 1 and not _nao_e_nome(w) and _fold(w) not in minusculas}


FUZZY_MIN = 0.75   # "Rosilene" ~ "Rosilaine" = 0,82


def _canonical(word: str, known: dict[str, str],
               trusted: Optional[dict[str, str]] = None) -> Optional[str]:
    """Nome informado/conhecido que corresponde à palavra ('fernando' ->
    'Fernando Souza'); None se não for um nome aceito.

    Grafia parecida com um nome INFORMADO (participante ou voz conhecida)
    vale por ele: o Whisper escreve "Rosilene" e quem se chama
    "Rosilaine Andrade" não pode virar uma pessoa nova."""
    key = _fold(word)
    if trusted:
        if key in trusted:
            return trusted[key]
        best = max(trusted, key=lambda k: SequenceMatcher(None, key, k).ratio())
        if SequenceMatcher(None, key, best).ratio() >= FUZZY_MIN:
            return trusted[best]
    return known.get(key)


def infer(segments: list[Segment], candidates: Iterable[str] = (),
          exclude: Optional[set] = None) -> dict[str, tuple[str, float, str]]:
    """{rótulo_cru: (nome, votos, motivo)} para os falantes identificados.

    `candidates`: nomes já conhecidos (participantes informados, vozes da
    memória). Além deles, aceita nomes próprios que aparecem na conversa.
    `exclude`: rótulos que já têm nome (pela voz) e não devem ser sugeridos.
    """
    segs = [s for s in segments if s.text.strip()]
    speakers = list(dict.fromkeys(s.speaker for s in segs))
    if len(speakers) < 1:
        return {}

    known: dict[str, str] = {}
    for full in candidates:
        full = full.strip()
        if not full:
            continue
        known.setdefault(_fold(full), full)
        first = full.split()[0]
        if len(first) >= 3:
            known.setdefault(_fold(first), full)
    trusted = dict(known)                     # só nomes informados/conhecidos
    for noun in _proper_nouns(segs):
        known.setdefault(_fold(noun), noun)

    votes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    reasons: dict[tuple[str, str], list[str]] = defaultdict(list)

    for i, seg in enumerate(segs):
        for m in [*_INTRO.finditer(seg.text), *_INTRO_AFTER.finditer(seg.text)]:
            if _nao_e_nome(m.group(1)):
                continue
            # quem se apresenta vale mesmo sem o nome ter aparecido antes
            name = _canonical(m.group(1), known, trusted) or m.group(1)
            if name:
                votes[seg.speaker][name] += 3.0
                reasons[(seg.speaker, name)].append("apresentou-se")

        for m in _VOCATIVE.finditer(seg.text):
            word = m.group(1)
            if _nao_e_nome(word):
                continue
            if _QUOTE.search(seg.text[:m.start()]):
                continue                             # está citando outra pessoa
            name = _canonical(word, known, trusted)
            if not name:
                continue
            # Quem chama provavelmente não é o chamado. Peso menor que o voto
            # a favor: falas curtas ("Valeu, Fernando.") às vezes caem no
            # falante errado, e o erro não pode anular as pistas boas.
            votes[seg.speaker][name] -= 0.5
            target = _addressee(segs, i)
            if target is not None:
                votes[target][name] += 1.0
                reasons[(target, name)].append("foi chamado pelo nome")

    # atribuição: maior voto primeiro, 1 nome por falante e vice-versa
    pairs = sorted(((v, spk, name) for spk, d in votes.items() for name, v in d.items()),
                   reverse=True)
    result: dict[str, tuple[str, float, str]] = {}
    used_names: set[str] = set()
    for v, spk, name in pairs:
        if v < MIN_VOTES or spk in result or name in used_names:
            continue
        if exclude and spk in exclude:
            continue
        others = [votes[o][name] for o in speakers if o != spk]
        if others and v - max(others) < 1.0:       # sem folga: ambíguo
            continue
        motivo = reasons[(spk, name)]
        resumo = ("apresentou-se" if "apresentou-se" in motivo
                  else f"chamado pelo nome {motivo.count('foi chamado pelo nome')}×")
        result[spk] = (name, v, resumo)
        used_names.add(name)
    return result


def _addressee(segs: list[Segment], i: int) -> Optional[str]:
    """Com quem a fala i está falando: quem fala em seguida (se for outra
    pessoa e logo depois); senão, quem falou por último antes dela."""
    me = segs[i].speaker
    for j in range(i + 1, len(segs)):
        if segs[j].start - segs[i].end > NEXT_TURN_SECONDS:
            break
        if segs[j].speaker != me:
            return segs[j].speaker
    for j in range(i - 1, -1, -1):
        if segs[j].speaker != me:
            return segs[j].speaker
    return None


ELIMINACAO_DETALHE = "único participante restante"


def _participante_de(nome: str, participantes: list[str]) -> Optional[int]:
    """Índice do participante a que um nome deduzido corresponde, ou None.

    Compara sem caixa e sem acento, aceita o primeiro nome ("Fernando" ->
    "Fernando TopVigia") e grafia parecida (mesmo limiar do FUZZY_MIN)."""
    chave = _fold(nome)
    melhor: Optional[int] = None
    nota = 0.0
    for i, p in enumerate(participantes):
        score = max(SequenceMatcher(None, chave, _fold(p)).ratio(),
                    SequenceMatcher(None, chave, _fold(p.split()[0])).ratio())
        if score > nota:
            melhor, nota = i, score
    return melhor if nota >= FUZZY_MIN else None


def por_eliminacao(falantes: Iterable[str], participantes: Iterable[str],
                   nomeados: dict[str, str]) -> dict[str, str]:
    """Sugere por eliminação: {rótulo: participante}, ou {} se não der.

    Só sugere quando o nº de falantes é igual ao nº de participantes
    informados, sobra exatamente 1 falante sem nome e exatamente 1
    participante ainda não usado. `nomeados`: {rótulo: nome} já decididos
    (voz, conversa ou informado pelo usuário).

    É SUGESTÃO: o resultado nunca vai para a memória de vozes, que só
    aprende o que o usuário confirma na revisão.
    """
    falantes = list(dict.fromkeys(falantes))
    parts = [p.strip() for p in dict.fromkeys(participantes) if p and p.strip()]
    if not parts or len(falantes) != len(parts):
        return {}
    sem_nome = [f for f in falantes if not (nomeados.get(f) or "").strip()]
    usados: set[int] = set()
    for nome in nomeados.values():
        if nome and nome.strip():
            idx = _participante_de(nome, parts)
            if idx is not None:
                usados.add(idx)
    livres = [p for i, p in enumerate(parts) if i not in usados]
    if len(sem_nome) != 1 or len(livres) != 1:
        return {}
    return {sem_nome[0]: livres[0]}
