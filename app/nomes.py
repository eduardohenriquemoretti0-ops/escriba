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
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Iterable, Optional

from . import irmaos as _irmaos
from .engines.base import CORO_LABEL, Segment

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


_fold = _irmaos.fold


# A stoplist é guardada JÁ normalizada. Antes ela tinha "então" com acento
# e a consulta usava _fold (sem acento): "Então" nunca era barrado e virou
# sugestão de nome (13 falas no salão).
_STOP_FOLD = {_fold(w) for w in _STOP}
# Palavras comuns que aparecem com maiúscula no começo de fala ou logo depois
# de vírgula ("Como vai", "Nós vamos"). Não são nomes de pessoa.
# Nomes sagrados e títulos que o Whisper escreve com maiúscula: não são pessoas
# da reunião ("Jeová" virou sugestão de falante no salão).
_SAGRADOS = {_fold(w) for w in ("jeová", "jeova", "jesus", "cristo", "satanás",
                                "satanas", "espírito", "espirito", "bíblia", "biblia")}
_COMUNS = {_fold(w) for w in (
    "como", "nós", "nos", "aqui", "agora", "depois", "também", "mas", "porque",
    "quando", "onde", "pra", "para", "uma", "uns", "isso", "esse", "essa", "aí",
    "vamos", "vou", "né", "daí", "assim", "então", "entao", "cada", "todo",
    "toda", "todos", "todas", "outro", "outra", "nada", "algo", "hoje", "ontem",
    # papéis da reunião e pronomes que o Whisper escreve com maiúscula
    "dirigente", "presidente", "leitor", "orador", "irmãos", "irmãs", "você",
    "voce", "vocês", "ele", "ela", "eles", "elas", "quem", "qual", "quais",
    "essa", "esse", "isso", "pessoal", "senhor", "senhora",
    # conjunções/advérbios que o Whisper escreve com maiúscula depois de
    # "obrigado"/ponto e viravam nome ("Obrigado. Que Jeová..." -> "Que")
    "que", "pois", "sim", "nao", "não", "em", "coro", "dentre", "antes",
    "pergunta", "resposta", "comentário", "comentario", "leitura", "cântico",
    "cantico", "texto", "parte", "designação", "designacao",
)}
# Lugares, livros e povos bíblicos: o Whisper escreve com maiúscula no meio da
# frase (nome próprio), mas não são pessoas da reunião ("fugir para o Egito"
# virava o falante "Egito"; "Judá", "Babilônia", "Jeremias" idem).
_LUGARES = {_fold(w) for w in (
    "egito", "judá", "juda", "israel", "babilônia", "babilonia", "jerusalém",
    "jerusalem", "roma", "betel", "jeremias", "ismael", "joanã", "joana",
    "gedalias", "babilônios", "babilonios", "rafa",
)}


def _nao_e_nome(palavra: str) -> bool:
    """Palavra que nunca é nome de pessoa (stoplist, comuns, sagrados)."""
    k = _fold(palavra)
    return k in _STOP_FOLD or k in _COMUNS or k in _SAGRADOS or k in _LUGARES


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


def _resolve(word: str, known: dict[str, str], trusted: Optional[dict[str, str]],
             entradas: list) -> Optional[str]:
    """Nome da lista de irmãos (com grafia corrigida) se a palavra casar com ela;
    senão, o caminho antigo (participantes informados e vozes conhecidas)."""
    if entradas:
        achado = _irmaos.corrige(word, entradas)
        if achado.casou:
            # Nome informado nesta reunião vence grafia/semelhança da lista:
            # "Fernando" com o participante "Fernando TopVigia" não vira o
            # "Fernando Croci" da lista só porque "Fernando" é grafia dele.
            if achado.origem != "exato" and trusted and _fold(word) in trusted:
                return trusted[_fold(word)]
            return achado.nome
    return _canonical(word, known, trusted)


def _mesmo_protegido(a: str, b: str, protegidos: set) -> bool:
    """Dois nomes EXATOS da lista (grafias diferentes) nunca se fundem: Mateus ≠ Matheus."""
    return (_fold(a) in protegidos and _fold(b) in protegidos and _fold(a) != _fold(b))


# --------------------------------------------------------------------------
# Reunião com vários falantes: a pessoa é nomeada pela POSIÇÃO no tempo.
# Ex. (salão): o dirigente chama antes do comentário ("Sim, irmão Fulano"),
# e agradece depois ("Obrigado, irmão Fulano"). Quem é chamado responde logo
# depois da chamada; quem é agradecido acabou de falar antes do agradecimento.
# --------------------------------------------------------------------------
CHAMADA_WINDOW_SECONDS = 15.0       # chamado responde até isso depois da chamada
AGRADECIMENTO_WINDOW_SECONDS = 60.0  # comentário agradecido até isso antes
PESO_POSICIONAL = 2.0               # voto de chamada/agradecimento
UNIFICA_MIN = 0.8                   # "Admir" ~ "Ademir" (0,91)
# "irmão", "irmã" etc. são título, não nome: ficam de fora da captura.
_TITULO = r"(?i:irm(?:ão|ã|aos|ãos|ãs|as|ao|a))"
_NC = rf"{_NAME}(?:\s+{_NAME})?"   # nome, ou nome composto ("José Eduardo")
# "Sim, irmão Fulano" / "Pois não, irmã Fulana"
_CHAMADA_TITULO = re.compile(
    rf"(?:^|[.!?…,]\s+)(?i:sim|pois não|pois nao|pode|certo|ok)\s*,\s*{_TITULO}\s+({_NC})\b")
# "irmão Fulano, por favor"
_CHAMADA_TITULO_FINAL = re.compile(
    rf"(?:^|[.!?…]\s+){_TITULO}\s+({_NC})\s*,?\s*(?i:por favor|por gentileza)\b")
# "Fulano, por favor"
_CHAMADA_NOME = re.compile(rf"(?:^|[.!?…,]\s+)({_NC})\s*,\s*(?i:por favor|por gentileza)\b")
# "Irmã Edna." / "O irmão Michel" / "Ainda o irmão Reinaldo": título + nome que
# não vem depois de "obrigado" na mesma frase é chamada (o dirigente passa a
# palavra). Frase = trecho desde o último ponto.
_CHAMADA_SOLTA = re.compile(rf"{_TITULO}\s+({_NC})\b")
# "Rita? Então," — nome seguido de pergunta do dirigente
_CHAMADA_QUESTAO = re.compile(rf"(?:^|[.!?…,]\s+)({_NC})\s*\?")
# "E informou o Michel?" / "O Fernando?" — artigo + nome no fim da pergunta
_CHAMADA_ARTIGO = re.compile(rf"(?:^|\s)(?i:o|a)\s+({_NC})\s*\?")
# "Obrigado, irmão Fulano" / "Muito bem, Fulana" / "obrigado irmão Admir"
_AGRADECIMENTO = re.compile(
    rf"(?i:muito obrigad[oa]|obrigad[oa]|muito bem|isso mesmo)\s*,?\s*"
    rf"(?:{_TITULO}\s+)?({_NC})\b")


def _unifica(nomes: list[str], protegidos: Optional[set] = None) -> dict[str, str]:
    """Grafias parecidas viram uma só (a mais frequente): "Admir" -> "Ademir".
    `protegidos`: nomes exatos da lista de irmãos (folded); não se fundem entre si."""
    protegidos = protegidos or set()
    freq = Counter(nomes)
    canon: list[str] = []
    mapa: dict[str, str] = {}
    # empate de frequência: vale a grafia que aparece primeiro na conversa
    for n in sorted(freq, key=lambda x: -freq[x]):
        achado = next((c for c in canon if SequenceMatcher(
            None, _fold(n), _fold(c)).ratio() >= UNIFICA_MIN
            and not _mesmo_protegido(n, c, protegidos)), None)
        if achado is None:
            canon.append(n)
            mapa[n] = n
        else:
            mapa[n] = achado
    return mapa


def _resumo_motivos(motivos: list[str]) -> str:
    """["chamado pelo dirigente", "agradecido", "agradecido"] ->
    "chamado pelo dirigente 1×, agradecido 2×" (apresentação vai sem contagem)."""
    if "apresentou-se" in motivos:
        return "apresentou-se"
    cont = Counter(motivos)
    return ", ".join(f"{m} {n}×" for m, n in cont.items())


def _limpa_nome(nome: str) -> str:
    """Tira do começo e do fim as palavras que não são nome de pessoa.

    O nome composto captura duas palavras ("Marques Agora"); a 2ª às vezes é
    uma palavra comum ("Agora", "Pergunta") que não faz parte do nome. Sobra o
    nome de verdade, ou "" se não restar nada."""
    partes = nome.split()
    while partes and _nao_e_nome(partes[-1]):
        partes.pop()
    while partes and _nao_e_nome(partes[0]):
        partes.pop(0)
    return " ".join(partes)


def eventos_do_texto(texto: str) -> list[tuple[str, str]]:
    """(tipo, nome cru) de cada chamada e agradecimento num texto.

    Cada posição do texto conta uma vez. Agradecimento tem prioridade sobre
    chamada ("Muito bem, irmã X" é agradecimento). Fala citada ("disse: sim,
    irmão X") não conta."""
    achados: dict[int, tuple[str, str]] = {}
    for m in _AGRADECIMENTO.finditer(texto):
        achados.setdefault(m.start(1), ("agradecimento", m.group(1)))
    for rx in (_CHAMADA_TITULO, _CHAMADA_TITULO_FINAL, _CHAMADA_NOME, _CHAMADA_QUESTAO,
               _CHAMADA_ARTIGO):
        for m in rx.finditer(texto):
            achados.setdefault(m.start(1), ("chamada", m.group(1)))
    for m in _CHAMADA_SOLTA.finditer(texto):
        if re.search(r"(?i)obrigad", re.split(r"[.!?…]", texto[:m.start()])[-1]):
            continue                             # frase de agradecimento
        achados.setdefault(m.start(1), ("chamada", m.group(1)))
    saida = []
    for pos, (tipo, nome) in sorted(achados.items()):
        nome = _limpa_nome(nome)                 # tira "Agora"/"Pergunta" etc. do nome
        if not nome:
            continue
        frase = re.split(r"[.!?…]", texto[:pos])[-1]
        if _QUOTE.search(frase):
            continue                             # está citando outra pessoa
        saida.append((tipo, nome))
    return saida


def _com_titulo_ou_pedido(texto: str, nome: str) -> bool:
    """O nome vem depois de "irmão/irmã" ou junto de "por favor"/"por gentileza"?"""
    n = re.escape(nome)
    return bool(re.search(rf"{_TITULO}\s+{n}\b", texto)
                or re.search(rf"(?i:por favor|por gentileza)\s*,?\s*{n}\b", texto)
                or re.search(rf"\b{n}\s*,?\s*(?i:por favor|por gentileza)", texto))


def _eventos_posicionais(segs: list[Segment]) -> list[tuple[str, int, str]]:
    """(tipo, índice da fala, nome cru) para chamadas e agradecimentos."""
    return [(t, i, n) for i, seg in enumerate(segs) for t, n in eventos_do_texto(seg.text)]


def _alvo_chamada(segs: list[Segment], i: int) -> Optional[str]:
    """Quem responde à chamada da fala i: o próximo falante diferente, se
    começar até CHAMADA_WINDOW_SECONDS depois."""
    me = segs[i].speaker
    for j in range(i + 1, len(segs)):
        if segs[j].start - segs[i].end > CHAMADA_WINDOW_SECONDS:
            return None
        if segs[j].speaker != me:
            return segs[j].speaker
    return None


def _alvo_agradecimento(segs: list[Segment], i: int) -> Optional[str]:
    """Quem é agradecido na fala i: o falante diferente mais recente antes,
    se terminou até AGRADECIMENTO_WINDOW_SECONDS antes."""
    me = segs[i].speaker
    for j in range(i - 1, -1, -1):
        if segs[i].start - segs[j].end > AGRADECIMENTO_WINDOW_SECONDS:
            return None
        if segs[j].speaker != me:
            return segs[j].speaker
    return None


def dirigente_de(segs: list[Segment]) -> Optional[str]:
    """Quem mais chama e agradece (pelo menos 2 vezes): o dirigente da reunião."""
    cont = Counter(segs[i].speaker for _, i, _ in _eventos_posicionais(segs))
    if not cont:
        return None
    spk, n = cont.most_common(1)[0]
    return spk if n >= 2 else None


def por_chamada(segments: list[Segment], entradas: Iterable = (),
                participantes: Iterable[str] = (),
                ) -> dict[str, tuple[str, str, bool]]:
    """{rótulo_cru: (nome, motivo, está_na_lista)} pela CHAMADA/AGRADECIMENTO.

    Regra do Eduardo: quando alguém é chamado ("convidamos o irmão Fulano",
    "irmão Fulano, por favor") ou agradecido ("obrigado, irmão Fulano"), o nome
    — com a grafia corrigida pela lista de irmãos — vai para a PARTE TODA: o
    próximo falante diferente (chamada) ou o falante anterior (agradecimento).
    Uma chamada clara basta; não há corte por margem de votos (ao contrário de
    `infer`), para o nome não sumir. `está_na_lista` é False quando o nome não
    casa com a lista nem com os participantes — aí vira dúvida (ouvir o trecho).
    """
    segs = [s for s in segments if s.text.strip()]
    if not segs:
        return {}
    entradas = list(entradas)
    parts = [p for p in participantes if p and p.strip()]
    known: dict[str, str] = {}
    for full in parts:
        known.setdefault(_fold(full), full)
        first = full.split()[0]
        if len(first) >= 3:
            known.setdefault(_fold(first), full)
    trusted = dict(known)

    eventos = []
    for tipo, i, cru in _eventos_posicionais(segs):
        nome = _resolve(cru, known, trusted, entradas) or cru
        eventos.append((tipo, i, nome))

    # Quem CONDUZ chama/passa a palavra muitas vezes (presidente, quem dirige
    # joias, dirigente do estudo). Um condutor NÃO é nomeado por uma chamada
    # solta que caiu de volta nele (o próximo falante depois da chamada às vezes
    # é o próprio condutor): ele é nomeado pelo ANÚNCIO de quem preside.
    chefe = presidente_reuniao(segs) or dirigente_de(segs)
    faz = Counter(segs[i].speaker for tipo, i, _ in eventos if tipo == "chamada")
    CONDUTOR_MIN = 3

    def _peso(quem: str) -> int:
        if chefe and quem == chefe:
            return 3                      # anúncio de quem preside: mais forte
        return 2 if faz.get(quem, 0) >= CONDUTOR_MIN else 1

    pronunciou: dict[str, set] = defaultdict(set)
    for tipo, i, nome in eventos:
        pronunciou[nome].add(segs[i].speaker)

    votos: dict[str, Counter] = defaultdict(Counter)
    motivos: dict[tuple[str, str], str] = {}
    anunciado: set[tuple[str, str]] = set()   # (alvo, nome) vindo de quem preside
    for tipo, i, nome in eventos:
        quem = segs[i].speaker
        alvo = (_alvo_chamada(segs, i) if tipo == "chamada"
                else _alvo_agradecimento(segs, i))
        if alvo is None or alvo in pronunciou[nome]:
            continue
        votos[alvo][nome] += _peso(quem)
        if chefe and quem == chefe:
            anunciado.add((alvo, nome))
        if (alvo, nome) not in motivos:
            if tipo == "chamada":
                motivos[(alvo, nome)] = ("anunciado por quem preside"
                                         if chefe and quem == chefe
                                         else "chamado pelo nome")
            else:
                motivos[(alvo, nome)] = "agradecido"

    # 1 nome por cluster (maior voto) e 1 cluster por nome. Para um cluster
    # CONDUTOR, só vale o nome que veio do anúncio de quem preside.
    pares = sorted(((n, spk, nome) for spk, c in votos.items()
                    for nome, n in c.items()), reverse=True)
    out: dict[str, tuple[str, str, bool]] = {}
    usados: set[str] = set()
    for _, spk, nome in pares:
        if spk in out or nome in usados:
            continue
        if (spk != chefe and faz.get(spk, 0) >= CONDUTOR_MIN
                and (spk, nome) not in anunciado):
            continue                      # condutor só é nomeado pelo anúncio
        na_lista = bool(
            (entradas and _irmaos.corrige_participante(nome, entradas) is not None)
            or _participante_de(nome, parts) is not None)
        out[spk] = (nome, motivos.get((spk, nome), "chamado pelo nome"), na_lista)
        usados.add(nome)
    return out


def presidente_reuniao(segments: list[Segment]) -> Optional[str]:
    """Rótulo cru de quem preside a reunião de meio de semana, ou None.

    O presidente abre a reunião e anuncia/agradece as partes. Âncora segura:
    o falante da PRIMEIRA fala que também está entre os dois que mais chamam e
    agradecem no total. Se a primeira fala não for de um chamador frequente
    (ex.: começou por cântico/oração), não arrisca e devolve None."""
    segs = [s for s in segments if s.text.strip() and s.speaker != CORO_LABEL]
    if not segs:
        return None
    primeiro = segs[0].speaker
    cont = Counter(segs[i].speaker for _, i, _ in _eventos_posicionais(segs))
    if not cont:
        return None
    top = [spk for spk, _ in cont.most_common(2)]
    if primeiro in top and cont[primeiro] >= 2:
        return primeiro
    return None


def infer(segments: list[Segment], candidates: Iterable[str] = (),
          exclude: Optional[set] = None,
          irmaos: Iterable = ()) -> dict[str, tuple[str, float, str]]:
    """{rótulo_cru: (nome, votos, motivo)} para os falantes identificados.

    `candidates`: nomes já conhecidos (participantes informados, vozes da
    memória). Além deles, aceita nomes próprios que aparecem na conversa.
    `exclude`: rótulos que já têm nome (pela voz) e não devem ser sugeridos.
    `irmaos`: entradas da lista de irmãos (`app.irmaos.Entrada`). Grafia
    parecida é corrigida para o nome da lista; nome exato não se funde com
    outro nome da lista.

    Com mais de 2 falantes, o nome vem da posição no tempo (chamada ->
    próximo falante; agradecimento -> falante anterior). Com 2, vale também
    o vocativo "ao interlocutor", como antes.
    """
    segs = [s for s in segments if s.text.strip()]
    speakers = list(dict.fromkeys(s.speaker for s in segs))
    if len(speakers) < 1:
        return {}
    entradas = list(irmaos)
    protegidos = _irmaos.nomes_protegidos(entradas)

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
    proprios = _proper_nouns(segs)
    for noun in proprios:
        known.setdefault(_fold(noun), noun)
    proprios_fold = {_fold(n) for n in proprios}

    votes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    reasons: dict[tuple[str, str], list[str]] = defaultdict(list)

    for seg in segs:
        for m in [*_INTRO.finditer(seg.text), *_INTRO_AFTER.finditer(seg.text)]:
            if _nao_e_nome(m.group(1)):
                continue
            # quem se apresenta vale mesmo sem o nome ter aparecido antes
            name = _resolve(m.group(1), known, trusted, entradas) or m.group(1)
            if name:
                votes[seg.speaker][name] += 3.0
                reasons[(seg.speaker, name)].append("apresentou-se")

    # --- chamadas e agradecimentos (posição no tempo) -----------------------
    eventos = _eventos_posicionais(segs)
    crus = [n for _, _, n in eventos]
    if entradas:
        # corrige a grafia antes de unificar: "Niffa" vira "Ninfa" pela lista
        corr = {c: _resolve(c, known, trusted, entradas) or c for c in set(crus)}
        crus = [corr[c] for c in crus]
    mapa = _unifica(crus, protegidos)
    nomes_evento = []
    for (tipo, i, cru0), cru in zip(eventos, crus):
        nome = mapa[cru] if entradas else (_canonical(mapa[cru], known, trusted) or mapa[cru])
        # Só vale nome com cara de nome: participante/voz conhecida, lista de
        # irmãos, ou palavra que aparece com maiúscula no meio de frase. Na
        # conversa de 2 pessoas, "Entendeu?" (3×) casava com a chamada "X?",
        # valia 6 votos e tirava o nome do Fernando (palavras com nome certo
        # 97,9% -> 0%).
        if not (_resolve(cru0, {}, trusted, entradas) or _fold(cru0) in proprios_fold
                or _fold(nome) in trusted or _com_titulo_ou_pedido(segs[i].text, cru0)):
            continue
        nomes_evento.append((tipo, i, nome))
    dirigente = dirigente_de(segs)
    # quem pronuncia o nome numa chamada/agradecimento não pode receber esse nome
    pronunciou: dict[str, set] = defaultdict(set)
    for tipo, i, nome in nomes_evento:
        pronunciou[nome].add(segs[i].speaker)
    for tipo, i, nome in nomes_evento:
        quem = segs[i].speaker
        alvo = (_alvo_chamada(segs, i) if tipo == "chamada"
                else _alvo_agradecimento(segs, i))
        votes[quem][nome] -= 0.5                 # quem chama/agradece não é o chamado
        if alvo is None or alvo in pronunciou[nome]:
            continue
        votes[alvo][nome] += PESO_POSICIONAL
        if tipo == "chamada":
            motivo = ("chamado pelo dirigente" if quem == dirigente
                      else "chamado pelo nome")
        else:
            motivo = "agradecido"
        reasons[(alvo, nome)].append(motivo)

    # --- vocativo "ao interlocutor": só em conversa de 2 pessoas ------------
    if len(speakers) <= 2:
        for i, seg in enumerate(segs):
            for m in _VOCATIVE.finditer(seg.text):
                word = m.group(1)
                if _nao_e_nome(word):
                    continue
                if _QUOTE.search(seg.text[:m.start()]):
                    continue                     # está citando outra pessoa
                name = _resolve(word, known, trusted, entradas)
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
        result[spk] = (name, v, _resumo_motivos(reasons[(spk, name)]))
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
