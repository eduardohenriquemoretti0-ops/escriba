"""Estudo bíblico por palavra: o dirigente chama, o chamado fala.

Camada sobre a separação por turnos (``turnos.reatribui``). Mede-se no salão
(10/10) que a voz não separa quem fala pelo mesmo microfone ou PA; a estrutura
da reunião separa. Esta camada é a combinação ADLVGC do laboratório do Escriba
v5 (fase F4), menos a retranscrição dos buracos (c), que fica em
``app/buracos.py``:

* (e) guarda: só liga com dirigente, pelo menos ``GUARDA_CHAMADAS`` chamadas e
  ``GUARDA_PARAGRAFOS`` anúncios de parágrafo/leitura. Sem ela, a conversa de
  2 pessoas ia de DER 13,0 para 30,8 com 2 nomes falsos ("Entendeu", "Sei").
* (a) dirigente ↔ chamado: depois de cada chamada do dirigente, a fala é do
  chamado até o dirigente voltar (agradecimento, "isso/muito bem/exatamente",
  pergunta nova, nova chamada ou anúncio de parágrafo), MESMO que a voz caia no
  cluster do dirigente ou do leitor (regra do Eduardo: estrutura manda).
* (d) nome só com evidência: trecho sem chamada nem agradecimento vira "[?]";
  homônimo da lista vira "[?] X (homônimo)"; chamada e agradecimento que
  discordam viram "[?]".
* (L) nome fora da lista de irmãos e dos participantes precisa de 2ª menção
  (chamada + agradecimento). Com uma menção só, sai "[?] X?".
* (V) troca de voz dentro do trecho do chamado: frase com cosseno abaixo de
  ``VOZ_TROCA`` em relação aos primeiros ``VOZ_REF_SECONDS`` s do trecho corta
  o trecho (92% de acerto nos cortes, medido no salão).
* (G) trechos "[?]" com voz parecida (>= ``GRUPO_DUVIDA``) ficam com o mesmo
  rótulo. O nome continua "[?]": nunca junta com pessoa nomeada.

Só sugere. Os nomes vão para ``nomes_auto`` pelo worker e nunca para a memória
de vozes sem confirmação do usuário. Calibração dentro da amostra: há uma
gravação de estudo só; guarda, ``VOZ_TROCA`` e a gramática de chamada
precisam ser validados noutra reunião.
"""

from __future__ import annotations

import bisect
import logging
import re
from collections import Counter
from difflib import SequenceMatcher
from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Sequence

import numpy as np

from . import irmaos as _irmaos
from . import nomes as _nomes
from . import turnos as _turnos
from .engines.base import CORO_LABEL, Segment

log = logging.getLogger(__name__)

GUARDA_CHAMADAS = 8  # chamadas do dirigente para ligar a estrutura
GUARDA_PARAGRAFOS = 2  # anúncios de parágrafo/leitura do dirigente
RETORNO_MAX_PAUSA = 20.0  # pausa maior que isso encerra o trecho do chamado
TRECHO_MAX = 150.0  # teto de um trecho de chamado (segurança)
LEITURA_MIN = 8.0  # bloco de leitura tem pelo menos isso (s)
FIM_APOS_ULTIMO_EVENTO = 90.0  # o estudo acaba até isso depois do último evento
UTTERANCE_MAX_GAP = 10.0  # igual ao motor local
AGRADECIMENTO_JANELA = 6.0  # s, agradecimento no começo da fala do dirigente
AGRADECIMENTO_PALAVRAS = 12  # palavras, idem
RETORNO_AGRADECE_PALAVRAS = 6  # agradecimento com nome no começo da frase
CORRIDA_MAX_PAUSA = 15.0  # trecho sem dono: agradecimento até isso depois
VOZ_TROCA = 0.40  # frase com cosseno < isso à voz do início = outra pessoa
VOZ_REF_SECONDS = 6.0  # voz de referência: primeiros s do trecho
GRUPO_DUVIDA = 0.80  # mesmo limiar de turnos.LIMIAR_VOZ; só entre "[?]"
GRUPO_DUVIDA_MIN_SECONDS = 3.0

DUVIDA = "[?]"
# Sugestão "[?] X?" (nome com uma só menção): só aparece se o candidato for
# parecido com a lista/participantes, vier depois de "irmão/irmã" ou de uma
# autoapresentação. Senão, só "[?]". No salão saíam "Temos?", "Digital?",
# "Eva?" (de "Adão e Eva") e "Fernando Cross?".
SUGESTAO_SEMELHANCA = 0.75
_TITULOS = {"irmao", "irma", "irmaos", "irmas"}
_POR_FAVOR_PALAVRAS = {"por favor", "por gentileza"}
_APRESENTACAO = re.compile(
    r"(?:\bsou (?:o|a)|\bnome e|\baqui e (?:o|a)|\bquem fala e (?:o|a))$"
)
# palavras que nunca são sugeridas como nome: comuns do português e nomes
# bíblicos que aparecem no texto lido
_BLOQUEIO_SUGESTAO = {
    _irmaos.fold(w)
    for w in (
        "adão",
        "eva",
        "jesus",
        "jeová",
        "cristo",
        "satanás",
        "diabo",
        "deus",
        "davi",
        "moisés",
        "paulo",
        "pedro",
        "joão",
        "abraão",
        "isaque",
        "jacó",
        "josé",
        "maria",
        "noé",
        "jó",
        "daniel",
        "elias",
        "isaías",
        "jeremias",
        "salomão",
        "saul",
        "samuel",
        "rute",
        "ester",
        "sara",
        "raquel",
        "lázaro",
        "mateus",
        "marcos",
        "lucas",
        "tiago",
        "timóteo",
        "tito",
        "filemom",
        "gênesis",
        "êxodo",
        "salmos",
        "provérbios",
        "apocalipse",
        "romanos",
        "coríntios",
        "hebreus",
        "efésios",
        "gálatas",
        "filipenses",
        "israel",
        "temos",
        "digital",
        "tem",
        "ter",
        "ser",
        "estar",
        "fazer",
        "pode",
        "podemos",
        "sabe",
        "sabemos",
        "vemos",
        "vimos",
        "veja",
        "vejam",
        "certo",
        "sim",
        "não",
        "bom",
        "boa",
        "bem",
        "muito",
        "mais",
        "menos",
        "todo",
        "nada",
        "aqui",
        "ali",
        "agora",
        "hoje",
        "então",
        "também",
        "porque",
        "quando",
        "como",
        "onde",
        "qual",
        "quem",
        "que",
        "isso",
        "esse",
        "essa",
        "este",
        "esta",
        "outro",
        "outra",
        "mesmo",
        "parágrafo",
        "pergunta",
        "resposta",
        "texto",
        "versículo",
        "capítulo",
        "sentinela",
        "estudo",
        "leitura",
    )
}
_PREFIXO_ROTULO_NOME = "E"  # rótulos novos de pessoa nomeada: E1, E2...
_PREFIXO_ROTULO_DUVIDA = "Q"  # rótulos de trecho em dúvida: Q1, Q2...

# anúncio de leitura: "parágrafo", "leitura", "Vamos ao 12.", "prosseguir 6 e 7", "16."
_ANUNCIO = re.compile(
    r"(?i)par[aá]grafo|leitura|\b(vamos|prosseguir|seguir|ir)\b[^.?!]*\b\d+"
    r"|^\s*\d+\s*[º°]?\s*[.!]?\s*$"
)
_RETORNO = re.compile(
    r"^(isso|muito bem|muito obrigad|obrigad|exatamente|perfeito|"
    r"correto|ta bom|bem lembrado|muito bom)\b"
)
_POR_FAVOR = re.compile(r"^(por favor|por gentileza|pode falar|pois nao)\W*$")
_NAO_VOCATIVO = re.compile(r"^(comentou|disse|falou|lembrou|mencionou|leu)$")


def e_duvida(nome: Optional[str]) -> bool:
    """Nome exibido que marca dúvida ("[?]", "[?] X?", "[?] X (homônimo)")."""
    return bool(nome) and nome.strip().startswith(DUVIDA)


# --------------------------------------------------------------------------
# Dados
# --------------------------------------------------------------------------
@dataclass
class Palavra:
    """Palavra do Whisper com a frase, o cluster de voz e o rótulo de base."""

    inicio: float
    fim: float
    texto: str
    frase: int = -1
    cluster: int = -1
    seg: int = -1  # bloco de base (saída de turnos.reatribui)
    rotulo_base: Optional[str] = None
    nome_base: str = ""


@dataclass
class Frase:
    """Frase da diarização: índices das palavras, cluster e voz."""

    i0: int
    i1: int
    cluster: int
    voz: Optional[np.ndarray] = None
    eventos: list[tuple[str, str, int]] = field(default_factory=list)

    def idx(self) -> range:
        """Índices das palavras da frase."""
        return range(self.i0, self.i1 + 1)


@dataclass
class Estrutura:
    """Estrutura do estudo detectada pela guarda."""

    ativa: bool
    dirigente: int = -1
    chamadas: int = 0
    paragrafos: int = 0
    inicio: float = 0.0
    fim: float = 0.0
    motivo: str = ""


@dataclass
class Dono:
    """Quem fala uma palavra: rótulo, nome exibido e de onde veio."""

    rotulo: Optional[str]
    nome: str
    origem: str
    exibido: Optional[str] = None  # nome mostrado, se diferente de `nome`

    @property
    def mostra(self) -> str:
        """Nome que vai para a tela e os arquivos."""
        return self.nome if self.exibido is None else self.exibido


@dataclass
class Resultado:
    """Saída da camada: blocos, pessoas (para o worker) e resumo."""

    ativa: bool
    segmentos: list[Segment]
    pessoas: list[_turnos.Pessoa]
    info: dict


class _Nomes:
    """Grafia e conhecimento dos nomes (lista de irmãos + participantes)."""

    def __init__(self, entradas: Sequence, participantes: Sequence[str]) -> None:
        self.entradas = list(entradas)
        self.part = [p for p in participantes if p and p.strip()]

    def corrige(self, nome: str) -> tuple[str, bool]:
        """(grafia da lista/participantes, é homônimo?).

        Participante exato vem antes da lista: "Eduardo" (participante) não
        pode virar "José Eduardo" (parte do nome)."""
        for pt in self.part:
            if _irmaos.fold(pt) == _irmaos.fold(nome):
                return pt, False
        n, amb = _turnos._na_lista(nome, self.entradas, self.part)
        return n or nome, amb

    def conhecido(self, nome: str) -> bool:
        """Nome EXATO da lista (nome ou grafia cadastrada) ou dos participantes.

        Correção por semelhança ("Ana" -> "Deana") não conta como conhecido."""
        if any(_irmaos.fold(pt) == _irmaos.fold(nome) for pt in self.part):
            return True
        if not self.entradas:
            return False
        c = _irmaos.corrige(nome, self.entradas)
        return c.casou and c.origem in ("exato", "grafia")

    def parecido(self, nome: str) -> bool:
        """Grafia parecida (>= SUGESTAO_SEMELHANCA) com um nome da lista
        (nome, partes do nome e grafias cadastradas) ou dos participantes."""
        alvos: set[str] = set()
        for e in self.entradas:
            alvos.add(_irmaos.fold(e.nome))
            alvos.update(_irmaos.fold(x) for x in e.nome.split())
            alvos.update(_irmaos.fold(g) for g in e.grafias)
        for pt in self.part:
            alvos.add(_irmaos.fold(pt))
            alvos.update(_irmaos.fold(x) for x in pt.split())
        chave = _irmaos.fold(nome)
        return any(
            SequenceMatcher(None, chave, a).ratio() >= SUGESTAO_SEMELHANCA
            for a in alvos
            if a
        )

    @staticmethod
    def mesmo(a: str, b: str) -> bool:
        """Mesma pessoa pela grafia (limiar de unificação dos nomes)."""
        return _turnos._mesmo_nome(a, b)


# --------------------------------------------------------------------------
# Texto
# --------------------------------------------------------------------------
def _texto(pal: list[Palavra], f: Frase) -> str:
    return "".join(pal[i].texto for i in f.idx()).strip()


def _limpa(t: str) -> str:
    return re.sub(r"[^\w ]", "", _irmaos.fold(t)).strip()


def _eventos_frase(pal: list[Palavra], f: Frase) -> list[tuple[str, str, int]]:
    """(tipo, nome, índice da palavra do nome) de chamadas e agradecimentos."""
    out = []
    for tipo, nome in _nomes.eventos_do_texto(_texto(pal, f)):
        alvo = _limpa(nome.split()[-1])
        pos = [i for i in f.idx() if _limpa(pal[i].texto) == alvo]
        if not pos:
            continue
        out.append((tipo, nome, pos[-1] if tipo == "chamada" else pos[0]))
    return out


# --------------------------------------------------------------------------
# (e) Guarda e estrutura
# --------------------------------------------------------------------------
def estrutura(pal: list[Palavra], frases: list[Frase]) -> Estrutura:
    """Dirigente (cluster que mais chama/agradece), região do estudo e guarda.

    A estrutura só liga com dirigente, >= GUARDA_CHAMADAS chamadas e
    >= GUARDA_PARAGRAFOS anúncios de parágrafo/leitura do dirigente."""
    cont: Counter = Counter()
    cham: Counter = Counter()
    for f in frases:
        f.eventos = _eventos_frase(pal, f)
        for tipo, _, _ in f.eventos:
            cont[f.cluster] += 1
            if tipo == "chamada":
                cham[f.cluster] += 1
    if not cont:
        return Estrutura(False, motivo="nenhuma chamada/agradecimento")
    d = cont.most_common(1)[0][0]
    par = sum(
        1 for f in frases if f.cluster == d and _turnos._LEITURA.search(_texto(pal, f))
    )
    e = Estrutura(True, d, cham[d], par)
    if cham[d] < GUARDA_CHAMADAS or par < GUARDA_PARAGRAFOS:
        e.ativa = False
        e.motivo = f"guarda: {cham[d]} chamadas, {par} anúncios de parágrafo"
        return e
    df = [f for f in frases if f.cluster == d]
    ini = next(
        (pal[f.i0].inicio for f in df if _turnos._ESTUDO.search(_texto(pal, f))), None
    )
    com_ev = [f for f in df if f.eventos]
    if ini is None:
        ini = pal[com_ev[0].i0].inicio
    fim = max(pal[f.i1].fim for f in com_ev) + FIM_APOS_ULTIMO_EVENTO
    for f in frases:  # a chamada da oração final encerra o estudo
        if pal[f.i0].inicio > ini and _turnos._ORACAO_CHAMADA.search(_texto(pal, f)):
            fim = min(fim, pal[f.i0].inicio)
            break
    e.inicio, e.fim = ini, fim
    return e


# --------------------------------------------------------------------------
# (a) Dirigente ↔ chamado
# --------------------------------------------------------------------------
def _e_retorno(pal: list[Palavra], f: Frase, d: int, leitura: set[int], k: int) -> bool:
    """A frase k devolve a palavra ao dirigente (ou é leitura)?"""
    if k in leitura:
        return True
    tx = _texto(pal, f)
    fo = _limpa(tx)
    if f.cluster == d:
        if _RETORNO.match(fo) or any(
            t in ("agradecimento", "chamada") for t, _, _ in f.eventos
        ):
            return True
        pergunta = tx.endswith("?") and not _irmaos.fold(tx).endswith(("ne?", "nao e?"))
        return pergunta and f.i1 - f.i0 + 1 >= 4
    # fora do cluster do dirigente: só agradecimento com nome logo no início
    return bool(_RETORNO.match(fo)) and any(
        t == "agradecimento" and i - f.i0 <= RETORNO_AGRADECE_PALAVRAS
        for t, _, i in f.eventos
    )


def leituras(pal: list[Palavra], frases: list[Frase], e: Estrutura) -> set[int]:
    """Índices das frases de leitura (depois do anúncio do dirigente)."""
    out: set[int] = set()
    k = 0
    while k < len(frases):
        f = frases[k]
        tx = _texto(pal, f)
        m = _ANUNCIO.search(tx) if f.cluster == e.dirigente else None
        if (
            m
            and e.inicio <= pal[f.i0].inicio < e.fim
            and not _turnos._AGRADECE.search(re.split(r"[.!?…]", tx[: m.start()])[-1])
        ):
            j = k + 1
            bloco = []
            while j < len(frases) and frases[j].cluster != e.dirigente:
                bloco.append(j)
                j += 1
            if (
                bloco
                and (pal[frases[bloco[-1]].i1].fim - pal[frases[bloco[0]].i0].inicio)
                >= LEITURA_MIN
            ):
                out.update(bloco)
            k = j
            continue
        k += 1
    return out


def _vocativo_final(
    pal: list[Palavra], frases: list[Frase], k: int, d: int
) -> Optional[tuple[str, int]]:
    """Chamada sem título no fim do bloco do dirigente: "...ideia? Fernando,".

    Olha as últimas 8 palavras das frases do dirigente que terminam na frase k.
    Vale a última palavra com maiúscula que não é palavra comum, seguida de
    pontuação, fora de oração de agradecimento e de citação ("como o X comentou")."""
    idx: list[int] = []
    j = k
    while j >= 0 and frases[j].cluster == d and len(idx) < 8:
        idx = list(frases[j].idx()) + idx
        j -= 1
    idx = idx[-8:]
    for pos in range(len(idx) - 1, -1, -1):
        i = idx[pos]
        w = pal[i].texto.strip()
        nome = re.sub(r"[^\w]", "", w)
        if len(nome) < 3 or not nome[:1].isupper() or _nomes._nao_e_nome(nome):
            continue
        if not (w[-1:] in ",.?!" or pos == len(idx) - 1):
            continue
        antes = " ".join(_limpa(pal[x].texto) for x in idx[max(0, pos - 3) : pos])
        depois = _limpa(pal[idx[pos + 1]].texto) if pos + 1 < len(idx) else ""
        if re.search(r"\b(como|que)( o| a)?$", antes) or _NAO_VOCATIVO.match(depois):
            continue
        clausula = re.split(r"[.!?…]", " ".join(pal[x].texto for x in idx[: pos + 1]))[
            -1
        ]
        if re.search(r"(?i)obrigad|muito bem|isso mesmo", clausula):
            return None
        return nome, i
    return None


def _pula_por_favor(pal: list[Palavra], i: int, f: Frase) -> int:
    while i <= f.i1 and _limpa(pal[i].texto) in ("por", "favor", "gentileza", ""):
        i += 1
    return i


def chamadas(
    pal: list[Palavra], frases: list[Frase], e: Estrutura, nm: _Nomes
) -> list[tuple[int, str, int, int]]:
    """(1ª palavra do trecho do chamado, nome cru, frase da chamada, palavra do nome)."""
    out: list[tuple[int, str, int, int]] = []
    for k, f in enumerate(frases):
        if not (e.inicio <= pal[f.i0].inicio < e.fim):
            continue
        if f.cluster == e.dirigente:
            ch = [(n, i) for t, n, i in f.eventos if t == "chamada"]
            if ch:
                n, i = ch[-1]
                out.append((_pula_por_favor(pal, i + 1, f), n, k, i))
            elif k + 1 < len(frases) and frases[k + 1].cluster != e.dirigente:
                v = _vocativo_final(pal, frases, k, e.dirigente)
                if v and not any(c[2] >= k - 3 and c[0] > v[1] - 1 for c in out[-1:]):
                    out.append((_pula_por_favor(pal, v[1] + 1, f), v[0], k, v[1]))
            continue
        # chamada no começo de turno alheio: "Irmão Luiz, porque..." / "Janice, aqui..."
        if k > 0 and frases[k - 1].cluster == e.dirigente:
            ws = [pal[i] for i in f.idx()][:3]
            for j, w in enumerate(ws):
                if w.texto.strip().endswith(","):
                    nome = re.sub(r"[^\w]", "", w.texto)
                    titulo = j == 0 or _limpa(ws[j - 1].texto) in ("irmao", "irma")
                    corr = _irmaos.corrige(nome, nm.entradas) if nm.entradas else None
                    if (
                        titulo
                        and nome[:1].isupper()
                        and not _nomes._nao_e_nome(nome)
                        and (corr is None or corr.casou)
                    ):
                        out.append((f.i0 + j + 1, nome, k, f.i0 + j))
                    break
    return out


def trecho_do_chamado(
    pal: list[Palavra],
    frases: list[Frase],
    ini: int,
    e: Estrutura,
    leitura: set[int],
    inicios_chamada: set[int],
) -> list[int]:
    """Palavras do chamado: de `ini` até o dirigente voltar."""
    out: list[int] = []
    j = ini
    primeira = True
    while j < len(pal) and pal[j].inicio < e.fim:
        k = pal[j].frase
        f = frases[k]
        if j == f.i0:
            if (
                _POR_FAVOR.match(_limpa(_texto(pal, f)))
                and f.cluster == e.dirigente
                and not out
            ):
                j = f.i1 + 1
                continue
            if _e_retorno(pal, f, e.dirigente, leitura, k) or (
                not primeira and j in inicios_chamada
            ):
                break
            if out and pal[j].inicio - pal[out[-1]].fim > RETORNO_MAX_PAUSA:
                break
            if out and pal[j].inicio - pal[out[0]].inicio > TRECHO_MAX:
                break
        if j in inicios_chamada and not primeira:
            break
        primeira = False
        out.append(j)
        j += 1
    return out


def agradecido_em(pal: list[Palavra], frases: list[Frase], k: int) -> Optional[str]:
    """Nome agradecido no começo da fala do dirigente que começa na frase k."""
    achado = _agradecimento_em(pal, frases, k)
    return achado[0] if achado else None


def _agradecimento_em(
    pal: list[Palavra], frases: list[Frase], k: int
) -> Optional[tuple[str, int]]:
    """(nome agradecido, palavra do nome) no começo da fala que começa na frase k."""
    if k >= len(frases):
        return None
    f0 = frases[k]
    for f in frases[k : k + 3]:
        if pal[f.i0].inicio - pal[f0.i0].inicio > AGRADECIMENTO_JANELA:
            break
        for t, n, i in f.eventos:
            if t == "agradecimento" and i - f0.i0 <= AGRADECIMENTO_PALAVRAS:
                return n, i
    return None


# --------------------------------------------------------------------------
# (V) Troca de voz dentro do trecho do chamado
# --------------------------------------------------------------------------
def corta_por_voz(
    pal: list[Palavra], frases: list[Frase], ws: list[int]
) -> list[list[int]]:
    """Divide o trecho onde a voz muda de verdade.

    Referência: frases dos primeiros VOZ_REF_SECONDS s do trecho. Corta no início
    da 1ª frase com cosseno < VOZ_TROCA quando a frase seguinte com voz também
    fica abaixo de VOZ_TROCA + 0,1 (ou não existe)."""
    ks = list(dict.fromkeys(pal[i].frase for i in ws))
    ref: list[np.ndarray] = []
    t_ref: Optional[float] = None
    for k in ks:
        f = frases[k]
        if f.voz is None:
            continue
        if t_ref is None:
            t_ref = pal[f.i0].inicio
        if pal[f.i0].inicio - t_ref <= VOZ_REF_SECONDS or not ref:
            ref.append(f.voz)
    if not ref or t_ref is None:
        return [ws]
    r = np.mean(ref, axis=0)
    r = r / (np.linalg.norm(r) + 1e-9)
    baixos = [
        k
        for k in ks
        if frases[k].voz is not None
        and pal[frases[k].i0].inicio - t_ref > VOZ_REF_SECONDS
        and float(frases[k].voz @ r) < VOZ_TROCA
    ]
    for k in baixos:
        seguintes = [x for x in ks if x > k and frases[x].voz is not None]
        if not seguintes or float(frases[seguintes[0]].voz @ r) < VOZ_TROCA + 0.1:
            corte = frases[k].i0
            a = [i for i in ws if i < corte]
            b = [i for i in ws if i >= corte]
            if a and b:
                return [a] + corta_por_voz(pal, frases, b)
    return [ws]


# --------------------------------------------------------------------------
# (G) Dúvidas com a mesma voz
# --------------------------------------------------------------------------
def agrupa_duvidas(
    pal: list[Palavra],
    dono: dict[int, Dono],
    embed: Callable[[float, float], np.ndarray],
) -> int:
    """Trechos "[?]" com voz parecida passam a ter o mesmo rótulo.

    Nunca junta com pessoa nomeada: o nome exibido continua "[?]". Trecho com
    menos de GRUPO_DUVIDA_MIN_SECONDS s fica sozinho. Devolve quantos juntou."""
    grupos: dict[str, list[int]] = {}
    for i, d in dono.items():
        if (
            d.rotulo
            and d.rotulo.startswith(_PREFIXO_ROTULO_DUVIDA)
            and d.nome == DUVIDA
        ):
            grupos.setdefault(d.rotulo, []).append(i)
    vozes: dict[str, np.ndarray] = {}
    for r, ws in grupos.items():
        a, b = pal[min(ws)].inicio, pal[max(ws)].fim
        if b - a >= GRUPO_DUVIDA_MIN_SECONDS:
            vozes[r] = embed(a, b)
    final: list[tuple[str, list[np.ndarray]]] = []
    troca: dict[str, str] = {}
    for r in sorted(vozes, key=lambda r: pal[min(grupos[r])].inicio):
        v = vozes[r]
        melhor, sim = None, -1.0
        for k, (_, vs) in enumerate(final):
            c = np.mean(vs, axis=0)
            c = c / (np.linalg.norm(c) + 1e-9)
            s = float(c @ v)
            if s > sim:
                melhor, sim = k, s
        if melhor is not None and sim >= GRUPO_DUVIDA:
            final[melhor][1].append(v)
            troca[r] = final[melhor][0]
        else:
            final.append((r, [v]))
    for i, d in list(dono.items()):
        if d.rotulo in troca:
            dono[i] = Dono(
                troca[d.rotulo],
                d.nome,
                d.origem + "; voz agrupada com outro [?]",
                d.exibido,
            )
    return len(troca)


# --------------------------------------------------------------------------
# Montagem
# --------------------------------------------------------------------------
def monta(
    frases_diar: Sequence, segmentos: Sequence[Segment], nomes_base: dict[str, str]
) -> tuple[list[Palavra], list[Frase]]:
    """Palavras e frases a partir da diarização e dos blocos de base.

    Args:
        frases_diar: frases da diarização (``diarize.Sentence``: ``words``,
            ``speaker`` e ``embedding``), em ordem de tempo.
        segmentos: blocos de base (saída de ``turnos.reatribui``, sem coro).
        nomes_base: rótulo de base -> nome exibido ("" se sem nome).
    """
    segs = [s for s in segmentos if s.speaker != CORO_LABEL]
    inicios = [s.start for s in segs]
    pal: list[Palavra] = []
    frases: list[Frase] = []
    for s in frases_diar:
        if not s.words:
            continue
        i0 = len(pal)
        for w in s.words:
            meio = (w.start + w.end) / 2
            j = max(0, bisect.bisect_right(inicios, meio) - 1) if segs else -1
            rot = segs[j].speaker if j >= 0 else None
            pal.append(
                Palavra(
                    w.start,
                    w.end,
                    w.text,
                    len(frases),
                    int(s.speaker),
                    j,
                    rot,
                    nomes_base.get(rot or "", ""),
                )
            )
        frases.append(
            Frase(i0, len(pal) - 1, int(s.speaker), getattr(s, "embedding", None))
        )
    return pal, frases


def _exibe_sugestao(
    pal: list[Palavra], cru: str, pos: int, nm: _Nomes, nome: str
) -> Optional[str]:
    """Nome exibido para a sugestão de uma menção só (None = "[?] nome?").

    Sai só "[?]" quando o candidato não tem cara de nome: palavra bloqueada
    (comum ou bíblica); correção por semelhança que troca a letra inicial
    ("Ana" -> "Deana"); ou nenhuma evidência de pessoa: sem "irmão/irmã",
    sem "por favor" junto, sem autoapresentação e sem semelhança com a lista
    ou os participantes. Nome composto com 2ª parte desconhecida fica só com a
    1ª ("Fernando Cross" -> "Fernando")."""
    partes = cru.split()
    if not partes or any(_irmaos.fold(x) in _BLOQUEIO_SUGESTAO for x in partes):
        return DUVIDA
    if any(_nomes._nao_e_nome(x) for x in partes):
        return DUVIDA
    mostrado = nome.split()
    if mostrado and _irmaos.fold(mostrado[0])[:1] != _irmaos.fold(partes[0])[:1]:
        return DUVIDA
    evidencia = nm.parecido(cru) or nm.parecido(partes[0])
    if 0 <= pos < len(pal):
        ini = max(0, pos - (len(partes) - 1))
        antes = [_limpa(pal[j].texto) for j in range(max(0, ini - 4), ini)]
        depois = [_limpa(pal[j].texto) for j in range(pos + 1, min(len(pal), pos + 3))]
        if antes and antes[-1] in _TITULOS:
            evidencia = True
        if _APRESENTACAO.search(" ".join(antes)):
            evidencia = True
        if " ".join(antes[-2:]) in _POR_FAVOR_PALAVRAS or (
            " ".join(depois) in _POR_FAVOR_PALAVRAS
        ):
            evidencia = True
    if not evidencia:
        return DUVIDA
    if len(partes) > 1 and len(mostrado) > 1 and not nm.parecido(partes[-1]):
        return f"{DUVIDA} {mostrado[0]}?"
    return None


def _donos(
    pal: list[Palavra],
    frases: list[Frase],
    e: Estrutura,
    nm: _Nomes,
    embed: Callable[[float, float], np.ndarray],
    info: dict,
) -> dict[int, Dono]:
    """Dono de cada palavra da região do estudo (a + d + L + V + G)."""
    reg = [i for i, p in enumerate(pal) if e.inicio <= p.inicio < e.fim]
    leitura = leituras(pal, frases, e)
    d_rot = Counter(
        pal[i].rotulo_base for i in reg if pal[i].cluster == e.dirigente
    ).most_common(1)[0][0]
    d_nome = next(pal[i].nome_base for i in reg if pal[i].rotulo_base == d_rot)
    l_idx = [i for k in sorted(leitura) for i in frases[k].idx()]
    l_rot = (
        Counter(pal[i].rotulo_base for i in l_idx).most_common(1)[0][0]
        if l_idx
        else None
    )
    l_nome = next((pal[i].nome_base for i in l_idx if pal[i].rotulo_base == l_rot), "")
    l_cl = (
        Counter(pal[i].cluster for i in l_idx).most_common(1)[0][0] if l_idx else None
    )
    # pessoa já nomeada fora do estudo (ex.: presidente que comenta): mesmo rótulo
    fora: dict[str, str] = {}
    for p in pal:
        if not (e.inicio <= p.inicio < e.fim) and p.nome_base and p.rotulo_base:
            fora.setdefault(p.nome_base, p.rotulo_base)
    novos: dict[str, str] = {}

    def rot_de(nome: str) -> str:
        if d_nome and nome == d_nome:
            return d_rot
        if l_rot and l_nome and nome == l_nome:
            return l_rot
        if nome in fora:
            return fora[nome]
        return novos.setdefault(nome, f"{_PREFIXO_ROTULO_NOME}{len(novos) + 1}")

    dono: dict[int, Dono] = {}
    for i in reg:
        p = pal[i]
        f = frases[p.frase]
        if p.frase in leitura:
            dono[i] = Dono(l_rot, l_nome, "leitura")
        elif p.cluster == e.dirigente:
            dono[i] = Dono(d_rot, d_nome, "dirigente")
        elif _RETORNO.match(_limpa(_texto(pal, f))) and any(
            t == "agradecimento" and j - f.i0 <= RETORNO_AGRADECE_PALAVRAS
            for t, _, j in f.eventos
        ):
            dono[i] = Dono(d_rot, d_nome, "agradecimento com nome (fora do cluster)")
    chs = chamadas(pal, frases, e, nm)
    inicios = {c[0] for c in chs}
    # palavras da chamada no começo de turno alheio voltam para o dirigente
    for ini, _, k, _pos in chs:
        f = frases[k]
        if f.cluster != e.dirigente:
            for i in range(f.i0, ini):
                dono[i] = Dono(d_rot, d_nome, "chamada no início do turno")
    trechos: list[tuple[list[int], Optional[str], int]] = []
    for ini, nome, _, pos in chs:
        ws = trecho_do_chamado(pal, frases, ini, e, leitura, inicios)
        if ws:
            for n_parte, parte in enumerate(corta_por_voz(pal, frases, ws)):
                trechos.append((parte, nome if n_parte == 0 else None, pos))
    info["cortes_voz_trecho"] = sum(1 for _, n, _ in trechos if n is None)
    for ws, nome, pos in trechos:
        if nome is None:  # parte depois da troca de voz: não é o chamado
            dentro = set(ws)
            for k in dict.fromkeys(pal[i].frase for i in ws):
                f = frases[k]
                dur = pal[f.i1].fim - pal[f.i0].inicio
                if f.cluster == e.dirigente:
                    v = Dono(d_rot, d_nome, "troca de voz: cluster do dirigente")
                elif l_cl is not None and f.cluster == l_cl and dur >= LEITURA_MIN:
                    v = Dono(l_rot, l_nome, "troca de voz: leitura")
                else:
                    v = Dono(None, "", "troca de voz dentro do trecho do chamado")
                for i in f.idx():
                    if i in dentro:
                        dono[i] = v
            continue
        k_ret = frases[pal[ws[-1]].frase].i1 + 1  # 1ª palavra depois do trecho
        agr = agradecido_em(pal, frases, pal[k_ret].frase) if k_ret < len(pal) else None
        n, amb = nm.corrige(nome)
        origem = "chamada do dirigente"
        confirmado = False
        nome_final: Optional[str] = n
        if agr:
            na, _ = nm.corrige(agr)
            if nm.mesmo(n, na):
                origem, confirmado = "chamada + agradecimento", True
            else:
                nome_final, origem = None, f"chamada {n} ≠ agradecimento {na}"
        fraco = nome_final is not None and not confirmado and not nm.conhecido(nome)
        for i in ws:
            if nome_final is None:
                dono[i] = Dono("?", DUVIDA, origem)
            elif fraco:
                dono[i] = Dono(
                    "?",
                    f"{DUVIDA} {nome_final}?",
                    origem + "; fora da lista, 1 menção",
                    _exibe_sugestao(pal, nome, pos, nm, nome_final),
                )
            elif amb:
                dono[i] = Dono("?", f"{DUVIDA} {nome_final} (homônimo)", origem)
            else:
                dono[i] = Dono(rot_de(nome_final), nome_final, origem)
    # corridas sem dono: agradecimento logo depois, senão "[?]"
    regl = sorted(reg)
    k = 0
    while k < len(regl):
        atual = dono.get(regl[k])
        if atual is not None and atual.rotulo is not None:
            k += 1
            continue
        run: list[int] = []
        while k < len(regl) and (
            dono.get(regl[k]) is None or dono[regl[k]].rotulo is None
        ):
            run.append(regl[k])
            k += 1
        nxt = run[-1] + 1
        agr = None
        pos_agr = -1
        if (
            nxt < len(pal)
            and pal[nxt].inicio - pal[run[-1]].fim <= CORRIDA_MAX_PAUSA
            and frases[pal[nxt].frase].cluster == e.dirigente
        ):
            achado = _agradecimento_em(pal, frases, pal[nxt].frase)
            if achado:
                agr, pos_agr = achado
        if agr:
            n, amb = nm.corrige(agr)
            fraco = not nm.conhecido(agr)
            for x in run:
                if fraco:
                    dono[x] = Dono(
                        "?",
                        f"{DUVIDA} {n}?",
                        "agradecimento; fora da lista, 1 menção",
                        _exibe_sugestao(pal, agr, pos_agr, nm, n),
                    )
                elif amb:
                    dono[x] = Dono("?", f"{DUVIDA} {n} (homônimo)", "agradecimento")
                else:
                    dono[x] = Dono(rot_de(n), n, "agradecimento")
        else:
            for x in run:
                dono[x] = Dono("?", DUVIDA, "sem chamada/agradecimento")
    # rótulo próprio para cada trecho em dúvida (Q1, Q2...)
    duvidas = 0
    ult: Optional[tuple[str, int, str]] = None
    for i in sorted(dono):
        d = dono[i]
        if d.rotulo == "?":
            if ult is None or ult[0] != d.nome or i - ult[1] > 1:
                duvidas += 1
                ult = (d.nome, i, f"{_PREFIXO_ROTULO_DUVIDA}{duvidas}")
            else:
                ult = (d.nome, i, ult[2])
            dono[i] = Dono(ult[2], d.nome, d.origem, d.exibido)
        else:
            ult = None
    info["duvidas"] = duvidas
    info["duvidas_agrupadas"] = agrupa_duvidas(pal, dono, embed)
    info["dirigente_rotulo"] = d_rot
    info["leitor_rotulo"] = l_rot
    return dono


def _blocos(
    pal: list[Palavra], segmentos: list[Segment], e: Estrutura, dono: dict[int, Dono]
) -> list[Segment]:
    """Blocos finais: os de base fora do estudo, refeitos por palavra dentro."""
    segs = [s for s in segmentos if s.speaker != CORO_LABEL]
    dentro = {i for i, p in enumerate(pal) if e.inicio <= p.inicio < e.fim}
    tocados = {pal[i].seg for i in dentro}
    out: list[Segment] = [s for j, s in enumerate(segs) if j not in tocados]
    # pedaços fora da região de blocos de base que cruzam a borda
    pedaco: list[int] = []

    def fecha() -> None:
        if pedaco:
            p0, p1 = pal[pedaco[0]], pal[pedaco[-1]]
            out.append(
                Segment(
                    p0.inicio,
                    p1.fim,
                    segs[p0.seg].speaker,
                    "".join(pal[i].texto for i in pedaco).strip(),
                )
            )
            pedaco.clear()

    for i, p in enumerate(pal):
        if p.seg in tocados and i not in dentro:
            if pedaco and (pal[pedaco[-1]].seg != p.seg or i != pedaco[-1] + 1):
                fecha()
            pedaco.append(i)
        else:
            fecha()
    fecha()
    # região: palavras seguidas do mesmo rótulo, pausa até UTTERANCE_MAX_GAP
    atual: Optional[list] = None
    for i in sorted(dentro):
        p, d = pal[i], dono[i]
        if atual and atual[2] == d.rotulo and p.inicio - atual[1] <= UTTERANCE_MAX_GAP:
            atual[1] = max(atual[1], p.fim)
            atual[3].append(p.texto)
        else:
            if atual:
                out.append(
                    Segment(atual[0], atual[1], atual[2], "".join(atual[3]).strip())
                )
            atual = [p.inicio, p.fim, d.rotulo, [p.texto]]
    if atual:
        out.append(Segment(atual[0], atual[1], atual[2], "".join(atual[3]).strip()))
    out = [s for s in out if s.text]
    out.sort(key=lambda s: s.start)
    return out


def _pessoas(
    dono: dict[int, Dono],
    segs_finais: list[Segment],
    pessoas_base: Sequence[_turnos.Pessoa],
    info: dict,
) -> list[_turnos.Pessoa]:
    """Pessoas para o worker: as de base que continuam + as novas do estudo."""
    fixos = {
        r: n
        for r, n in (
            (info.get("dirigente_rotulo"), "Dirigente"),
            (info.get("leitor_rotulo"), "Leitor"),
        )
        if r
    }
    usados = {s.speaker for s in segs_finais}
    por_rotulo = {p.rotulo: p for p in pessoas_base}
    seg_por_rotulo: dict[str, float] = {}
    for s in segs_finais:
        seg_por_rotulo[s.speaker] = seg_por_rotulo.get(s.speaker, 0.0) + (
            s.end - s.start
        )
    # trechos por rótulo (sequências de palavras com o mesmo dono)
    trechos: dict[str, list[tuple[str, str]]] = {}
    ant: Optional[str] = None
    for i in sorted(dono):
        d = dono[i]
        if d.rotulo != ant:
            trechos.setdefault(d.rotulo or "", []).append((d.mostra, d.origem))
        ant = d.rotulo
    papeis_fixos = {"dirigente", "leitor"}
    out: list[_turnos.Pessoa] = []
    for p in pessoas_base:
        if p.rotulo not in usados:
            continue
        ts = trechos.get(p.rotulo, [])
        if ts and p.papel not in papeis_fixos and "comentarista" not in p.papel:
            p = replace(
                p,
                papel=f"{p.papel} + comentarista",
                comentarios=p.comentarios + len(ts),
            )
        out.append(p)
    for rot, ts in trechos.items():
        if not rot or rot in por_rotulo or rot not in usados:
            continue
        segundos = round(seg_por_rotulo.get(rot, 0.0), 1)
        if rot in fixos:  # dirigente/leitor sem pessoa de base: sem nome
            out.append(
                _turnos.Pessoa(
                    rot,
                    fixos[rot],
                    fixos[rot].lower(),
                    "",
                    len(ts),
                    segundos,
                    nomeada=False,
                )
            )
            continue
        nome = ts[0][0]
        origens = Counter(o for _, o in ts)
        origem = ", ".join(f"{o} {n}×" for o, n in origens.items())
        duvida = e_duvida(nome) or not nome
        out.append(
            _turnos.Pessoa(
                rot,
                nome or DUVIDA,
                "comentarista",
                origem,
                len(ts),
                segundos,
                nomeada=not duvida,
                duvida=duvida,
            )
        )
    return out


def aplica(
    frases_diar: Sequence,
    segmentos: Sequence[Segment],
    pessoas_base: Sequence[_turnos.Pessoa],
    embed: Callable[[float, float], np.ndarray],
    entradas: Sequence = (),
    participantes: Sequence[str] = (),
) -> Resultado:
    """Aplica a estrutura do estudo por palavra, se a guarda deixar.

    Args:
        frases_diar: frases da diarização (com ``words``, ``speaker`` e
            ``embedding``), em ordem de tempo.
        segmentos: blocos de ``turnos.reatribui`` (sem coro).
        pessoas_base: pessoas de ``turnos.reatribui``.
        embed: ``embed(inicio, fim)`` devolve a voz normalizada de um trecho.
        entradas: lista de irmãos (só nomes).
        participantes: nomes informados no job (grafia já corrigida).

    Returns:
        Resultado. Com a guarda desligada, os blocos e as pessoas voltam como
        vieram e ``ativa`` é False.
    """
    segs = sorted(
        (s for s in segmentos if s.speaker != CORO_LABEL), key=lambda s: s.start
    )
    nomes_base = {p.rotulo: p.nome for p in pessoas_base if p.nomeada}
    pal, frases = monta(frases_diar, segs, nomes_base)
    if not pal:
        return Resultado(
            False,
            list(segmentos),
            list(pessoas_base),
            {"ativa": False, "motivo": "sem palavras"},
        )
    e = estrutura(pal, frases)
    info: dict = {
        "ativa": e.ativa,
        "chamadas": e.chamadas,
        "paragrafos": e.paragrafos,
        "motivo": e.motivo,
    }
    if not e.ativa:
        log.info("Estudo por palavra desligado: %s", e.motivo)
        return Resultado(False, list(segmentos), list(pessoas_base), info)
    info["regiao"] = [round(e.inicio, 1), round(e.fim, 1)]
    nm = _Nomes(entradas, participantes)
    dono = _donos(pal, frases, e, nm, embed, info)
    finais = _blocos(pal, segs, e, dono)
    pessoas = _pessoas(dono, finais, pessoas_base, info)
    info["nomeados"] = sorted(
        {d.nome for d in dono.values() if d.nome and not e_duvida(d.nome)}
    )
    log.info(
        "Estudo por palavra: %d chamadas, %d anúncios, região %.0f–%.0f s, "
        "%d trechos em dúvida",
        e.chamadas,
        e.paragrafos,
        e.inicio,
        e.fim,
        info.get("duvidas", 0),
    )
    return Resultado(True, finais, pessoas, info)
