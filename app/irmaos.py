"""Irmãos / nomes conhecidos: a lista de nomes da congregação.

Guarda só NOMES, sem biometria, em <dados>/nomes/nomes.json, neste computador.
Cada entrada tem um `id`, um `observacao` opcional que distingue homônimos
("Matheus (1)", "Matheus (2)") e `grafias`, variantes que o Whisper escreve
e que a semelhança de letras não alcança ("Marileneza" para "Marines").

Usos:
- hotwords do Whisper, em frase pontuada (nunca lista solta de nomes);
- correção de grafia dos nomes extraídos da conversa (nomes.py e turnos.py).
  Nome EXATO da lista não se funde com outro nome da lista: Mateus ≠ Matheus;
- sugestões no campo de nome da revisão (datalist).

Nada daqui é aprendido sozinho: a lista só muda pelo que o usuário digita.
"""
from __future__ import annotations

import json
import re
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Optional, Sequence

from . import arquivo_seguro, config

LIMITE_CORRECAO = 0.80     # "Niffa" ~ "Ninfa" = 0,80; "Admir" ~ "Nadir" = 0,80 (ver docs)
MARGEM_EMPATE = 0.02       # dois candidatos quase iguais: não corrige
MAX_ENTRADAS = 500
MAX_GRAFIAS = 10
MAX_NOME = 60
MAX_OBSERVACAO = 80          # mesmo limite do campo da API (main.IrmaoIn)
AMBIGUO_MSG = "{n} pessoas com esse nome; confirme pela voz"

_LETRAS = "A-Za-zÀ-ÖØ-öø-ÿ"
_NOME_RX = re.compile(rf"^[{_LETRAS}][{_LETRAS}'’ -]{{1,{MAX_NOME - 1}}}$")
_OBS_RX = re.compile(rf"^[{_LETRAS}0-9 ()ºª.,/'’-]{{0,{MAX_OBSERVACAO}}}$")
_CTRL = re.compile(r"[\x00-\x1f\x7f]")

_lock = threading.RLock()


class IrmaoError(ValueError):
    """Entrada inválida ou lista ilegível. A mensagem vai para a tela."""


@dataclass(frozen=True)
class Entrada:
    id: str
    nome: str
    observacao: str = ""
    grafias: tuple[str, ...] = ()
    atualizado: float = 0.0

    @property
    def rotulo(self) -> str:
        """Como aparece nas listas: "Matheus (1)" quando há observação."""
        return f"{self.nome} ({self.observacao})" if self.observacao else self.nome

    def to_dict(self) -> dict:
        return {"id": self.id, "nome": self.nome, "observacao": self.observacao,
                "grafias": list(self.grafias), "atualizado": self.atualizado,
                "rotulo": self.rotulo}


@dataclass(frozen=True)
class Correcao:
    """Resultado de `corrige`. `casou` False = o nome não está na lista."""
    nome: str
    casou: bool
    origem: str = ""            # "exato", "grafia" ou "parecido"
    ambiguo: bool = False       # mais de uma pessoa com esse nome
    observacao: str = ""        # observação da entrada, ou aviso de ambiguidade


# --------------------------------------------------------------------------
# Texto
# --------------------------------------------------------------------------
def fold(texto: str) -> str:
    """Caixa e acento ignorados: "Mateus" == "mateus" == "Mateús"."""
    return "".join(c for c in unicodedata.normalize("NFD", texto.casefold())
                   if unicodedata.category(c) != "Mn")


def _limpa(valor: object) -> str:
    return re.sub(r"\s+", " ", _CTRL.sub(" ", str(valor or ""))).strip()


def _valida(nome: object, observacao: object, grafias: Iterable[object]) -> tuple[str, str, tuple[str, ...]]:
    nome, obs = _limpa(nome), _limpa(observacao)
    if not _NOME_RX.match(nome):
        raise IrmaoError("Nome: use de 2 a 60 letras (acentos, espaço, hífen ou apóstrofo).")
    if not _OBS_RX.match(obs):
        raise IrmaoError("Observação: até 80 caracteres (letras, números e ( ) , . / -).")
    vistas: list[str] = []
    for g in grafias or []:
        g = _limpa(g)
        if not g:
            continue
        if not _NOME_RX.match(g):
            raise IrmaoError(f"Grafia '{g}': use só letras, espaço, hífen ou apóstrofo.")
        if fold(g) not in {fold(v) for v in vistas}:
            vistas.append(g)
    if len(vistas) > MAX_GRAFIAS:
        raise IrmaoError(f"No máximo {MAX_GRAFIAS} grafias por nome.")
    return nome, obs, tuple(vistas)


# --------------------------------------------------------------------------
# Armazenamento
# --------------------------------------------------------------------------
def _file() -> Path:
    return config.DATA_DIR / "nomes" / "nomes.json"


def _de_dict(d: dict) -> Entrada:
    return Entrada(id=str(d["id"]), nome=str(d["nome"]),
                   observacao=str(d.get("observacao", "")),
                   grafias=tuple(str(g) for g in d.get("grafias", [])),
                   atualizado=float(d.get("atualizado", 0.0)))


def _ler() -> list[Entrada]:
    f = _file()
    if not f.exists():
        return []
    try:
        data = json.loads(f.read_text(encoding="utf-8-sig"))
        return [_de_dict(d) for d in data.get("nomes", [])]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise IrmaoError("A lista de nomes (nomes.json) está ilegível. "
                         "Corrija ou apague o arquivo.") from exc


def _gravar(entradas: list[Entrada]) -> None:
    corpo = {"versao": 1, "nomes": [
        {"id": e.id, "nome": e.nome, "observacao": e.observacao,
         "grafias": list(e.grafias), "atualizado": e.atualizado} for e in entradas]}
    # Windows: o arquivo pode estar aberto em outro programa; tenta de novo e,
    # se não der, levanta ArquivoBloqueado (a API responde 409 com a mensagem).
    arquivo_seguro.gravar_texto(_file(), json.dumps(corpo, ensure_ascii=False, indent=2))


def listar() -> list[Entrada]:
    """Todas as entradas, por nome. Lista ilegível levanta IrmaoError."""
    with _lock:
        return sorted(_ler(), key=lambda e: (fold(e.nome), fold(e.observacao)))


def _duplicada(entradas: list[Entrada], nome: str, obs: str, ignorar: Optional[str]) -> bool:
    return any(e.id != ignorar and fold(e.nome) == fold(nome) and fold(e.observacao) == fold(obs)
               for e in entradas)


def criar(nome: object, observacao: object = "", grafias: Sequence[object] = ()) -> Entrada:
    nome, obs, gr = _valida(nome, observacao, grafias)
    with _lock:
        entradas = _ler()
        if len(entradas) >= MAX_ENTRADAS:
            raise IrmaoError(f"A lista já tem {MAX_ENTRADAS} nomes.")
        if _duplicada(entradas, nome, obs, None):
            raise IrmaoError("Já existe um nome igual, com a mesma observação.")
        nova = Entrada(id=uuid.uuid4().hex[:12], nome=nome, observacao=obs,
                       grafias=gr, atualizado=time.time())
        _gravar(entradas + [nova])
        return nova


def atualizar(id_: str, nome: object, observacao: object = "",
              grafias: Sequence[object] = ()) -> Optional[Entrada]:
    """Entrada atualizada, ou None se o id não existe."""
    nome, obs, gr = _valida(nome, observacao, grafias)
    with _lock:
        entradas = _ler()
        alvo = next((e for e in entradas if e.id == id_), None)
        if alvo is None:
            return None
        if _duplicada(entradas, nome, obs, id_):
            raise IrmaoError("Já existe um nome igual, com a mesma observação.")
        nova = Entrada(id=id_, nome=nome, observacao=obs, grafias=gr, atualizado=time.time())
        _gravar([nova if e.id == id_ else e for e in entradas])
        return nova


def apagar(id_: str) -> bool:
    with _lock:
        entradas = _ler()
        restantes = [e for e in entradas if e.id != id_]
        if len(restantes) == len(entradas):
            return False
        _gravar(restantes)
        return True


def listar_seguro(log=None) -> list[Entrada]:
    """Para o motor e a conversa: lista ilegível vira lista vazia (sem derrubar a reunião)."""
    try:
        return listar()
    except IrmaoError as exc:
        if log is not None:
            log.warning("Lista de nomes ignorada: %s", exc)
        return []


# --------------------------------------------------------------------------
# Correção de grafia
# --------------------------------------------------------------------------
def _grupos(entradas: Sequence[Entrada]) -> dict[str, list[Entrada]]:
    """fold(nome) -> entradas com esse nome (mais de uma = homônimos)."""
    grupos: dict[str, list[Entrada]] = {}
    for e in entradas:
        grupos.setdefault(fold(e.nome), []).append(e)
    return grupos


def _do_grupo(grupo: list[Entrada], origem: str) -> Correcao:
    if len(grupo) == 1:
        e = grupo[0]
        return Correcao(e.nome, True, origem, False, e.observacao)
    return Correcao(grupo[0].nome, True, origem, True,
                    AMBIGUO_MSG.format(n=len(grupo)))


def corrige(cru: str, entradas: Sequence[Entrada]) -> Correcao:
    """Nome da lista para a palavra extraída da conversa.

    1) exato (caixa e acento ignorados): vale o nome da lista;
    2) grafia cadastrada da entrada: "Marileneza" -> "Marines";
    3) semelhança >= LIMITE_CORRECAO com o nome (ou com uma parte dele, como
       "Eduardo" para "José Eduardo"). Empate entre dois nomes: não corrige.
    Nome sem casamento volta como veio (`casou` False).
    """
    key = fold(_limpa(cru))
    if not key or not entradas:
        return Correcao(cru, False)
    grupos = _grupos(entradas)
    if key in grupos:
        return _do_grupo(grupos[key], "exato")
    for e in entradas:
        if any(fold(g) == key for g in e.grafias):
            return _do_grupo(grupos[fold(e.nome)], "grafia")
    if len(key) < 3:
        return Correcao(cru, False)
    notas: dict[str, float] = {}
    for chave, grupo in grupos.items():
        alvos = {chave, *chave.split()}
        for e in grupo:
            alvos.update(fold(g) for g in e.grafias)
        notas[chave] = max(SequenceMatcher(None, key, a).ratio() for a in alvos)
    ordem = sorted(notas.items(), key=lambda kv: -kv[1])
    melhor_chave, melhor = ordem[0]
    if melhor < LIMITE_CORRECAO:
        return Correcao(cru, False)
    if len(ordem) > 1 and ordem[1][1] >= LIMITE_CORRECAO and melhor - ordem[1][1] < MARGEM_EMPATE:
        return Correcao(cru, False)
    return _do_grupo(grupos[melhor_chave], "parecido")


def nomes_protegidos(entradas: Iterable[Entrada]) -> set[str]:
    """Nomes EXATOS da lista (folded). Dois nomes protegidos diferentes nunca se fundem."""
    return {fold(e.nome) for e in entradas}


def ausentes(nomes_confirmados: Iterable[str], entradas: Sequence[Entrada]) -> list[str]:
    """Nomes confirmados na revisão que ainda não estão na lista (para oferecer ao usuário)."""
    conhecidos = set()
    for e in entradas:
        conhecidos.add(fold(e.nome))
        conhecidos.update(fold(g) for g in e.grafias)
    saida: list[str] = []
    for n in nomes_confirmados:
        n = _limpa(n)
        if n and fold(n) not in conhecidos and n not in saida:
            saida.append(n)
    return saida


# --------------------------------------------------------------------------
# Hotwords
# --------------------------------------------------------------------------
def _frase(nomes: list[str]) -> str:
    if len(nomes) == 1:
        corpo = nomes[0]
    else:
        corpo = ", ".join(nomes[:-1]) + " e " + nomes[-1]
    return f"Participam irmãos como {corpo}."


def frase_irmaos(entradas: Sequence[Entrada], limite: int,
                 excluir: Iterable[str] = ()) -> str:
    """Frase pontuada com os nomes mais recentes que cabem em `limite` caracteres.

    Nunca corta no meio: um nome que não cabe é pulado e a frase fecha com ponto.
    `excluir`: nomes já presentes em outra parte do texto (não repete)."""
    excl = {fold(x) for x in excluir}
    escolhidos: list[str] = []
    vistos: set[str] = set()
    for e in sorted(entradas, key=lambda x: -x.atualizado):
        k = fold(e.nome)
        if k in excl or k in vistos:
            continue
        cand = _frase([*escolhidos, e.nome])
        if len(cand) > limite:
            continue
        escolhidos.append(e.nome)
        vistos.add(k)
    return _frase(escolhidos) if escolhidos else ""


# --------------------------------------------------------------------------
# Participantes digitados
# --------------------------------------------------------------------------
_PALAVRA = re.compile(r"[^\W\d_][\w'’-]*")


def mapa_grafias(entradas: Sequence[Entrada]) -> dict[str, str]:
    """Grafia cadastrada (sem caixa e acento) -> nome certo da lista.

    Fica de fora a grafia que é o nome de outra entrada ou que aparece em duas
    entradas com nomes diferentes: aí não há como saber quem é."""
    nomes = {fold(e.nome) for e in entradas}
    mapa: dict[str, str] = {}
    conflito: set[str] = set()
    for e in entradas:
        for g in e.grafias:
            k = fold(_limpa(g))
            if not k or k in nomes:
                continue
            if k in mapa and mapa[k] != e.nome:
                conflito.add(k)
            mapa.setdefault(k, e.nome)
    return {k: v for k, v in mapa.items() if k not in conflito}


def corrige_texto(texto: str, entradas: Sequence[Entrada]) -> str:
    """Troca no texto a palavra que É uma grafia cadastrada pelo nome da lista.

    Só palavra com maiúscula ("Módemir" -> "Ademir"); caixa e acento não
    contam. Palavra parecida que não está cadastrada fica como está. Grafia
    de mais de uma palavra é trocada pela frase inteira."""
    mapa = mapa_grafias(entradas)
    if not mapa or not texto:
        return texto
    for k, nome in mapa.items():
        if " " in k:
            texto = re.sub(r"(?<![\w])" + r"\s+".join(re.escape(x) for x in k.split())
                           + r"(?![\w])", nome, texto, flags=re.IGNORECASE)

    def troca(m: re.Match) -> str:
        w = m.group(0)
        return mapa.get(fold(w), w) if w[:1].isupper() else w

    return _PALAVRA.sub(troca, texto)


def corrige_participante(nome: str, entradas: Sequence[Entrada]) -> Optional[str]:
    """Nome da lista para um participante digitado com erro, ou None.

    Compara o nome INTEIRO (ou uma grafia cadastrada). Parte do nome não conta:
    "Eduardo" não vira "José Eduardo". Homônimos na lista não corrigem."""
    key = fold(_limpa(nome))
    if not key or not entradas:
        return None
    grupos = _grupos(entradas)
    if key in grupos:
        g = grupos[key]
        return g[0].nome if len(g) == 1 else None
    for e in entradas:
        if any(fold(g) == key for g in e.grafias):
            grupo = grupos[fold(e.nome)]
            return e.nome if len(grupo) == 1 else None
    notas: dict[str, float] = {}
    for chave, grupo in grupos.items():
        alvos = {chave}
        for e in grupo:
            alvos.update(fold(g) for g in e.grafias)
        notas[chave] = max(SequenceMatcher(None, key, a).ratio() for a in alvos)
    ordem = sorted(notas.items(), key=lambda kv: -kv[1])
    melhor_chave, melhor = ordem[0]
    if melhor < LIMITE_CORRECAO:
        return None
    if len(ordem) > 1 and ordem[1][1] >= LIMITE_CORRECAO and melhor - ordem[1][1] < MARGEM_EMPATE:
        return None
    grupo = grupos[melhor_chave]
    return grupo[0].nome if len(grupo) == 1 else None


def corrige_participantes(participantes: Iterable[str],
                          entradas: Sequence[Entrada]) -> list[str]:
    """Participantes com a grafia da lista quando há erro de digitação.

    "José Edaurdo" vira "José Eduardo". Sem casamento, o nome fica como foi
    digitado. Sem duplicatas (caixa e acento ignorados)."""
    saida: list[str] = []
    vistos: set[str] = set()
    for p in participantes:
        p = _limpa(p)
        if not p:
            continue
        nome = corrige_participante(p, entradas) or p
        if fold(nome) not in vistos:
            vistos.add(fold(nome))
            saida.append(nome)
    return saida
