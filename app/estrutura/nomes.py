"""Resolução de nomes: o que foi OUVIDO -> pessoa da LISTA de irmãos.

Regra única: nunca inventar. Um nome ouvido vira pessoa da lista só quando
casa sem ambiguidade. Quando não casa, ou casa com mais de uma pessoa, sai
como dúvida "[?] Nome?" com os candidatos — e o tempo vai junto no relatório.

Este arquivo é ESTRUTURAL: não ajuste aqui palavras ou grafias. Grafias novas
vão na lista de irmãos (campo "grafias"); palavras que não são nome vão em
regras/*.json ("nao_e_nome"). Só se mexe aqui se a lógica de casamento mudar.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Iterable, Optional


def fold(texto: str) -> str:
    """minúsculas, sem acento."""
    t = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def fonetica(palavra: str) -> str:
    """Chave fonética simples do português: grafias que soam igual dão a mesma chave.
    Matheus/Mateus, Caom/Caon, Kátya/Katia, Marinez/Marines, Jhuan/Juan, Michelle/Michel."""
    p = palavra.lower().replace("ç", "s")
    p = fold(p)
    p = re.sub(r"[^a-z]", "", p)
    for a, b in (("ph", "f"), ("th", "t"), ("y", "i"), ("w", "v"), ("k", "c"),
                 ("qu", "c"), ("ch", "x"), ("lh", "li"), ("nh", "ni"), ("sh", "x")):
        p = p.replace(a, b)
    p = re.sub(r"c(?=[ei])", "s", p)
    p = re.sub(r"g(?=[ei])", "j", p)
    p = p.replace("h", "").replace("z", "s")
    p = re.sub(r"(.)\1+", r"\1", p)          # letras dobradas
    p = re.sub(r"m$", "n", p)                # Caom -> Caon
    if len(p) > 4:
        p = re.sub(r"(?<=[^aeiou])e$", "", p)  # Michele -> Michel, Croce -> Cros
    return p


_FEMININOS_SEM_A = {"ines", "marines", "arlete", "janice", "denise", "ereni", "carmem",
                    "rosemary", "monique", "isabelle", "nadir", "ivone", "michelle",
                    "raquel", "ruth", "edna", "beatriz", "mercedes", "odete", "leila"}


def genero(primeiro_nome: str) -> str:
    f = fold(primeiro_nome)
    if f in _FEMININOS_SEM_A or f.endswith("a"):
        return "f"
    return "m"


@dataclass
class Pessoa:
    nome: str
    observacao: str = ""
    grafias: list = field(default_factory=list)
    origem: str = "lista"          # "lista" | "reuniao" (anunciado nesta reunião, fora da lista)

    @property
    def tokens(self) -> list[str]:
        return [t for t in re.split(r"\s+", self.nome) if fold(t) not in ("de", "da", "do", "dos", "das")]

    @property
    def genero(self) -> str:
        return genero(self.tokens[0])


@dataclass
class Resolucao:
    ouvido: str
    pessoa: Optional[Pessoa]
    candidatos: list
    confianca: str                 # "alta" | "media" | "baixa"
    motivo: str

    @property
    def nome(self) -> str:
        if self.pessoa:
            return self.pessoa.nome
        if self.candidatos:
            return f"[?] {self.ouvido}? ({' ou '.join(p.nome for p in self.candidatos[:3])})"
        return f"[?] {self.ouvido}?"

    @property
    def duvida(self) -> bool:
        return self.pessoa is None


class Lista:
    """Lista de irmãos indexada para casar o que foi ouvido."""

    LIMIAR_PARECIDO = 0.84

    def __init__(self, pessoas: Iterable[Pessoa]):
        self.pessoas: list[Pessoa] = list(pessoas)

    @classmethod
    def de_json(cls, dados: dict) -> "Lista":
        return cls(Pessoa(e["nome"], e.get("observacao", ""), list(e.get("grafias", [])))
                   for e in dados.get("nomes", []))

    def com_extras(self, extras: Iterable[Pessoa]) -> "Lista":
        """Lista + pessoas anunciadas nesta reunião (orador visitante, leitor fora da lista)."""
        conhecidos = {fold(p.nome) for p in self.pessoas}
        novos = [p for p in extras if fold(p.nome) not in conhecidos]
        return Lista(self.pessoas + novos)

    # ---------------------------------------------------------------- casamento
    @staticmethod
    def _iguais(a: str, b: str) -> bool:
        return fonetica(a) == fonetica(b)

    @classmethod
    def _parecidos(cls, a: str, b: str) -> bool:
        fa, fb = fonetica(a), fonetica(b)
        return fa == fb or (min(len(fa), len(fb)) >= 4 and
                            SequenceMatcher(None, fa, fb).ratio() >= cls.LIMIAR_PARECIDO)

    def resolver(self, ouvido: str, genero_dica: Optional[str] = None) -> Resolucao:
        ouvido = ouvido.strip()
        toks = [t for t in ouvido.split() if fold(t) not in ("de", "da", "do", "dos", "das")]
        if not toks:
            return Resolucao(ouvido, None, [], "baixa", "vazio")

        def filtra_genero(cands):
            if genero_dica and len(cands) > 1:
                f = [p for p in cands if p.genero == genero_dica]
                if f:
                    return f
            return cands

        # 1) nome completo ou grafia registrada
        exatos = [p for p in self.pessoas
                  if fonetica("".join(p.tokens)) == fonetica("".join(toks))
                  or any(fonetica(g.replace(" ", "")) == fonetica("".join(toks)) for g in p.grafias)]
        exatos = filtra_genero(exatos)
        if len(exatos) == 1:
            return Resolucao(ouvido, exatos[0], [], "alta", "nome completo/grafia da lista")

        # 1b) nome partido em duas palavras pelo reconhecimento ("Maria Inês" = "Marinês")
        if len(toks) == 2:
            junto = "".join(toks)
            part = filtra_genero([p for p in self.pessoas if self._parecidos(junto, p.tokens[0])
                                  or any(" " not in g and self._parecidos(junto, g) for g in p.grafias)])
            if len(part) == 1 and not any(self._iguais(toks[0], p.tokens[0]) for p in self.pessoas):
                return Resolucao(ouvido, part[0], [], "media", f"ouvido em duas palavras ('{ouvido}')")
        else:
            part = []

        # 2) dois ou mais nomes ouvidos: todos batem com o começo de um nome da lista
        if len(toks) >= 2:
            pref = [p for p in self.pessoas if len(p.tokens) >= len(toks)
                    and all(self._parecidos(a, b) for a, b in zip(toks, p.tokens))]
            if not pref:   # primeiro + último ("Ademir Paulino" não casa; "Fernando Croci" casa)
                pref = [p for p in self.pessoas if len(p.tokens) >= 2
                        and self._parecidos(toks[0], p.tokens[0]) and self._parecidos(toks[-1], p.tokens[-1])]
            pref = filtra_genero(pref)
            if len(pref) == 1:
                return Resolucao(ouvido, pref[0], [], "alta", "nome e sobrenome")
            if len(pref) > 1:
                return Resolucao(ouvido, None, pref, "baixa", "mais de uma pessoa com esse nome")
            # o primeiro nome existe, o sobrenome ouvido não: não crava (pode ser visitante)
            so_primeiro = [p for p in self.pessoas if self._iguais(toks[0], p.tokens[0])]
            if so_primeiro:
                cands = part + [p for p in so_primeiro if p not in part]     # "Maria Inês": Marinês também
                return Resolucao(ouvido, None, cands, "baixa", "sobrenome ouvido não está na lista")
            # a primeira palavra nem é nome e a última é um primeiro nome único ("Toma Denise")
            ult = filtra_genero([p for p in self.pessoas if self._iguais(toks[-1], p.tokens[0])])
            if len(ult) == 1:
                return Resolucao(ouvido, ult[0], [], "media", f"só '{toks[-1]}' casa com a lista")
            return Resolucao(ouvido, None, ult, "baixa", "fora da lista")

        tok = toks[0]
        # 3) um nome só: primeiro nome da lista
        prim = filtra_genero([p for p in self.pessoas if self._iguais(tok, p.tokens[0])
                              or any(self._iguais(tok, g) for g in p.grafias if " " not in g)])
        if len(prim) == 1:
            return Resolucao(ouvido, prim[0], [], "media", "primeiro nome único na lista")
        if len(prim) > 1:
            # mesma grafia exata vence grafia só parecida (William x Willian) — mas sem cravar: confiança média
            exata = [p for p in prim if fold(p.tokens[0]) == fold(tok)]
            if len(exata) == 1:
                outros = ", ".join(p.nome for p in prim if p is not exata[0])
                return Resolucao(ouvido, exata[0], [], "media", f"grafia exata do primeiro nome (parecido: {outros})")
            return Resolucao(ouvido, None, prim, "baixa", "primeiro nome de mais de uma pessoa")
        # 4) sobrenome ("irmão Caon")
        sob = filtra_genero([p for p in self.pessoas if any(self._iguais(tok, t) for t in p.tokens[1:])])
        if len(sob) == 1:
            return Resolucao(ouvido, sob[0], [], "media", "sobrenome único na lista")
        if len(sob) > 1:
            return Resolucao(ouvido, None, sob, "baixa", "sobrenome de mais de uma pessoa")
        # 5) parecido (erro de reconhecimento): só sugere, nunca crava
        par = filtra_genero([p for p in self.pessoas if self._parecidos(tok, p.tokens[0])])
        return Resolucao(ouvido, None, par, "baixa", "fora da lista" + (" (parecido)" if par else ""))
