"""Correção de NOME no texto da transcrição (não é grafia da lista).

O motor já decide quem fala e resolve os nomes da reunião (papéis + chamados).
Mas o TEXTO continua o que o ASR ouviu: o presidente anuncia "Ademir Baldino" e
o reconhecimento escreve "Ademir Balbino"/"Ademir Baulino". Aqui trocamos, no
texto, o nome ERRADO pelo nome RESOLVIDO quando bate por fonética — e só então.

Seguro por construção:
- Só corrige para nomes que JÁ foram resolvidos nesta reunião (os `alvos`).
- Nome de 2+ palavras: o PRIMEIRO nome tem de bater exato por fonética e o
  sobrenome tem de ser PARECIDO (limiar do motor). Assim "Ademir Balbino" vira
  "Ademir Baldino", mas "Ademir Duarte" (outro irmão real) NÃO é tocado, porque
  "Duarte" não é parecido com "Baldino".
- Garble severo ("Palpino", "Módulo Vir") não casa por fonética e fica como está
  (é trabalho do passe de IA ancorada, não deste módulo).
"""
from __future__ import annotations

import re
from typing import Iterable

from .estrutura.nomes import Lista, fold, fonetica

# palavra com inicial maiúscula (nome próprio candidato)
_MAIUSC = r"[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõçüñ]+"


def _tokens(nome: str) -> list[str]:
    return [t for t in re.split(r"\s+", nome.strip())
            if fold(t) not in ("de", "da", "do", "dos", "das") and t]


def corrige(texto: str, alvos: Iterable[str]) -> str:
    """Troca no texto garbles de nome pelos `alvos` resolvidos, por fonética."""
    if not texto:
        return texto
    # só alvos com nome + sobrenome (o de 1 nome já é coberto pela grafia da lista)
    nomes = []
    for a in alvos:
        toks = _tokens(a)
        if len(toks) >= 2:
            nomes.append((toks, a.strip()))
    if not nomes:
        return texto

    def troca(m: re.Match) -> str:
        bruto = m.group(0)
        btoks = bruto.split()
        for toks, canonico in nomes:
            if len(btoks) < 2:
                continue
            # primeiro nome exato por fonética; último token parecido com o sobrenome
            if (fonetica(btoks[0]) == fonetica(toks[0])
                    and Lista._parecidos(btoks[-1], toks[-1])
                    and fonetica(btoks[-1]) != fonetica(toks[0])):
                return canonico
        return bruto

    # sequências de 2 a 3 palavras maiúsculas (com de/da opcional no meio)
    seq = rf"{_MAIUSC}(?:\s+(?:d[aeo]s?\s+)?{_MAIUSC}){{1,2}}"
    return re.sub(seq, troca, texto)
