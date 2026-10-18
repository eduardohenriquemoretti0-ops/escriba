"""Leitura das REGRAS (regras/*.json) e detecção de eventos no texto.

Evento = algo dito que marca a estrutura: chamada ("Irmã Fátima, por
gentileza"), agradecimento ("Obrigado, Marcos"), dica de parágrafo ("Vamos ao
12"), cântico, oração, anúncio de orador/dirigente/leitor, designação de parte.

ESTRUTURAL: este arquivo só compila e aplica o que está no JSON. Frase nova,
jeito novo de chamar, palavra que não é nome -> mexa no JSON, não aqui.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .nomes import fold

NOME_RX = (r"(?P<nome>[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõçüñ]+"
           r"(?:\s+(?:d[aeo]s?\s+)?[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõçüñ]+){0,2})")

PASTA_REGRAS = Path(__file__).resolve().parent.parent / "regras"


@dataclass
class Evento:
    tipo: str
    ini: int                 # posição no texto (caractere)
    fim: int
    nome: Optional[str] = None
    num: Optional[str] = None
    forca: str = "forte"


class Regras:
    def __init__(self, dados: dict):
        self.dados = dados
        self.limites = dados.get("limites", {})
        self.nao_e_nome = {fold(w) for w in dados.get("nao_e_nome", [])}
        self.livros = {fold(w) for w in dados.get("livros_com_nome_de_pessoa", [])}
        self.palavras_retorno = {fold(w) for w in dados.get("palavras_retorno", [])}
        # papel -> palavras que, ditas perto de um nome, dizem o que a pessoa vai fazer
        self.papel_palavras = {p: [fold(w) for w in ws] for p, ws in dados.get("papel_palavras", {}).items()}
        self._ev: dict[str, list[tuple[re.Pattern, str]]] = {}
        for tipo, lista in dados["eventos"].items():
            comp = []
            for item in lista:
                padrao, forca = (item["padrao"], item.get("forca", "forte")) if isinstance(item, dict) else (item, "forte")
                comp.append((re.compile(padrao.replace("{NOME}", NOME_RX)), forca))
            self._ev[tipo] = comp
        self.tipo_reuniao = {k: [re.compile(p) for p in v] for k, v in dados["tipo_reuniao"].items()}
        self.partes = [(p["id"], re.compile(p["padrao"]), p["modo"]) for p in dados.get("partes_meio_de_semana", [])]

    @classmethod
    def carregar(cls, nome: str = "religiosa", pasta: Optional[Path] = None) -> "Regras":
        caminho = (pasta or PASTA_REGRAS) / f"{nome}.json"
        return cls(json.loads(caminho.read_text(encoding="utf-8")))

    def lim(self, chave: str, padrao=None):
        return self.limites.get(chave, padrao)

    # ------------------------------------------------------------------ nomes
    def limpa_nome(self, nome: Optional[str], texto: str, fim: int) -> Optional[str]:
        """Tira do nome capturado palavras que não são nome; recusa livro bíblico seguido de número."""
        if not nome:
            return None
        partes = [p for p in nome.split() if fold(p) not in self.nao_e_nome]
        if not partes:
            return None
        if fold(partes[0]) in self.livros and re.match(r"\s*(?:\d|cap[íi]tulo|vers[íi]culo)", texto[fim:], re.I):
            return None
        return " ".join(partes)

    # ---------------------------------------------------------------- eventos
    def tem(self, tipo: str, texto: str) -> bool:
        return any(rx.search(texto) for rx, _ in self._ev.get(tipo, []))

    def eventos(self, texto: str, tipos: Optional[list[str]] = None) -> list[Evento]:
        achados: list[Evento] = []
        for tipo in (tipos or self._ev.keys()):
            for rx, forca in self._ev.get(tipo, []):
                for m in rx.finditer(texto):
                    gd = m.groupdict()
                    nome = gd.get("nome")
                    if "nome" in gd:
                        nome = self.limpa_nome(nome, texto, m.end("nome") if nome else m.end())
                        if rx.pattern.count("(?P<nome>") and nome is None and tipo in ("chamada", "agradecimento", "convite_dirigente", "leitor", "oracao_designada", "anuncio_orador"):
                            continue
                    achados.append(Evento(tipo, m.start(), m.end(), nome, gd.get("num"), forca))
        # mesma posição de nome: agradecimento vence chamada ("Muito bem, irmã X")
        saida: list[Evento] = []
        for e in sorted(achados, key=lambda e: (e.ini, e.tipo != "agradecimento")):
            if e.tipo == "chamada" and any(o.tipo == "agradecimento" and o.nome == e.nome and abs(o.fim - e.fim) < 3 for o in achados):
                continue
            if e.tipo == "chamada" and any(o.tipo == "chamada" and o.nome == e.nome and o.fim == e.fim and o is not e and o in saida for o in achados):
                continue
            saida.append(e)
        return saida

    def tipo_de_reuniao(self, textos: list[str]) -> str:
        tudo = " ".join(textos)
        pontos = {k: sum(1 for rx in v if rx.search(tudo)) for k, v in self.tipo_reuniao.items()}
        return max(pontos, key=pontos.get) if any(pontos.values()) else "desconhecido"

    def e_retorno(self, texto: str) -> bool:
        """"Isso." "Muito bem." "Exatamente." "Como?" — só palavras de retorno, curto."""
        palavras = [fold(w) for w in re.findall(r"[\wÀ-ÿ]+", texto)]
        return (0 < len(palavras) <= self.lim("retorno_max_palavras", 5)
                and all(w in self.palavras_retorno for w in palavras))
