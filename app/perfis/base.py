"""Base dos perfis: o contexto comum e as primitivas de nomeação.

`Ctx.montar` pré-calcula uma vez tudo que os perfis compartilham (mapa de
rótulos, início de cada falante, lista de irmãos, participantes, memória de
vozes, palpites de conversa). As primitivas (`passo_*`) são os tijolos que cada
perfil combina — assim um tipo novo de reunião é um módulo, não uma emenda no
núcleo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .. import exports, irmaos, nomes, vozes
from ..engines.base import CORO_LABEL


def _conhecido(nome: str, entradas, participantes) -> bool:
    """Nome que consta na lista de irmãos (inclusive por grafia) ou nos participantes."""
    if irmaos.corrige_participante(nome, entradas) is not None:
        return True
    return nomes._participante_de(nome, list(participantes)) is not None


@dataclass
class Ctx:
    """Tudo que os perfis precisam, calculado uma vez."""
    job: object
    segments: list                       # sem o coro
    voices: dict
    display: dict                        # rótulo cru -> "Falante N"
    inicio: dict                         # "Falante N" -> início (s)
    entradas: list                       # lista de irmãos
    participantes: list                  # participantes (grafia corrigida)
    by_voice: dict                       # rótulo cru -> (nome, similaridade)
    strong: set                          # rótulos com voz muito parecida
    by_talk: dict                        # rótulo cru -> (nome, _, motivo)

    @classmethod
    def montar(cls, job, segments, voices) -> "Ctx":
        import logging
        log = logging.getLogger("escriba")
        segs = [s for s in segments if s.speaker != CORO_LABEL]
        display = exports.speaker_map(segs)
        inicio: dict[str, float] = {}
        for s in segs:
            lab = display.get(s.speaker)
            if lab and lab not in inicio:
                inicio[lab] = s.start
        by_voice = vozes.match({lab: v["vetor"] for lab, v in voices.items()})
        strong = {lab for lab, (_, sim) in by_voice.items()
                  if sim >= vozes.MATCH_STRONG}
        entradas = irmaos.listar_seguro(log)
        participantes = irmaos.corrige_participantes(job.participants, entradas)
        candidates = [*participantes, *(p["nome"] for p in vozes.list_people())]
        by_talk = nomes.infer(segs, candidates, exclude=strong, irmaos=entradas)
        return cls(job, segs, voices, display, inicio, entradas, participantes,
                   by_voice, strong, by_talk)

    # ---- helpers -----------------------------------------------------------
    def com_tempo(self, label: str, detalhe: str) -> str:
        t = self.inicio.get(label)
        return f"{detalhe} · em {exports.hms(t)}" if t is not None else detalhe

    def conhecido(self, nome: str) -> bool:
        return _conhecido(nome, self.entradas, self.participantes)

    # ---- primitivas de nomeação (tijolos dos perfis) -----------------------
    def passo_voz_e_conversa(self, auto: dict) -> None:
        """Voz reconhecida (memória) e palpite da conversa (chamada solta)."""
        for raw, label in self.display.items():
            if raw in self.by_voice and (raw in self.strong or raw not in self.by_talk):
                name, sim = self.by_voice[raw]
                auto[label] = {"nome": name, "fonte": "voz",
                               "detalhe": f"voz reconhecida ({round(sim * 100)}% parecida)"}
            elif raw in self.by_talk:
                name, _, why = self.by_talk[raw]
                if self.conhecido(name):
                    auto[label] = {"nome": name, "fonte": "conversa", "detalhe": why}
                else:
                    auto[label] = {"nome": f"[?] {name}?", "fonte": "duvida",
                                   "detalhe": self.com_tempo(label, f"{why} · fora da "
                                   "lista e dos participantes · ouça o trecho e confirme")}

    def passo_chamada(self, auto: dict) -> None:
        """Chamada/agradecimento: o nome chamado (grafia da lista) vai à parte toda."""
        for raw, (nome, motivo, na_lista) in nomes.por_chamada(
                self.segments, self.entradas, self.participantes).items():
            label = self.display.get(raw)
            if not label or auto.get(label, {}).get("fonte") in ("voz", "presidente", "turno"):
                continue
            # chamada fora da lista é só dúvida: não apaga um nome já firme
            # (ex.: "Entendeu?" lido como chamada por cima de "chamado pelo nome 5×")
            if not na_lista and auto.get(label, {}).get("fonte") not in (None, "duvida"):
                continue
            if na_lista:
                auto[label] = {"nome": nome, "fonte": "chamado", "detalhe": motivo}
            else:
                auto[label] = {"nome": f"[?] {nome}?", "fonte": "duvida",
                               "detalhe": self.com_tempo(label, f"{motivo} · fora da "
                               "lista · ouça o trecho e confirme")}

    def passo_homonimo_e_dedup(self, auto: dict) -> None:
        """Observação de homônimo e um nome por voz (menos os "[?]")."""
        for label, info in auto.items():
            if info["fonte"] == "duvida":
                continue
            aviso = irmaos.corrige(info["nome"], self.entradas)
            if getattr(aviso, "ambiguo", False):
                info["detalhe"] = f"{info['detalhe']} · {aviso.observacao}"
        seen: set = set()
        for label in list(auto):
            if auto[label]["fonte"] == "duvida":
                continue
            if auto[label]["nome"] in seen:
                del auto[label]
            else:
                seen.add(auto[label]["nome"])

    def passo_eliminacao(self, auto: dict) -> None:
        """1 falante sem nome + 1 participante livre: sugestão por eliminação."""
        nomeados = {**self.job.speaker_names,
                    **{lab: i["nome"] for lab, i in auto.items()}}
        for label, nome in nomes.por_eliminacao(self.display.values(),
                                                self.participantes, nomeados).items():
            auto[label] = {"nome": nome, "fonte": "eliminacao",
                           "detalhe": nomes.ELIMINACAO_DETALHE}


@dataclass
class Perfil:
    nome: str
    descricao: str
    nomeia: Callable[[Ctx], dict]
