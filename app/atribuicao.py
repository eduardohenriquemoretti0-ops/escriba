"""Ponte entre o motor de estrutura (app.estrutura) e o Escriba.

O motor decide quem fala TRECHO A TRECHO pela estrutura da reunião (ver
app/estrutura e ESPECIFICACAO). Este módulo só faz a cola, sem regra de
reunião: converte os segmentos do engine para o motor, roda `analisar`, e
devolve o resultado em segmentos rotulados por PESSOA (P01, P02…) mais o
`nomes_auto` que as telas/saídas já sabem usar ("Falante N" -> nome).

A lista de irmãos é OBRIGATÓRIA (regra do Eduardo): sem lista, o motor não
roda e o job segue pelo caminho antigo (perfis).
"""
from __future__ import annotations

import json
import logging

from . import config, exports
from .engines.base import CORO_LABEL, FORA_LABEL, NAO_PESSOA, Segment
from .estrutura import analisar

log = logging.getLogger("escriba.atribuicao")

ATRIB_FILE = "atribuicao.json"
# quem/papel que NÃO é uma pessoa da reunião
_CORO_QUEM = "Em coro"
_FORA_QUEM = "(fora da reunião)"
_CORO_PAPEIS = ("cântico", "amém")


def _lista_json() -> dict | None:
    """Conteúdo do nomes.json (mesmo formato do Escriba). None se não houver lista."""
    caminho = config.DATA_DIR / "nomes" / "nomes.json"
    try:
        dados = json.loads(caminho.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        log.exception("nomes.json ilegível para o motor de estrutura")
        return None
    return dados if dados.get("nomes") else None


def rodar(job, raw_segments: list[Segment]) -> dict | None:
    """Roda o motor de estrutura nos segmentos CRUS (antes do merge).

    Devolve o dict de `Resultado.como_dict()` ou None se a lista de irmãos não
    existir (aí o job segue pelo caminho antigo). Não grava nada: quem grava é
    o worker."""
    lista = _lista_json()
    if lista is None:
        log.warning("motor de estrutura pulado: nomes.json ausente ou vazio "
                    "(a lista de irmãos é obrigatória)")
        return None
    segs = [{"inicio": s.start, "fim": s.end, "texto": s.text, "voz": s.speaker}
            for s in raw_segments if (s.text or "").strip()]
    papeis = (job.meeting or {}).get("papeis") or None
    # tipo escolhido no cadastro (fim/meio de semana) vale sobre a detecção
    tipo = (job.meeting or {}).get("tipo_reuniao") or None
    # quem o usuário digitou em "participaram da reunião" (antes era ignorado aqui:
    # só servia de vocabulário para o Whisper — bug do 17-10-2026)
    participantes = [p for p in (getattr(job, "participants", None) or []) if p]
    res = analisar(segs, lista, papeis=papeis, tipo=tipo, participantes=participantes)
    atrib = res.como_dict()
    motivo = _estrutura_falhou(atrib)
    if motivo:
        # trava (17-10-2026: a reunião inteira saiu "[?]" porque a estrutura não foi achada):
        # resultado sem estrutura não vai para o PDF; o job segue pelo caminho antigo (perfis)
        log.warning("motor de estrutura descartado: %s; seguindo pelos perfis", motivo)
        return None
    return atrib


# fração máxima do tempo de fala que pode ficar "sem estrutura"
MAX_SEM_ESTRUTURA = 0.35


def _estrutura_falhou(atrib: dict) -> str:
    """Motivo para descartar o resultado do motor ('' se ele achou a estrutura)."""
    if not atrib.get("blocos"):
        return "nenhuma parte da reunião foi achada"
    falas = [f for f in atrib.get("falas", []) if f.get("quem") not in (_CORO_QUEM, _FORA_QUEM)]
    total = sum(f["fim"] - f["ini"] for f in falas)
    sem = sum(f["fim"] - f["ini"] for f in falas if f.get("parte") == "sem estrutura")
    if total > 0 and sem / total > MAX_SEM_ESTRUTURA:
        return f"{sem / total:.0%} do tempo de fala ficou sem estrutura"
    return ""


def _pessoa_de(fala: dict) -> str:
    """Rótulo estável por pessoa; CORO_LABEL/FORA_LABEL para quem não é pessoa."""
    quem, papel = fala.get("quem", ""), fala.get("papel", "")
    if quem == _CORO_QUEM or papel in _CORO_PAPEIS:
        return CORO_LABEL
    if quem == _FORA_QUEM:
        return FORA_LABEL
    # desconhecido sem nome: separa por voz (vozes diferentes = pessoas diferentes)
    if quem == "[?]":
        return "[?]|" + (fala.get("voz") or "")
    return quem


def segmentos(atrib: dict) -> list[Segment]:
    """Segmentos do motor (texto já dividido) com `speaker` = P01, P02… por
    pessoa, na ordem em que aparecem. Coro vira CORO_LABEL."""
    rotulo: dict[str, str] = {}
    out: list[Segment] = []
    for f in atrib.get("falas", []):
        pk = _pessoa_de(f)
        if pk in NAO_PESSOA:
            spk = pk
        else:
            spk = rotulo.get(pk)
            if spk is None:
                spk = f"P{len(rotulo) + 1:02d}"
                rotulo[pk] = spk
        out.append(Segment(float(f["ini"]), float(f["fim"]), spk, str(f["texto"])))
    return out


def nomes_auto(atrib: dict, segs: list[Segment] | None = None) -> dict:
    """{'Falante N': {nome, fonte, detalhe, confianca, duvida}} a partir do motor.

    A chave é o rótulo de exibição ("Falante N") que o Escriba usa em
    `nomes_auto`/`speaker_names` — calculado pelo mesmo `exports.speaker_map`
    que as saídas usam, para casar certinho."""
    if segs is None:
        segs = segmentos(atrib)
    display = exports.speaker_map(segs)           # P01 -> "Falante 1"
    auto: dict[str, dict] = {}
    for seg, f in zip(segs, atrib.get("falas", [])):
        if seg.speaker in NAO_PESSOA:
            continue
        label = display.get(seg.speaker)
        if not label or label in auto:            # só a 1ª fala de cada pessoa
            continue
        auto[label] = {
            "nome": f.get("quem", ""),
            "fonte": f.get("papel") or "estrutura",
            "detalhe": f.get("motivo", ""),
            "confianca": f.get("confianca", ""),
            "duvida": bool(f.get("duvida")),
        }
    return auto
