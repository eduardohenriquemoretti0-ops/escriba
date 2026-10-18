#!/usr/bin/env python3
"""Reconstrói os segmentos CRUS (pré-merge) do motor a partir do atribuicao.json
de um job do Escriba, para usar como ENTRADA de gabarito do motor de estrutura.

Por que: os bugs do motor (causa H) só reproduzem com as fronteiras reais do
reconhecimento. Re-rodar o ASR é caro e o merge perde as fronteiras. O motor,
porém, grava em `atribuicao.json` as `falas` com o campo `seg` (índice do
segmento cru que a fala veio). Como a voz é constante dentro de um `seg`, dá
para reconstruir o segmento cru agrupando as falas por `seg`, concatenando o
texto e pegando início/fim das pontas. O motor re-divide igual → determinístico.

Uso:
  python reconstruir_crus.py <jobs>/<id>/atribuicao.json  dados/<saida>.json

A saída é a lista [{inicio, fim, texto, voz}] que `ler_srt_motor`/o teste de
gabarito consomem via `analisar(segs, lista, papeis)`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def reconstruir(atrib: dict) -> list[dict]:
    grupos: dict = {}
    for f in atrib.get("falas", []):
        grupos.setdefault(f["seg"], []).append(f)
    crus = []
    for seg in sorted(grupos):
        g = grupos[seg]
        crus.append({
            "inicio": float(g[0]["ini"]),
            "fim": float(g[-1]["fim"]),
            "texto": " ".join(x["texto"] for x in g),
            "voz": g[0]["voz"],
        })
    return crus


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    atrib = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    crus = reconstruir(atrib)
    Path(sys.argv[2]).write_text(json.dumps(crus, ensure_ascii=False), encoding="utf-8")
    print(f"{len(crus)} segmentos crus → {sys.argv[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
