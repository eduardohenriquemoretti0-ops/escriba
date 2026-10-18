"""Regressão do motor de estrutura contra os gabaritos ouvidos pelo Eduardo.

Roda o motor REAL do app (app.estrutura) — o mesmo código que o Escriba usa.
Nenhuma mudança (regra ou motor) entra se este teste piorar. Item 'eduardo'
que falha = erro. Item 'texto' que falha = alerta (vira erro quando confirmado).
Rodar: PYTHONPATH=. python -m pytest tests/estrutura -q
"""
import json
import sys
from pathlib import Path

from app.estrutura import analisar

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from ler_srt_motor import ler_srt  # noqa: E402
LISTA = json.loads((AQUI / "dados/lista_irmaos.json").read_text(encoding="utf-8"))


def _carrega(g):
    ent = AQUI / "dados" / Path(g["entrada"]).name
    return json.loads(ent.read_text(encoding="utf-8")) if ent.suffix == ".json" else ler_srt(ent)


def avaliar(arquivo):
    g = json.loads(Path(arquivo).read_text(encoding="utf-8"))
    res = analisar(_carrega(g), LISTA, g.get("papeis_informados"))
    erros, alertas = [], []
    if g.get("tipo") and res.tipo != g["tipo"]:
        erros.append(f"tipo: {res.tipo} != {g['tipo']}")
    for it in g["itens"]:
        falas = [f for f in res.falas if it["ancora"] in f.texto]
        if not falas:
            falas = [f for f in res.falas if it["ancora"][:25] in f.texto]
        if not falas:
            (erros if it["fonte"] == "eduardo" else alertas).append(f"âncora não achada: {it['ancora']}")
            continue
        f = falas[0]
        ok = True
        if "quem" in it and f.quem != it["quem"]:
            ok = False
        if "quem_um_de" in it and f.quem not in it["quem_um_de"]:
            ok = False
        if "papel" in it and f.papel != it["papel"]:
            ok = False
        if "duvida_com" in it and not (f.quem == it["duvida_com"] or (f.duvida and it["duvida_com"] in f.quem)
                                       or (f.duvida and it["duvida_com"].split()[0] in f.quem)):
            ok = False
        if "nao_quem" in it and it["nao_quem"] in f.quem:
            ok = False
        if not ok:
            msg = f"{it['ancora'][:45]!r}: saiu '{f.quem}' ({f.papel}); esperado {it.get('quem') or it.get('quem_um_de') or it.get('duvida_com') or ('não ' + it.get('nao_quem', ''))}"
            (erros if it["fonte"] == "eduardo" else alertas).append(msg)
    return res, erros, alertas, len(g["itens"])


def _teste(nome):
    res, erros, alertas, n = avaliar(AQUI / "gabaritos" / nome)
    for a in alertas:
        print("ALERTA", a)
    assert not erros, "\n".join(erros)


def test_10_10_fim_de_semana():
    _teste("10-10-2026.json")


def test_13_10_meio_de_semana():
    _teste("13-10-2026.json")


def test_10_10_local_causa_H():
    """Bugs do motor (causa H) no caminho local, com os segmentos crus reais."""
    _teste("10-10-2026-local-H.json")
