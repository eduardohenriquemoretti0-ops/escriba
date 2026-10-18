"""Lê um .srt do Escriba ("Nome: texto") como segmentos crus.
O nome que o Escriba pôs vira só um RÓTULO DE VOZ (V1, V2...): o motor não vê nomes antigos."""
import re

def ler_srt(caminho):
    blocos = re.split(r"\n\s*\n", open(caminho, encoding="utf-8").read().strip())
    segs, vozes = [], {}
    def seg(t):
        h, m, s = t.replace(",", ".").split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    for b in blocos:
        linhas = b.strip().splitlines()
        if len(linhas) < 3:
            continue
        a, z = [x.strip() for x in linhas[1].split("-->")]
        txt = " ".join(linhas[2:])
        m = re.match(r"^(.*?):\s(.*)$", txt)
        rot, txt = (m.group(1), m.group(2)) if m else ("?", txt)
        voz = vozes.setdefault(rot, f"V{len(vozes) + 1}")
        segs.append({"inicio": seg(a), "fim": seg(z), "texto": txt, "voz": voz, "_rotulo_antigo": rot})
    return segs
