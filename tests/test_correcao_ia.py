"""Passe de IA ancorada (app/correcao_ia) — plumbing e guarda, sem API real.

Usa um cliente-stub: não gasta crédito. Prova que a resposta boa é aplicada e
que resposta malformada/contagem errada cai de volta no texto ORIGINAL."""
import json

from app import correcao_ia


class _Bloco:
    def __init__(self, text): self.type = "text"; self.text = text


class _Msg:
    def __init__(self, text): self.content = [_Bloco(text)]


class _Client:
    """Stub: devolve o que `fn(prompt)` mandar."""
    def __init__(self, fn): self.messages = self; self._fn = fn
    def create(self, model, max_tokens, messages):
        return _Msg(self._fn(messages[0]["content"]))


BLOCOS = [
    {"quem": "Rinaldo Santos", "texto": "irmão Ademir Balbino, sua esposa"},
    {"quem": "Rinaldo Santos", "texto": "Módulo Vir, Palpino, esposa"},
]
NOMES = ["Ademir Baldino", "Rinaldo Santos"]


def test_aplica_correcao_boa():
    resp = json.dumps(["irmão Ademir Baldino, sua esposa", "Ademir Baldino, esposa"])
    out = correcao_ia.corrige_blocos(BLOCOS, NOMES, client=_Client(lambda p: resp))
    assert out == ["irmão Ademir Baldino, sua esposa", "Ademir Baldino, esposa"]


def test_contagem_errada_mantem_original():
    # modelo devolveu 1 item para 2 blocos -> guarda: originais
    out = correcao_ia.corrige_blocos(BLOCOS, NOMES, client=_Client(lambda p: '["só um"]'))
    assert out == [b["texto"] for b in BLOCOS]


def test_resposta_nao_json_mantem_original():
    out = correcao_ia.corrige_blocos(BLOCOS, NOMES, client=_Client(lambda p: "desculpe, não consigo"))
    assert out == [b["texto"] for b in BLOCOS]


def test_nomes_vao_no_prompt():
    visto = {}
    correcao_ia.corrige_blocos(BLOCOS, NOMES, client=_Client(lambda p: visto.setdefault("p", p) or json.dumps([b["texto"] for b in BLOCOS])))
    assert "Ademir Baldino" in visto["p"] and "array JSON" in visto["p"]
