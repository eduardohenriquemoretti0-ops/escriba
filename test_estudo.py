"""Testes do estudo por palavra (app/estudo.py), sem modelos.

Reunião sintética: cada item vira uma frase da diarização (palavras com tempo,
cluster de voz e embedding) e um bloco de base com o rótulo do cluster.

Rodar:  python -m pytest test_estudo.py -v
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from app import config, estudo, irmaos, jobs, review, turnos, vozes, worker
from app.engines.base import Segment
from app.engines.diarize import Sentence, Word

LETRAS = "ABCDEFGH"
LISTA = [
    irmaos.Entrada(id=str(i), nome=n)
    for i, n in enumerate(
        ["Paulo", "Ana", "Rute", "Silas", "Lia", "Davi", "Eva", "Noé"]
    )
]


def _voz(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=64).astype(np.float32)
    return v / np.linalg.norm(v)


def _reuniao(itens: list[tuple[int, str, float]]):
    """(cluster, texto, segundos) -> (frases da diarização, blocos de base, embed)."""
    t = 0.0
    frases, segs, vozes_t = [], [], []
    for cl, txt, dur in itens:
        toks = txt.split()
        dt = dur / len(toks)
        ws = [
            Word(t + k * dt, t + (k + 1) * dt - 0.02, " " + tok)
            for k, tok in enumerate(toks)
        ]
        s = Sentence(ws, cl)
        s.embedding = _voz(cl) if dur >= 1.0 else None
        frases.append(s)
        segs.append(Segment(ws[0].start, ws[-1].end, LETRAS[cl], txt))
        vozes_t.append((ws[0].start, ws[-1].end, _voz(cl)))
        t += dur + 0.5

    def embed(a: float, b: float) -> np.ndarray:
        for ini, fim, v in vozes_t:
            if ini <= a < fim + 0.5:
                return v
        return _voz(999)

    return frases, segs, embed


D, L = 0, 1  # cluster do dirigente e do leitor
LEITURA = "Esta é a leitura do parágrafo com bastante texto para durar muito tempo."
COMENTARIO = "Eu acho que Jeová cuida de cada um de nós com muito carinho."


def _estudo(extra: list[tuple[int, str, float]] = ()) -> list[tuple[int, str, float]]:
    """Estudo com 4 parágrafos e 8 chamadas do dirigente (a guarda liga)."""
    itens = [(D, "Vamos começar o estudo de A Sentinela.", 3)]
    pessoas = [("Paulo", "Ana"), ("Rute", "Silas"), ("Lia", "Davi"), ("Eva", "Noé")]
    for n, (a, b) in enumerate(pessoas, 1):
        itens += [
            (D, f"Parágrafo {n}.", 1.5),
            (L, LEITURA, 10),
            (D, f"Qual é a resposta? Irmão {a}, por favor.", 3),
            (2 + n % 3, COMENTARIO, 6),
            (D, f"Obrigado, irmão {a}. Irmã {b}, por favor.", 3),
            (3 + n % 3, COMENTARIO, 6),
            (D, f"Muito bem, irmã {b}.", 2),
        ]
    return itens + list(extra)


def _nome_em(res: estudo.Resultado, instante: float) -> tuple[str, str]:
    """(rótulo, nome) do bloco que cobre o instante."""
    seg = next(s for s in res.segmentos if s.start <= instante <= s.end)
    pessoa = next((p for p in res.pessoas if p.rotulo == seg.speaker), None)
    return seg.speaker, (pessoa.nome if pessoa else "")


def _inicio_do_texto(frases, trecho: str, ordem: int = 0) -> float:
    achados = [f.start for f in frases if trecho in f.text]
    return achados[ordem]


# --------------------------------------------------------------------------
# (e) guarda
# --------------------------------------------------------------------------
def test_guarda_conversa_de_duas_pessoas_nao_ativa_o_estudo():
    itens = [
        (0, "Fernando, por favor, me fala do orçamento.", 3),
        (1, "Então, o orçamento fechou ontem e ficou dentro do previsto.", 6),
        (0, "Muito bem, Fernando. E o cronograma?", 3),
        (1, "O cronograma atrasou duas semanas por causa do fornecedor.", 6),
        (0, "Eduardo, por favor, anota isso.", 3),
        (1, "Combinado, vou mandar por escrito amanhã cedo.", 4),
    ]
    frases, segs, embed = _reuniao(itens)
    base = [turnos.Pessoa("A", "Eduardo", "comentarista", "", 1, 9.0, True)]
    res = estudo.aplica(frases, segs, base, embed, LISTA, ["Eduardo", "Fernando"])
    assert not res.ativa
    assert res.segmentos == segs and res.pessoas == base
    assert res.info["motivo"].startswith("guarda:")


def test_guarda_exige_anuncios_de_paragrafo():
    # 8 chamadas, mas nenhum "parágrafo"/"leitura": não é estudo
    itens = [(D, "Vamos começar o estudo.", 2)]
    for n in ["Paulo", "Ana", "Rute", "Silas", "Lia", "Davi", "Eva", "Noé"]:
        itens += [
            (D, f"Irmão {n}, por favor.", 2),
            (2, COMENTARIO, 5),
            (D, f"Obrigado, irmão {n}.", 2),
        ]
    frases, segs, embed = _reuniao(itens)
    res = estudo.aplica(frases, segs, [], embed, LISTA, [])
    assert not res.ativa and "0 anúncios" in res.info["motivo"]


def test_estudo_completo_liga_a_estrutura():
    frases, segs, embed = _reuniao(_estudo())
    res = estudo.aplica(frases, segs, [], embed, LISTA, [])
    assert res.ativa
    assert res.info["chamadas"] >= estudo.GUARDA_CHAMADAS
    assert res.info["paragrafos"] >= estudo.GUARDA_PARAGRAFOS


# --------------------------------------------------------------------------
# (a) chamada -> fala do chamado
# --------------------------------------------------------------------------
def test_fala_depois_da_chamada_e_do_chamado():
    frases, segs, embed = _reuniao(_estudo())
    res = estudo.aplica(frases, segs, [], embed, LISTA, [])
    t = _inicio_do_texto(frases, "Jeová cuida", 0) + 1
    assert _nome_em(res, t)[1] == "Paulo"
    t = _inicio_do_texto(frases, "Jeová cuida", 1) + 1
    assert _nome_em(res, t)[1] == "Ana"


def test_chamado_com_voz_no_cluster_do_dirigente_continua_do_chamado():
    # o comentário da Rute caiu no cluster do dirigente (mesmo microfone)
    itens = _estudo()
    k = next(i for i, it in enumerate(itens) if "Irmão Rute" in it[1])
    itens[k + 1] = (D, COMENTARIO, 6)
    frases, segs, embed = _reuniao(itens)
    res = estudo.aplica(frases, segs, [], embed, LISTA, [])
    t = frases[k + 1].start + 1
    rot, nome = _nome_em(res, t)
    assert nome == "Rute" and rot != "A"
    # e o dirigente continua com o rótulo dele
    assert _nome_em(res, frases[k].start + 0.5)[0] == "A"


def test_mesmo_nome_reaproveita_o_rotulo_de_fora_do_estudo():
    # Paulo já tinha nome fora do estudo (ex.: presidente): o comentário dele
    # usa o mesmo rótulo, para não virar duas pessoas com o mesmo nome
    itens = [(5, "Bom dia a todos, sejam bem-vindos à nossa reunião.", 5)] + _estudo()
    frases, segs, embed = _reuniao(itens)
    base = [turnos.Pessoa("F", "Paulo", "presidente", "lista", 0, 5.0, True)]
    res = estudo.aplica(frases, segs, base, embed, LISTA, [])
    t = _inicio_do_texto(frases, "Jeová cuida", 0) + 1
    assert _nome_em(res, t) == ("F", "Paulo")
    papel = next(p.papel for p in res.pessoas if p.rotulo == "F")
    assert papel == "presidente + comentarista"


# --------------------------------------------------------------------------
# (d) + (L) nome só com evidência
# --------------------------------------------------------------------------
def test_nome_fora_da_lista_sem_segunda_mencao_vira_duvida():
    extra = [
        (D, "Mais um comentário? Irmão Jorge, por favor.", 3),
        (6, COMENTARIO, 6),
        (D, "Vamos ao parágrafo 5.", 2),
    ]
    frases, segs, embed = _reuniao(_estudo(extra))
    res = estudo.aplica(frases, segs, [], embed, LISTA, [])
    t = _inicio_do_texto(frases, "Jeová cuida", 8) + 1
    rot, nome = _nome_em(res, t)
    assert nome == "[?] Jorge?" and rot.startswith("Q")
    pessoa = next(p for p in res.pessoas if p.rotulo == rot)
    assert pessoa.duvida and not pessoa.nomeada


def test_nome_fora_da_lista_com_chamada_e_agradecimento_vale():
    extra = [
        (D, "Mais um comentário? Irmão Jorge, por favor.", 3),
        (6, COMENTARIO, 6),
        (D, "Obrigado, irmão Jorge.", 2),
    ]
    frases, segs, embed = _reuniao(_estudo(extra))
    res = estudo.aplica(frases, segs, [], embed, LISTA, [])
    t = _inicio_do_texto(frases, "Jeová cuida", 8) + 1
    assert _nome_em(res, t)[1] == "Jorge"


def test_participante_informado_vale_como_conhecido():
    extra = [
        (D, "Mais um comentário? Irmão Jorge, por favor.", 3),
        (6, COMENTARIO, 6),
        (D, "Vamos ao parágrafo 5.", 2),
    ]
    frases, segs, embed = _reuniao(_estudo(extra))
    res = estudo.aplica(frases, segs, [], embed, LISTA, ["Jorge"])
    t = _inicio_do_texto(frases, "Jeová cuida", 8) + 1
    assert _nome_em(res, t)[1] == "Jorge"


def test_trecho_sem_chamada_nem_agradecimento_vira_duvida():
    extra = [
        (D, "Alguém mais quer comentar?", 2),
        (6, COMENTARIO, 6),
        (D, "Exatamente. Vamos ao parágrafo 5.", 3),
    ]
    frases, segs, embed = _reuniao(_estudo(extra))
    res = estudo.aplica(frases, segs, [], embed, LISTA, [])
    t = _inicio_do_texto(frases, "Jeová cuida", 8) + 1
    assert _nome_em(res, t)[1] == "[?]"


def test_chamada_e_agradecimento_que_discordam_viram_duvida():
    extra = [
        (D, "Mais um comentário? Irmã Lia, por favor.", 3),
        (6, COMENTARIO, 6),
        (D, "Obrigado, irmão Davi.", 2),
    ]
    frases, segs, embed = _reuniao(_estudo(extra))
    res = estudo.aplica(frases, segs, [], embed, LISTA, [])
    t = _inicio_do_texto(frases, "Jeová cuida", 8) + 1
    assert _nome_em(res, t)[1] == "[?]"


def test_homonimo_da_lista_vira_duvida():
    lista = LISTA + [
        irmaos.Entrada("m1", "Matheus", "Matheus 1"),
        irmaos.Entrada("m2", "Matheus", "Matheus 2"),
    ]
    extra = [
        (D, "Mais um comentário? Irmão Matheus, por favor.", 3),
        (6, COMENTARIO, 6),
        (D, "Obrigado, irmão Matheus.", 2),
    ]
    frases, segs, embed = _reuniao(_estudo(extra))
    res = estudo.aplica(frases, segs, [], embed, lista, [])
    t = _inicio_do_texto(frases, "Jeová cuida", 8) + 1
    assert _nome_em(res, t)[1] == "[?] Matheus (homônimo)"


# --------------------------------------------------------------------------
# (V) troca de voz e (G) dúvidas agrupadas
# --------------------------------------------------------------------------
def test_troca_de_voz_no_trecho_corta_o_chamado():
    extra = [
        (D, "Mais um comentário? Irmão Silas, por favor.", 3),
        (6, COMENTARIO, 7),
        (6, COMENTARIO, 5),
        (7, "E eu queria acrescentar que a oração também ajuda muito.", 5),
        (7, "Porque Jeová escuta cada pedido que fazemos a ele.", 5),
        (D, "Exatamente. Vamos ao parágrafo 5.", 3),
    ]
    frases, segs, embed = _reuniao(_estudo(extra))
    res = estudo.aplica(frases, segs, [], embed, LISTA, [])
    assert _nome_em(res, _inicio_do_texto(frases, "Jeová cuida", 8) + 1)[1] == "Silas"
    # a outra voz não fica com o nome do chamado: sem agradecimento, "[?]"
    # (com "Obrigado, irmão X" logo depois, o trecho cortado fica com X, como
    # no protótipo medido)
    assert _nome_em(res, _inicio_do_texto(frases, "acrescentar") + 1)[1] == "[?]"
    assert res.info["cortes_voz_trecho"] >= 1


def test_duvidas_com_a_mesma_voz_ficam_juntas_e_sem_nome():
    extra = [
        (D, "Alguém mais quer comentar?", 2),
        (6, COMENTARIO, 6),
        (D, "Exatamente. E sobre o versículo?", 3),
        (6, COMENTARIO, 6),
        (D, "Exatamente. Vamos ao parágrafo 5.", 3),
    ]
    frases, segs, embed = _reuniao(_estudo(extra))
    res = estudo.aplica(frases, segs, [], embed, LISTA, [])
    a = _nome_em(res, _inicio_do_texto(frases, "Jeová cuida", 8) + 1)
    b = _nome_em(res, _inicio_do_texto(frases, "Jeová cuida", 9) + 1)
    assert a == b and a[1] == "[?]"
    assert res.info["duvidas_agrupadas"] == 1


def test_texto_de_fora_do_estudo_fica_igual():
    itens = [(5, "Bom dia a todos, sejam bem-vindos à nossa reunião.", 5)] + _estudo()
    frases, segs, embed = _reuniao(itens)
    res = estudo.aplica(frases, segs, [], embed, LISTA, [])
    assert res.segmentos[0] == segs[0]
    palavras = " ".join(s.text for s in res.segmentos).split()
    assert palavras == " ".join(s.text for s in segs).split(), "nenhuma palavra some"


# --------------------------------------------------------------------------
# Worker e memória de vozes: "[?]" nunca vira nome
# --------------------------------------------------------------------------
@pytest.fixture
def dados(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    config.JOBS_DIR.mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_worker_mostra_duvida_sem_trocar_por_nome_da_conversa(dados):
    segs = [
        Segment(0, 3, "A", "Bom dia, aqui é a Rosilene."),
        Segment(4, 8, "Q1", "Eu acho que sim, Rosilene."),
        Segment(9, 12, "Q2", "Concordo com tudo isso."),
    ]
    job = jobs.new_job(
        filename="r.wav",
        engine="local",
        language="pt",
        min_speakers=None,
        max_speakers=None,
        want_summary=False,
    )
    job.falantes_estudo = [
        {
            "rotulo": "Q1",
            "nome": "[?]",
            "papel": "comentarista",
            "origem": "sem chamada",
            "comentarios": 1,
            "segundos": 4.0,
            "nomeada": False,
            "duvida": True,
        },
        {
            "rotulo": "Q2",
            "nome": "[?] Jorge?",
            "papel": "comentarista",
            "origem": "chamada do dirigente; fora da lista, 1 menção",
            "comentarios": 1,
            "segundos": 3.0,
            "nomeada": False,
            "duvida": True,
        },
    ]
    job.save()
    worker._auto_names(jobs.load(job.id), segs, {})
    auto = jobs.load(job.id).nomes_auto
    assert auto["Falante 2"]["nome"] == "[?]" and auto["Falante 2"]["fonte"] == "duvida"
    assert auto["Falante 3"]["nome"] == "[?] Jorge?"
    review.save_raw(jobs.load(job.id), segs)
    review.render_outputs(job.id)
    dados_json = json.loads(
        jobs.load(job.id).path("transcricao.json").read_text("utf-8")
    )
    assert [s["falante"] for s in dados_json["segmentos"]][1:] == ["[?]", "[?] Jorge?"]


def test_memoria_de_vozes_nunca_aprende_duvida(dados):
    v = list(_voz(1).astype(float))
    aprendidos = vozes.learn(
        "job1",
        {"A": "[?] Jorge?", "B": "Ana"},
        {
            "A": {"vetor": v, "segundos": 30},
            "B": {"vetor": list(_voz(2).astype(float)), "segundos": 30},
        },
    )
    assert aprendidos == ["Ana"]
    assert [p["nome"] for p in vozes.list_people()] == ["Ana"]


# --------------------------------------------------------------------------
# Sugestão "[?] X?": só candidato com cara de nome
# --------------------------------------------------------------------------
def _sugestao(chamada: str, participantes=()) -> str:
    extra = [(D, chamada, 3), (6, COMENTARIO, 6), (D, "Vamos ao parágrafo 5.", 2)]
    frases, segs, embed = _reuniao(_estudo(extra))
    res = estudo.aplica(frases, segs, [], embed, LISTA, list(participantes))
    return _nome_em(res, _inicio_do_texto(frases, "Jeová cuida", 8) + 1)[1]


def test_palavra_comum_nao_vira_sugestao():
    # "Temos?" casa com a gramática de chamada ("X?"), mas não é nome
    assert _sugestao("E o que mais nós Temos?") == "[?]"


def test_nome_biblico_do_texto_nao_vira_sugestao():
    assert _sugestao("O que aprendemos com Moisés?") == "[?]"


def test_palavra_sem_titulo_e_sem_semelhanca_nao_vira_sugestao():
    assert _sugestao("E sobre o mundo Digital?") == "[?]"


def test_sugestao_depois_de_irmao_aparece():
    assert _sugestao("Mais alguém? Irmã Glaucia, por favor.") == "[?] Glaucia?"


def test_sugestao_parecida_com_participante_aparece():
    # "Glauce" não está na lista, mas é parecido com o participante "Glaucia"
    assert _sugestao("Mais alguém? Glauce?", ["Glaucia"]) == "[?] Glaucia?"


def test_sugestao_nao_muda_o_rotulo_do_trecho():
    # trocar a sugestão por "[?]" é só exibição: a partição fica igual
    extra = [
        (D, "E o que mais nós Temos?", 3),
        (6, COMENTARIO, 6),
        (D, "Vamos ao parágrafo 5.", 2),
    ]
    frases, segs, embed = _reuniao(_estudo(extra))
    res = estudo.aplica(frases, segs, [], embed, LISTA, [])
    rot = _nome_em(res, _inicio_do_texto(frases, "Jeová cuida", 8) + 1)[0]
    assert rot.startswith("Q")
    assert sum(1 for p in res.pessoas if p.rotulo == rot) == 1


def test_correcao_que_troca_a_inicial_nao_vira_sugestao():
    # "Ena" ~ "Deena" (0,75) pela semelhança, mas a inicial muda: não é ela
    # (no salão, "Ana, Ana" virou "Deana?" e era a Edna)
    assert _sugestao("E aí Ena, Ena, por favor.", ["Deena"]) == "[?]"


def test_chamada_com_por_favor_vale_como_evidencia():
    assert _sugestao("Mais alguém? Por favor, Glaucia.") == "[?] Glaucia?"


def test_nome_composto_com_segunda_parte_desconhecida_fica_so_o_primeiro():
    assert _sugestao("Mais alguém? Irmão Fernando Cross.") == "[?] Fernando?"


def test_nome_composto_sem_primeiro_nome_conhecido_vira_duvida():
    # "né, Monja Ilson?" era o Jailson: "Ilson" parece Jailson, "Monja" não
    lista = LISTA + [irmaos.Entrada("j", "Jailson")]
    extra = [(D, "Para reforçar, né, Monja Ilson?", 3), (6, COMENTARIO, 6),
             (D, "Vamos ao parágrafo 5.", 2)]
    frases, segs, embed = _reuniao(_estudo(extra))
    res = estudo.aplica(frases, segs, [], embed, lista, [])
    assert _nome_em(res, _inicio_do_texto(frases, "Jeová cuida", 8) + 1)[1] == "[?]"


def test_chamada_com_artigo_vale_no_estudo():
    extra = [(D, "E informou o Silas?", 2), (D, COMENTARIO, 6),
             (D, "Isso mesmo. Obrigado. Parágrafo 5.", 3)]
    frases, segs, embed = _reuniao(_estudo(extra))
    res = estudo.aplica(frases, segs, [], embed, LISTA, [])
    assert _nome_em(res, _inicio_do_texto(frases, "Jeová cuida", 8) + 1)[1] == "Silas"
