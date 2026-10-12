"""Testes da memória de vozes e da dedução de nomes pela conversa.

Rodar:  python -m pytest test_vozes_nomes.py -v
"""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import config, jobs, review, vozes, worker
from app.engines.base import Segment
from app.main import app
from app.nomes import infer


@pytest.fixture(autouse=True)
def dados_temporarios(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)


def _voz(seed: int, ruido: float = 0.0) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=192)
    if ruido:
        v = v + ruido * np.random.default_rng(seed + 100).normal(size=192)
    return list(v / np.linalg.norm(v))


# --------------------------------------------------------------------------
# Memória de vozes
# --------------------------------------------------------------------------
def test_reconhece_voz_confirmada_em_outra_reuniao():
    vozes.learn("job1", {"A": "Fernando", "B": "Eduardo"},
                {"A": {"vetor": _voz(1), "segundos": 600}, "B": {"vetor": _voz(2), "segundos": 400}})
    # reunião nova: mesmas pessoas, com variação de timbre/microfone, ordem trocada
    achou = vozes.match({"A": _voz(2, 0.4), "B": _voz(1, 0.4)})
    assert achou["A"][0] == "Eduardo" and achou["B"][0] == "Fernando"


def test_voz_desconhecida_nao_vira_ninguem():
    vozes.learn("job1", {"A": "Fernando"}, {"A": {"vetor": _voz(1), "segundos": 60}})
    assert vozes.match({"A": _voz(99)}) == {}


def test_confirmar_de_novo_substitui_e_renomear_move():
    voices = {"A": {"vetor": _voz(1), "segundos": 60}}
    vozes.learn("job1", {"A": "Fernando"}, voices)
    vozes.learn("job1", {"A": "Fernando"}, voices)          # salvou 2x a revisão
    assert vozes.list_people()[0]["reunioes"] == 1
    vozes.learn("job1", {"A": "Fernando Souza"}, voices)    # corrigiu o nome
    assert [p["nome"] for p in vozes.list_people()] == ["Fernando Souza"]
    vozes.learn("job1", {"A": ""}, voices)                   # apagou o nome
    assert vozes.list_people() == []


def test_aprende_com_varias_reunioes_e_esquece():
    for i in range(3):
        vozes.learn(f"job{i}", {"A": "Ana"}, {"A": {"vetor": _voz(5, 0.3 + i / 10), "segundos": 300}})
    pessoa = vozes.list_people()[0]
    assert pessoa["reunioes"] == 3 and pessoa["minutos"] == 15.0
    assert vozes.forget("Ana") and vozes.list_people() == []
    assert not vozes.forget("Ana")


def test_memoria_de_outro_modelo_e_guardada_a_parte(monkeypatch):
    vozes.learn("job1", {"A": "Fernando"}, {"A": {"vetor": _voz(1), "segundos": 60}})
    monkeypatch.setattr(config, "DIARIZATION_MODEL", "/x/outro_modelo.onnx")
    assert vozes.list_people() == []
    assert (config.DATA_DIR / "vozes" / "vozes.nemo_en_titanet_small.json").exists()


def test_arquivo_e_json_legivel():
    vozes.learn("job1", {"A": "Bia"}, {"A": {"vetor": _voz(3), "segundos": 10}})
    data = json.loads((config.DATA_DIR / "vozes" / "vozes.json").read_text())
    assert "Bia" in data["pessoas"]


# --------------------------------------------------------------------------
# Nomes pela conversa
# --------------------------------------------------------------------------
def _seg(t, spk, txt):
    return Segment(t, t + 3, spk, txt)


def test_quem_e_chamado_pelo_nome_e_o_outro():
    segs = [_seg(0, "A", "Oi, tudo bem?"), _seg(4, "B", "Tudo, e você?"),
            _seg(8, "A", "Então fechou, Fernando."), _seg(12, "B", "Fechou."),
            _seg(16, "A", "Fernando, me liga amanhã."), _seg(20, "B", "Ligo sim.")]
    assert infer(segs)["B"][0] == "Fernando"
    assert "A" not in infer(segs)


def test_fala_citada_nao_conta():
    segs = [_seg(0, "A", "Ele disse: você demorou demais, Fernando."), _seg(4, "B", "Pois é."),
            _seg(8, "A", "Ele falou, Fernando."), _seg(12, "B", "Sim.")]
    assert infer(segs) == {}


def test_apresentacao_e_grafia_parecida_com_participante():
    segs = [_seg(0, "A", "Bom dia, aqui é a Rosilene."), _seg(4, "B", "Bom dia.")]
    assert infer(segs, ["Rosilaine Andrade"])["A"][0] == "Rosilaine Andrade"


def test_palavras_comuns_nao_viram_nome():
    segs = [_seg(0, "A", "Então, Beleza."), _seg(4, "B", "Exatamente."),
            _seg(8, "A", "Perfeito, Combinado."), _seg(12, "B", "Valeu, Obrigado.")]
    assert infer(segs) == {}


def test_nao_sugere_para_quem_ja_tem_nome_pela_voz():
    segs = [_seg(0, "A", "Valeu, Fernando."), _seg(4, "B", "Valeu."),
            _seg(8, "A", "Até mais, Fernando."), _seg(12, "B", "Tchau.")]
    assert infer(segs, exclude={"B"}) == {}


# --------------------------------------------------------------------------
# Sugestão por eliminação
# --------------------------------------------------------------------------
from app.nomes import ELIMINACAO_DETALHE, por_eliminacao  # noqa: E402


def test_eliminacao_dois_participantes_um_deduzido():
    # (a) 2 participantes, 1 falante já deduzido -> o outro recebe o restante
    sugerido = por_eliminacao(["Falante 1", "Falante 2"],
                              ["Eduardo Moretti", "Fernando TopVigia"],
                              {"Falante 2": "Fernando TopVigia"})
    assert sugerido == {"Falante 1": "Eduardo Moretti"}
    assert ELIMINACAO_DETALHE == "único participante restante"


def test_eliminacao_tolerante_a_caixa_acento_e_primeiro_nome():
    # o nome deduzido pode ser só o primeiro nome / outra caixa
    sugerido = por_eliminacao(["Falante 1", "Falante 2"],
                              ["Eduardo Moretti", "José Fernando"],
                              {"Falante 1": "FERNANDO"})
    assert sugerido == {"Falante 2": "Eduardo Moretti"}


def test_eliminacao_nao_sugere_com_3_participantes_e_2_vozes():
    # (b) nº de falantes diferente do nº de participantes -> não sugere
    assert por_eliminacao(["Falante 1", "Falante 2"],
                          ["Ana", "Bia", "Carla"], {"Falante 2": "Bia"}) == {}


def test_eliminacao_sem_participantes_nao_sugere():
    # (c) participantes vazios
    assert por_eliminacao(["Falante 1", "Falante 2"], [], {}) == {}
    assert por_eliminacao(["Falante 1", "Falante 2"], ["", "  "], {}) == {}


def test_eliminacao_dois_sem_nome_e_dois_livres_nao_sugere():
    # (d) 2 falantes sem nome e 2 participantes livres -> não sugere
    assert por_eliminacao(["Falante 1", "Falante 2"],
                          ["Eduardo Moretti", "Fernando TopVigia"], {}) == {}



# --------------------------------------------------------------------------
# Fluxo real: sugestão automática -> revisão -> memória de vozes
# (o que a tela envia é o que está no campo; sugestão só vira nome se o
#  usuário confirmar, digitando ou clicando em "Usar")
# --------------------------------------------------------------------------
@pytest.fixture
def cliente(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    config.JOBS_DIR.mkdir(parents=True, exist_ok=True)
    return TestClient(app)


def _job_pronto(segmentos, voices, participantes=()):
    """Job concluído com transcrição crua e assinaturas de voz, como o worker deixa."""
    job = jobs.new_job(filename="reuniao.wav", engine="local", language="pt",
                       min_speakers=None, max_speakers=None, want_summary=False)

    def concluir(j: jobs.Job) -> None:
        j.status = jobs.STATUS_DONE
        j.participants = list(participantes)
    jobs.update(job.id, concluir)
    review.save_raw(job, segmentos)
    job.path("vozes.json").write_text(json.dumps(voices), encoding="utf-8")
    return job.id


def _corpo_da_tela(rv: dict) -> dict:
    """Replica coletarRevisao() no front: só entram os campos preenchidos."""
    return {"speaker_names": {s["label"]: s["nome"] for s in rv["speakers"] if s["nome"]}}


def test_sugestao_por_conversa_so_aprende_o_que_for_confirmado(cliente):
    segs = [_seg(0, "A", "Bom dia, aqui é a Rosilene."), _seg(4, "B", "Bom dia."),
            _seg(8, "A", "Vamos começar."), _seg(12, "B", "Pode ser.")]
    voices = {"A": {"vetor": _voz(1), "segundos": 300}, "B": {"vetor": _voz(2), "segundos": 200}}
    jid = _job_pronto(segs, voices, participantes=["Rosilaine Andrade", "Eduardo Moretti"])

    # 1) o worker deduz nomes: vão só para nomes_auto
    worker._auto_names(jobs.load(jid), segs, voices)
    review.render_outputs(jid)
    job = jobs.load(jid)
    assert job.speaker_names == {}, "sugestão não pode virar nome confirmado"
    assert job.participants == ["Rosilaine Andrade", "Eduardo Moretti"]
    assert job.nomes_auto["Falante 1"]["nome"] == "Rosilaine Andrade"
    assert job.nomes_auto["Falante 2"]["fonte"] == "eliminacao"
    txt = job.path("transcricao.txt").read_text(encoding="utf-8")
    # a exibição usa a sugestão (pedido do Eduardo); a memória de vozes continua
    # aprendendo só o que for confirmado (ver o restante deste teste)
    assert "Rosilaine Andrade · " in txt and "Falante 1 · " not in txt, \
        "transcrição deve mostrar a sugestão como nome"

    # 2) a tela mostra a sugestão com o campo vazio
    rv = cliente.get(f"/api/jobs/{jid}/review").json()
    assert [(s["nome"], s["auto"]["nome"]) for s in rv["speakers"]] == [
        ("", "Rosilaine Andrade"), ("", "Eduardo Moretti")]

    # 3) salvar a revisão sem mexer nos campos não aprende nada
    r = cliente.put(f"/api/jobs/{jid}/review", json=_corpo_da_tela(rv)).json()
    assert r["vozes_aprendidas"] == [] and vozes.list_people() == []

    # 4) o usuário aceita só a primeira sugestão ("Usar"); a segunda fica vazia
    r = cliente.put(f"/api/jobs/{jid}/review",
                    json={"speaker_names": {"Falante 1": "Rosilaine Andrade"}}).json()
    assert r["vozes_aprendidas"] == ["Rosilaine Andrade"]
    assert [p["nome"] for p in vozes.list_people()] == ["Rosilaine Andrade"]
    txt = job.path("transcricao.txt").read_text(encoding="utf-8")
    # o 2º falante ainda não foi confirmado: a sugestão aparece no arquivo, mas
    # a memória de vozes não aprende o que não foi confirmado
    assert "Rosilaine Andrade · " in txt and "Eduardo Moretti · " in txt
    assert [p["nome"] for p in vozes.list_people()] == ["Rosilaine Andrade"]


def test_sugestao_por_voz_nao_reforca_a_memoria(cliente):
    # reunião anterior confirmada pelo usuário: Fernando
    vozes.learn("job_antiga", {"A": "Fernando"}, {"A": {"vetor": _voz(1), "segundos": 600}})
    segs = [_seg(0, "A", "Vamos ver o orçamento."), _seg(4, "B", "Tudo certo."),
            _seg(8, "A", "Combinado."), _seg(12, "B", "Ótimo.")]
    voices = {"A": {"vetor": _voz(1), "segundos": 300}, "B": {"vetor": _voz(2), "segundos": 200}}
    jid = _job_pronto(segs, voices)

    worker._auto_names(jobs.load(jid), segs, voices)
    job = jobs.load(jid)
    assert job.nomes_auto["Falante 1"]["fonte"] == "voz"
    assert job.speaker_names == {}

    # salvar sem confirmar: a pessoa continua com 1 reunião (só a antiga)
    rv = cliente.get(f"/api/jobs/{jid}/review").json()
    cliente.put(f"/api/jobs/{jid}/review", json=_corpo_da_tela(rv))
    assert [(p["nome"], p["reunioes"]) for p in vozes.list_people()] == [("Fernando", 1)]

    # confirmar digitando: aprende nesta reunião também
    r = cliente.put(f"/api/jobs/{jid}/review",
                    json={"speaker_names": {"Falante 1": "Fernando"}}).json()
    assert r["vozes_aprendidas"] == ["Fernando"]
    assert [(p["nome"], p["reunioes"]) for p in vozes.list_people()] == [("Fernando", 2)]


# --------------------------------------------------------------------------
# Palavras comuns e acento (regressão do "Então" sugerido 13×)
# --------------------------------------------------------------------------
def test_entao_com_acento_nunca_vira_nome():
    segs = [Segment(0, 3, "A", "Então, Fernando, vamos lá."),
            Segment(4, 6, "B", "Sim, então fechou."),
            Segment(7, 9, "A", "Então, Fernando, a gente começa.")]
    nomes = [v[0] for v in infer(segs).values()]
    assert "Então" not in nomes
    assert "Fernando" in nomes                  # o nome próprio continua valendo


def test_como_nos_e_jeova_nao_viram_nome():
    segs = [Segment(0, 3, "A", "Como você está?"),
            Segment(4, 6, "B", "Nós vamos, Jeová ajuda."),
            Segment(7, 9, "A", "Jeová é fiel.")]
    nomes = [v[0] for v in infer(segs).values()]
    assert not {"Como", "Nós", "Jeová"} & set(nomes)


def test_palavra_minuscula_no_meio_da_frase_nao_e_nome():
    from app.nomes import _proper_nouns

    segs = [Segment(0, 3, "A", "Vamos lá. Ele disse que Marcos vai. Marcos é o nome."),
            Segment(4, 6, "B", "e assim marcos resolveu isso.")]
    assert "Marcos" not in _proper_nouns(segs)


# --------------------------------------------------------------------------
# Reunião com vários falantes: chamada, agradecimento e dirigente
# --------------------------------------------------------------------------
from app.nomes import dirigente_de, _unifica  # noqa: E402


def _reuniao():
    """D é o dirigente: chama antes do comentário e agradece depois."""
    return [
        _seg(0, "D", "Vamos começar. Quem comenta primeiro? Sim, irmão Fulano."),
        _seg(3, "A", "Bom dia a todos. Eu acho que a gente deve orar."),
        _seg(9, "D", "Obrigado, irmão Fulano. Irmã Beltrana, por favor."),
        _seg(12, "B", "Olá, eu acho que Jeová se importa."),
        _seg(18, "D", "Muito bem, irmã Beltrana. Vamos ao próximo."),
        _seg(25, "C", "Pois não, é isso mesmo, a gente precisa de paciência."),
        _seg(31, "D", "Obrigado, irmão Ciclano."),
    ]


def test_chamada_vai_para_quem_responde_logo_depois():
    # "Sim, irmão Fulano." -> quem fala a seguir (A) é o Fulano
    s = _reuniao()
    nomes = infer(s)
    assert nomes["A"][0] == "Fulano"
    assert nomes["A"][2] == "chamado pelo dirigente 1×, agradecido 1×"


def test_agradecimento_vai_para_quem_acabou_de_falar():
    # "Obrigado, irmão Fulano." (t=9) -> o falante anterior (A) é o Fulano;
    # "Muito bem, irmã Beltrana." (t=18) -> o anterior (B) é a Beltrana
    s = _reuniao()
    nomes = infer(s)
    assert nomes["B"][0] == "Beltrana"
    # um único agradecimento já vale o mínimo (peso posicional 2): sugere
    assert nomes["C"][0] == "Ciclano" and nomes["C"][2] == "agradecido 1×"


def test_dirigente_nao_recebe_o_nome_que_ele_pronuncia():
    s = _reuniao()
    assert dirigente_de(s) == "D"
    assert "D" not in infer(s)


def test_nome_composto_e_titulo_fora_do_nome():
    s = [_seg(0, "D", "Sim, irmão José Eduardo."), _seg(4, "A", "Pois é, bom dia."),
         _seg(8, "D", "Obrigado, irmão José Eduardo."), _seg(12, "B", "Amém."),
         _seg(16, "D", "Obrigado, irmão José Eduardo.")]
    nomes = [v[0] for v in infer(s).values()]
    assert "José Eduardo" in nomes
    assert "irmão" not in " ".join(nomes).casefold()


def test_grafias_parecidas_viram_uma():
    assert _unifica(["Admir", "Ademir", "Ademir"]) == {"Admir": "Ademir", "Ademir": "Ademir"}
    assert _unifica(["Ana", "Bia"]) == {"Ana": "Ana", "Bia": "Bia"}


def test_fala_citada_de_chamada_nao_conta():
    # "ele disse: Sim, irmão Fulano" é citação, não chamada de agora
    s = [_seg(0, "D", "Ele disse: sim, irmão Fulano, vem aqui."), _seg(4, "A", "Tá bom.")]
    assert "A" not in infer(s)


def test_participante_vence_grafia_da_lista_na_conversa():
    # 2026-10-12: "Fernando" cadastrado como grafia de "Fernando Croci" fazia o
    # Fernando TopVigia (participante desta reunião) virar Fernando Croci
    from app.irmaos import Entrada
    lista = [Entrada("1", "Fernando Croci", grafias=("Fernando Cross", "Fernando"))]
    segs = [_seg(0, "A", "Então fechou, Fernando."), _seg(4, "B", "Fechou."),
            _seg(8, "A", "Valeu, Fernando."), _seg(12, "B", "Até mais.")]
    nomes = infer(segs, ["Eduardo Moretti", "Fernando TopVigia"], irmaos=lista)
    assert nomes["B"][0] == "Fernando TopVigia"
