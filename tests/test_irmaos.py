"""Testes da lista de irmãos (nomes conhecidos), sem modelos e sem biometria.

Rodar:  python -m pytest tests/test_irmaos.py -v
"""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import config, irmaos, main, nomes, turnos
from app.engines import local_engine
from app.engines.base import Segment


@pytest.fixture(autouse=True)
def dados_temporarios(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)


def _lista(*itens):
    """Entradas em memória: _lista(("Matheus", "1"), ("Ninfa", "", ["Niffa"]))."""
    out = []
    for i, item in enumerate(itens):
        nome, obs, *resto = item
        grafias = tuple(resto[0]) if resto else ()
        out.append(irmaos.Entrada(id=f"{i:012x}", nome=nome, observacao=obs,
                                  grafias=grafias, atualizado=float(100 - i)))
    return out


EXEMPLO = _lista(
    ("José Eduardo", ""), ("Ninfa", "", ["Niffa"]), ("Janice", "", ["Danice"]),
    ("Denise", ""), ("Marines", "", ["Marilenez", "Marileneza"]), ("Nadir", "", ["Nadeira"]),
    ("Edna", "", ["Helio"]), ("Mateus", "", ["Marteus"]),
    ("Matheus", "1"), ("Matheus", "2"), ("Caio", ""), ("Catarina", ""),
)


# --------------------------------------------------------------------------
# Correção de grafia
# --------------------------------------------------------------------------
def test_nome_exato_ignora_caixa_e_acento():
    c = irmaos.corrige("fátima", _lista(("Fátima", "")))
    assert c.casou and c.nome == "Fátima" and c.origem == "exato" and not c.ambiguo


def test_grafia_parecida_vai_para_o_nome_da_lista():
    assert irmaos.corrige("Niffa", EXEMPLO).nome == "Ninfa"        # 0,80 de semelhança
    assert irmaos.corrige("Nadeira", EXEMPLO).nome == "Nadir"
    assert irmaos.corrige("Marteus", EXEMPLO).nome == "Mateus"


def test_danice_vai_para_janice_e_nao_denise():
    c = irmaos.corrige("Danice", EXEMPLO)
    assert c.nome == "Janice"


def test_grafia_cadastrada_corrige_mesmo_com_pouca_semelhanca():
    # "Marileneza" tem só 0,71 de semelhança; a grafia cadastrada resolve.
    c = irmaos.corrige("Marileneza", EXEMPLO)
    assert c.nome == "Marines" and c.origem == "grafia"
    assert irmaos.corrige("Helio", EXEMPLO).nome == "Edna"


def test_parte_do_nome_casa_com_nome_completo():
    assert irmaos.corrige("Eduardo", EXEMPLO).nome == "José Eduardo"


def test_nome_fora_da_lista_volta_como_veio():
    c = irmaos.corrige("Rosilene", EXEMPLO)
    assert not c.casou and c.nome == "Rosilene"


def test_sem_lista_nao_corrige_nada():
    c = irmaos.corrige("Niffa", [])
    assert not c.casou and c.nome == "Niffa"


def test_empate_entre_dois_nomes_nao_corrige():
    lista = _lista(("Ana Rita", ""), ("Ana Lúcia", ""))
    c = irmaos.corrige("Ana", lista)
    # "Ana" casa igual com os dois nomes: não corrige por semelhança
    assert not c.casou


# --------------------------------------------------------------------------
# Nomes exatos não se fundem (Mateus ≠ Matheus; Denise ≠ Janice)
# --------------------------------------------------------------------------
def test_mateus_exato_fica_mateus_e_nao_vira_matheus():
    c = irmaos.corrige("Mateus", EXEMPLO)
    assert c.nome == "Mateus" and c.origem == "exato" and not c.ambiguo


def test_unifica_nao_funde_nomes_exatos_da_lista():
    protegidos = irmaos.nomes_protegidos(EXEMPLO)
    mapa = nomes._unifica(["Mateus", "Matheus", "Mateus"], protegidos)
    assert mapa["Mateus"] == "Mateus" and mapa["Matheus"] == "Matheus"


def test_sem_lista_o_antigo_unifica_mateus_e_matheus():
    # controle: é exatamente o que a lista impede
    mapa = nomes._unifica(["Mateus", "Matheus", "Mateus"])
    assert mapa["Matheus"] == "Mateus"


def test_denise_e_janice_continuam_separadas():
    protegidos = irmaos.nomes_protegidos(EXEMPLO)
    mapa = nomes._unifica(["Denise", "Janice", "Janice"], protegidos)
    assert mapa["Denise"] == "Denise" and mapa["Janice"] == "Janice"


# --------------------------------------------------------------------------
# Homônimos
# --------------------------------------------------------------------------
def test_nome_com_dois_homonimos_e_ambiguo_com_observacao():
    c = irmaos.corrige("Matheus", EXEMPLO)
    assert c.casou and c.ambiguo
    assert c.nome == "Matheus"
    assert c.observacao == "2 pessoas com esse nome; confirme pela voz"


def test_rotulo_mostra_a_observacao():
    entradas = _lista(("Matheus", "1"), ("Matheus", "2"))
    assert [e.rotulo for e in entradas] == ["Matheus (1)", "Matheus (2)"]


# --------------------------------------------------------------------------
# Hotwords: frase pontuada, limite de tamanho, sem corte no meio
# --------------------------------------------------------------------------
def test_frase_de_irmaos_e_pontuada_e_sem_lista_solta():
    frase = irmaos.frase_irmaos(_lista(("Ninfa", ""), ("Janice", ""), ("Marines", ""),
                                       ("Nadir", "")), limite=160)
    assert frase == "Participam irmãos como Ninfa, Janice, Marines e Nadir."


def test_frase_respeita_o_limite_e_fecha_com_ponto():
    grandes = _lista(*[(f"Nome{i:02d}Longo", "") for i in range(60)])
    frase = irmaos.frase_irmaos(grandes, limite=120)
    assert len(frase) <= 120 and frase.endswith(".")


def test_frase_prefere_os_mais_recentes():
    entradas = [
        irmaos.Entrada("a" * 12, "Antigo", atualizado=1.0),
        irmaos.Entrada("b" * 12, "Recente", atualizado=9.0),
    ]
    assert irmaos.frase_irmaos(entradas, limite=35) == "Participam irmãos como Recente."


def test_frase_nao_repete_nomes_ja_citados():
    frase = irmaos.frase_irmaos(_lista(("Ninfa", ""), ("Janice", "")), 160,
                                excluir=["Ninfa"])
    assert frase == "Participam irmãos como Janice."


def test_hotwords_do_motor_une_participantes_e_irmaos_em_frases():
    texto = local_engine._hotwords(["Fernando"], "pt", _lista(("Ninfa", ""), ("Janice", "")))
    assert texto.startswith("Reunião com Fernando.")
    assert "Participam irmãos como Ninfa e Janice." in texto
    assert "initial_prompt" not in texto


def test_hotwords_sem_lista_fica_como_antes():
    assert local_engine._hotwords(["Fernando"], "pt", []) == "Reunião com Fernando."


def test_hotwords_nunca_corta_no_meio_de_palavra():
    texto = local_engine._corta("Reunião com A. " + "Participam irmãos como " + "X" * 500 + ".",
                                local_engine.HOTWORDS_MAX_CHARS)
    assert len(texto) <= local_engine.HOTWORDS_MAX_CHARS
    assert texto.endswith(".") or texto == "Reunião com A."


# --------------------------------------------------------------------------
# Armazenamento e validação
# --------------------------------------------------------------------------
def test_crud_grava_so_nomes_em_nomes_json():
    e = irmaos.criar("Matheus", "1", ["Matheu"])
    assert (config.DATA_DIR / "nomes" / "nomes.json").exists()
    gravado = json.loads((config.DATA_DIR / "nomes" / "nomes.json").read_text(encoding="utf-8"))
    assert set(gravado["nomes"][0]) <= {"id", "nome", "observacao", "grafias", "atualizado"}
    assert irmaos.listar()[0].id == e.id
    atualizado = irmaos.atualizar(e.id, "Matheus", "2", [])
    assert atualizado.observacao == "2" and atualizado.grafias == ()
    assert irmaos.apagar(e.id) is True
    assert irmaos.listar() == []
    assert irmaos.apagar(e.id) is False


def test_valida_nome_e_observacao():
    with pytest.raises(irmaos.IrmaoError):
        irmaos.criar("Ana123")                    # número
    with pytest.raises(irmaos.IrmaoError):
        irmaos.criar("<script>")                  # símbolo
    with pytest.raises(irmaos.IrmaoError):
        irmaos.criar("Ana", "x" * 81)             # observação longa
    with pytest.raises(irmaos.IrmaoError):
        irmaos.criar("Ana", "", ["Ana<b>"])       # grafia inválida


def test_nao_aceita_duplicata_com_mesma_observacao():
    irmaos.criar("Matheus", "1")
    with pytest.raises(irmaos.IrmaoError):
        irmaos.criar("matheus", "1")              # caixa ignorada
    irmaos.criar("Matheus", "2")                  # homônimo com outra observação: ok


def test_lista_ilegivel_nao_derruba_a_reuniao():
    f = config.DATA_DIR / "nomes" / "nomes.json"
    f.parent.mkdir(parents=True)
    f.write_text("{não é json", encoding="utf-8")
    with pytest.raises(irmaos.IrmaoError):
        irmaos.listar()
    assert irmaos.listar_seguro() == []


def test_ausentes_so_lista_o_que_falta():
    entradas = _lista(("Ninfa", "", ["Niffa"]))
    assert irmaos.ausentes(["Niffa", "Janice", "Janice", "", "Ninfa"], entradas) == ["Janice"]


# --------------------------------------------------------------------------
# API /api/nomes (token de sessão, validação)
# --------------------------------------------------------------------------
@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(main, "_SESSION_TOKEN", "tok-de-teste")
    return TestClient(main.app)


def test_api_sem_token_recebe_403(api):
    assert api.get("/api/nomes").status_code == 403
    assert api.post("/api/nomes", json={"nome": "Ninfa"}).status_code == 403
    assert api.put("/api/nomes/000000000000", json={"nome": "Ninfa"}).status_code == 403
    assert api.delete("/api/nomes/000000000000").status_code == 403


def test_api_crud_com_token(api):
    assert api.get("/api/nomes?t=tok-de-teste").status_code == 200   # cookie passa a valer
    r = api.post("/api/nomes", json={"nome": "Ninfa", "grafias": ["Niffa"]})
    assert r.status_code == 201, r.text
    criado = r.json()
    assert criado["rotulo"] == "Ninfa" and criado["grafias"] == ["Niffa"]

    lista = api.get("/api/nomes").json()["nomes"]
    assert [n["nome"] for n in lista] == ["Ninfa"]

    r = api.put(f"/api/nomes/{criado['id']}", json={"nome": "Ninfa", "observacao": "2"})
    assert r.status_code == 200 and r.json()["rotulo"] == "Ninfa (2)"

    assert api.delete(f"/api/nomes/{criado['id']}").json() == {"ok": True}
    assert api.get("/api/nomes").json()["nomes"] == []


def test_api_valida_entrada_e_id(api):
    api.get("/api/nomes?t=tok-de-teste")
    assert api.post("/api/nomes", json={"nome": "Ana123"}).status_code == 422
    assert api.post("/api/nomes", json={"nome": "A" * 121}).status_code == 422
    assert api.post("/api/nomes", json={"nome": "Ana", "grafias": ["x"] * 21}).status_code == 422
    assert api.put("/api/nomes/../x", json={"nome": "Ana"}).status_code in (404, 405)
    assert api.put("/api/nomes/zzzz", json={"nome": "Ana"}).status_code == 404
    assert api.delete("/api/nomes/000000000000").status_code == 404


# --------------------------------------------------------------------------
# Turnos com a lista: Ninfa/Janice/Marines/Nadir/Edna, oração final, co-dirigente
# --------------------------------------------------------------------------
def _voz(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=64)
    return v / np.linalg.norm(v)


def _seg(ini, fim, falante, texto):
    return Segment(float(ini), float(fim), falante, texto)


def _reuniao_com_nomes():
    """D chama Niffa (=Ninfa) e agradece Helio (=Edna); Danice (=Janice) e Denise são
    pessoas diferentes; Marinês e Marilenez são a mesma pessoa."""
    segs = [
        _seg(0, 3, "D", "Vamos começar o estudo da revista."),
        _seg(4, 6, "D", "Irmã Niffa, por favor."),
        _seg(7, 20, "A", "Eu acho que a oração ajuda muito."),
        _seg(21, 23, "D", "Obrigado, irmã Niffa. Irmã Danice, por favor."),
        _seg(24, 32, "B", "Concordo com o parágrafo que lemos."),
        _seg(33, 35, "D", "Irmã Denise, por favor."),
        _seg(36, 44, "C", "Eu lembro de uma experiência parecida."),
        _seg(45, 47, "D", "Obrigado, irmã Denise. Irmã Marinês, por favor."),
        _seg(48, 56, "E", "Comento a outra pergunta do estudo."),
        _seg(57, 59, "D", "Irmã Edna, por favor."),
        _seg(60, 70, "F", "Sou a Edna e tenho uma experiência."),
        _seg(71, 73, "D", "Obrigado, irmã Helio."),
        _seg(74, 80, "G", "Mais um comentário de outra pessoa."),
        _seg(81, 83, "D", "Irmã Marileneza, por favor."),
        _seg(84, 94, "H", "Mais um comentário da mesma irmã."),
    ]
    vozes = {
        "A": _voz(1), "B": _voz(2), "C": _voz(3), "E": _voz(4), "F": _voz(5),
        "G": _voz(6), "H": _voz(7),
    }

    def embed(a, b):
        for s in segs:
            if abs(s.start - a) < 1e-6:
                return vozes.get(s.speaker, _voz(9))
        return _voz(9)
    return segs, embed


def _nomes_das_pessoas(pessoas):
    return sorted(p.nome for p in pessoas if p.papel == "comentarista" and p.nomeada)


def test_turnos_com_a_lista_corrigem_grafias():
    segs, embed = _reuniao_com_nomes()
    _, pessoas, _ = turnos.reatribui(segs, embed, irmaos=EXEMPLO)
    nomeadas = _nomes_das_pessoas(pessoas)
    assert "Ninfa" in nomeadas and "Janice" in nomeadas and "Denise" in nomeadas
    assert "Nadir" not in nomeadas                     # não há Nadeira nesta reunião
    assert "Edna" in nomeadas and "Helio" not in nomeadas
    assert "Niffa" not in nomeadas and "Danice" not in nomeadas


def test_denise_e_janice_nao_se_fundem_mesmo_com_voz_igual():
    segs, _ = _reuniao_com_nomes()
    voz_igual = _voz(42)
    _, pessoas, _ = turnos.reatribui(segs, lambda a, b: voz_igual, irmaos=EXEMPLO)
    nomeadas = _nomes_das_pessoas(pessoas)
    assert "Denise" in nomeadas and "Janice" in nomeadas


def test_marilenez_e_marines_viram_uma_pessoa():
    segs, embed = _reuniao_com_nomes()
    _, pessoas, _ = turnos.reatribui(segs, embed, irmaos=EXEMPLO)
    marines = [p for p in pessoas if p.nome == "Marines"]
    assert len(marines) == 1 and marines[0].comentarios >= 2


def test_mateus_e_matheus_separados_quando_os_dois_estao_na_lista():
    segs = [
        _seg(0, 3, "D", "Vamos começar o estudo."),
        _seg(4, 6, "D", "Irmão Mateus, por favor."),
        _seg(7, 20, "A", "Comentário do Mateus."),
        _seg(21, 23, "D", "Obrigado, irmão Mateus. Irmão Matheus, por favor."),
        _seg(24, 40, "B", "Comentário do Matheus."),
        _seg(41, 43, "D", "Obrigado, irmão Matheus."),
    ]
    vozes = {"A": _voz(1), "B": _voz(2)}
    _, pessoas, _ = turnos.reatribui(
        segs, lambda a, b: vozes["A"] if a < 20 else vozes["B"], irmaos=EXEMPLO)
    nomeadas = _nomes_das_pessoas(pessoas)
    assert "Mateus" in nomeadas
    assert "Matheus" in nomeadas          # ambíguo, mas não é fundido com Mateus
    assert "Mateus" != "Matheus" and nomeadas.count("Mateus") == 1


def test_turno_com_homonimos_sai_marcado_para_confirmar_pela_voz():
    segs = [
        _seg(0, 3, "D", "Vamos começar o estudo."),
        _seg(4, 6, "D", "Irmão Matheus, por favor."),
        _seg(7, 20, "A", "Comentário."),
        _seg(21, 23, "D", "Obrigado, irmão Matheus. Irmão Caio, por favor."),
        _seg(24, 30, "B", "Outro comentário."),
        _seg(31, 33, "D", "Obrigado, irmão Caio."),
    ]
    _, pessoas, _ = turnos.reatribui(segs, lambda a, b: _voz(1), irmaos=EXEMPLO)
    p = next(p for p in pessoas if p.nome == "Matheus")
    assert "homônimos" in p.origem


def _reuniao_com_oracao(voz_oracao):
    """Discurso de G antes do estudo, estudo do dirigente F, oração final de quem
    D chamou ("o irmão Ademir nos represente em oração")."""
    segs = [_seg(i * 10, i * 10 + 8, "G", "Discurso bem longo sobre a fé.") for i in range(6)]
    segs += [
        _seg(100, 103, "F", "Vamos começar o estudo da revista."),
        _seg(104, 106, "F", "Irmão Caio, por favor."),
        _seg(107, 115, "A", "Comentário do Caio sobre a leitura."),
        _seg(116, 118, "F", "Obrigado, irmão Caio."),
        _seg(119, 130, "A", "Outro comentário do mesmo irmão."),
        _seg(131, 133, "F", "Obrigado, irmã Catarina."),
        _seg(134, 145, "C", "Comentário da Catarina."),
        _seg(146, 148, "F", "Obrigado, irmã Catarina."),
        _seg(150, 152, "D", "O irmão Ademir nos represente em oração, por favor."),
        _seg(160, 175, "B", "Querido Deus, obrigado por tudo."),
        _seg(176, 190, "I", "Pedimos a tua ajuda para o trabalho."),
        _seg(191, 193, "F", "Amém."),
    ]
    voz_disc = _voz(11)

    def embed(a, b):
        if a < 100:
            return voz_disc
        if 160 <= a < 200:
            return voz_oracao
        return _voz(20 + int(a))
    return segs, embed


def test_oracao_final_nao_conta_como_comentario_e_fica_com_o_nome():
    segs, embed = _reuniao_com_oracao(_voz(99))          # voz diferente do discurso
    out, pessoas, stats = turnos.reatribui(segs, embed, irmaos=EXEMPLO)
    assert stats["oracao"]["nome"] == "Ademir"
    assert stats["oracao"]["ligada_ao_orador"] is False
    oracao = [p for p in pessoas if p.nome == "Ademir"]
    assert len(oracao) == 1 and oracao[0].papel == "oração"
    assert oracao[0].comentarios == 0
    # a oração não entra como comentarista (nenhum turno com esse rótulo)
    assert all(p.papel != "comentarista" or p.nome != "Ademir" for p in pessoas)
    # os blocos da oração têm um só rótulo
    rot = {s.speaker for s in out if s.start >= 160 and s.end <= 191}
    assert len(rot) == 1


def test_oracao_com_voz_do_discurso_liga_ao_orador():
    segs, embed = _reuniao_com_oracao(_voz(11))           # mesma voz do discurso
    out, pessoas, stats = turnos.reatribui(segs, embed, irmaos=EXEMPLO)
    assert stats["oracao"]["ligada_ao_orador"] is True
    orador = [p for p in pessoas if p.papel == "orador"]
    assert len(orador) == 1 and orador[0].rotulo == "G" and orador[0].nome == "Ademir"
    assert all(s.speaker == "G" for s in out if 160 <= s.start < 191)


def test_co_dirigente_nao_vira_comentarista():
    segs = [_seg(0, 3, "F", "Vamos começar o estudo.")]
    t = 4.0
    nomes_fake = ["Ana", "Bia", "Caio", "Dora", "Edu", "Fábio", "Gil", "Hugo", "Ivo",
                  "Jaci", "Kátia", "Lia"]
    for k in range(12):                          # F e E alternam: 12 eventos cada
        falante = "F" if k % 2 == 0 else "E"
        nome = nomes_fake[k]
        segs.append(_seg(t, t + 2, falante, f"Irmã {nome}, por favor."))
        segs.append(_seg(t + 3, t + 10, "A", f"Comentário número {k} da reunião."))
        segs.append(_seg(t + 11, t + 13, falante, f"Obrigado, irmã {nome}."))
        t += 15
    _, pessoas, _ = turnos.reatribui(segs, lambda a, b: _voz(1))
    papeis = {p.rotulo: p.papel for p in pessoas}
    assert "E" not in [p.rotulo for p in pessoas if p.papel == "comentarista"]
    assert papeis.get("E") == "dirigente" and papeis.get("F") == "dirigente"


# --- grafias cadastradas trocadas no texto da transcrição (2026-10-12) ---
def test_corrige_texto_troca_so_grafia_cadastrada():
    from app.irmaos import Entrada, corrige_texto
    lista = [Entrada("1", "Ademir", grafias=("Modemir", "Módemir", "Admir")),
             Entrada("2", "Marines", grafias=("Marilês",))]
    txt = "Irmão Orador, Módemir, Paulino. Modemir e MÓDEMIR. Irmã Marilês. Modemiro e Ademar."
    assert corrige_texto(txt, lista) == (
        "Irmão Orador, Ademir, Paulino. Ademir e Ademir. Irmã Marines. Modemiro e Ademar.")


def test_corrige_texto_ignora_minuscula_e_conflito():
    from app.irmaos import Entrada, corrige_texto
    lista = [Entrada("1", "Edna", grafias=("Helio",)), Entrada("2", "Rita", grafias=("Helio",)),
             Entrada("3", "Nadir", grafias=("Nadeira",))]
    assert corrige_texto("Helio e Nadeira, nadeira.", lista) == "Helio e Nadir, nadeira."
    assert corrige_texto("sem lista", []) == "sem lista"


def test_transcricao_sai_com_grafia_corrigida_e_cru_intacto(tmp_path, monkeypatch):
    import json
    from app import config, irmaos, jobs, review
    from app.engines.base import Segment
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    config.JOBS_DIR.mkdir(parents=True, exist_ok=True)
    irmaos.criar("Ademir", "orador", ["Módemir"])
    job = jobs.new_job(filename="r.wav", engine="local", language="pt",
                       min_speakers=None, max_speakers=None, want_summary=False)
    review.save_raw(job, [Segment(0, 3, "A", "Bem-vindo, irmão Módemir Paulino.")])
    review.render_outputs(job.id)
    assert "irmão Ademir Paulino" in job.path("transcricao.txt").read_text("utf-8")
    cru = json.loads(job.path("segmentos.json").read_text("utf-8"))
    assert cru[0]["text"] == "Bem-vindo, irmão Módemir Paulino."
