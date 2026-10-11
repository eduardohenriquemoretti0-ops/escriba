"""Testes da separação de comentaristas por turno (estudo, sem modelos).

Rodar:  python -m pytest test_turnos.py -v
"""
import numpy as np

from app import turnos
from app.engines.base import Segment


def _voz(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=64)
    return v / np.linalg.norm(v)


def _embed(mapa):
    """Vetor por trecho: o primeiro intervalo de `mapa` que contém o início."""
    def f(a, b):
        for ini, fim, vec in mapa:
            if ini <= a < fim:
                return vec
        return _voz(999)
    return f


def _seg(ini, fim, falante, texto):
    return Segment(float(ini), float(fim), falante, texto)


def _reuniao_basica():
    """D é o dirigente. Fulano (A) responde e é agradecido; Beltrana (B) idem;
    um comentário sem nome (C) com a voz do Fulano, e um de voz nova (E)."""
    segs = [
        _seg(0, 3, "D", "Vamos começar o estudo da revista A Sentinela."),
        _seg(4, 6, "D", "Sim, irmão Fulano, por favor."),
        _seg(7, 20, "A", "Olá a todos, eu acho que a gente deve orar."),
        _seg(21, 23, "D", "Obrigado, irmão Fulano. Irmã Beltrana, por favor."),
        _seg(24, 32, "B", "Eu acho que Jeová se importa muito conosco."),
        _seg(33, 35, "D", "Muito bem, irmã Beltrana."),
        _seg(36, 40, "C", "Concordo, a oração ajuda muito."),
        _seg(41, 43, "D", "Obrigado, irmão Fulano."),
        _seg(44, 55, "E", "Eu lembro de uma experiência com a oração."),
        _seg(56, 58, "D", "Vamos seguir adiante com o próximo parágrafo."),
    ]
    vozes = [
        (7, 20, _voz(1)),          # Fulano
        (24, 32, _voz(2)),         # Beltrana (voz bem diferente)
        (36, 40, _voz(1) * 0.98 + 0.02 * _voz(50)),   # Fulano de novo, mesmo timbre
        (44, 55, _voz(3)),         # alguém novo
    ]
    return segs, _embed(vozes)


def test_estrutura_so_ativa_com_dirigente_e_eventos():
    segs, _ = _reuniao_basica()
    est = turnos.estrutura(segs)
    assert est is not None and est.dirigente == "D"
    assert est.eventos >= turnos.MIN_EVENTOS


def test_reuniao_de_duas_pessoas_nao_ativa():
    segs = [_seg(0, 5, "A", "Bom dia a todos."), _seg(6, 12, "B", "Bom dia, tudo bem?")]
    out, pessoas, stats = turnos.reatribui(segs, lambda a, b: _voz(1))
    assert out == segs and pessoas == [] and stats["ativo"] is False


def test_nome_por_chamada_e_agradecimento_e_mesma_pessoa_por_nome():
    segs, emb = _reuniao_basica()
    out, pessoas, stats = turnos.reatribui(segs, emb)
    nomes = [p.nome for p in pessoas]
    assert "Fulano" in nomes and "Beltrana" in nomes
    # o turno de Fulano (A) e o de C (voz parecida, sem nome) são a mesma pessoa
    assert out[2].speaker == out[6].speaker
    assert out[2].speaker != out[4].speaker          # Beltrana é outra pessoa


def test_turno_sem_nome_e_voz_nova_vira_comentarista():
    segs, emb = _reuniao_basica()
    _, pessoas, _ = turnos.reatribui(segs, emb)
    comentaristas = [p for p in pessoas if p.papel == "comentarista" and not p.nomeada]
    assert [p.nome for p in comentaristas] == ["Comentarista 1"]


def test_chamada_e_agradecimento_discordando_a_chamada_vence():
    segs = [
        _seg(0, 3, "D", "Vamos começar o estudo da revista A Sentinela."),
        _seg(4, 6, "D", "Sim, irmão Fulano, por favor."),
        _seg(7, 20, "A", "Olá a todos, eu acho que a gente deve orar."),
        _seg(21, 23, "D", "Obrigado, irmão Ciclano. Irmã Beltrana, por favor."),
        _seg(24, 32, "B", "Eu acho que Jeová se importa muito conosco."),
        _seg(33, 35, "D", "Muito bem, irmã Beltrana."),
    ]
    vozes = [(7, 20, _voz(1)), (24, 32, _voz(2))]
    _, pessoas, stats = turnos.reatribui(segs, _embed(vozes))
    assert stats["discordancias"] == 1
    nomeados = [p for p in pessoas if p.nomeada]
    assert "Fulano" in [p.nome for p in nomeados]          # a chamada venceu
    assert "Ciclano" not in [p.nome for p in nomeados]     # o agradecimento perdeu
    assert "agradecimento disse Ciclano" in [p.origem for p in nomeados if p.nome == "Fulano"][0] \
        or any("agradecimento disse Ciclano" in t for t in [p.origem for p in nomeados])


def test_interjeicao_curta_fica_no_turno_anterior_e_recebe_o_nome():
    # C fala 10 s; A interrompe com 2 s ("Isso.") e o dirigente agradece C
    segs = [
        _seg(0, 3, "D", "Vamos começar o estudo da revista A Sentinela."),
        _seg(4, 6, "D", "Sim, irmão Fulano, por favor."),
        _seg(7, 17, "C", "Olha, na minha opinião a oração é o caminho."),
        _seg(17.5, 19.5, "A", "Isso."),
        _seg(20, 23, "C", "E também a leitura da Bíblia."),
        _seg(24, 26, "D", "Obrigado, irmão Fulano. Irmã Beltrana, por favor."),
        _seg(27, 35, "B", "Eu concordo com isso."),
        _seg(36, 38, "D", "Muito bem, irmã Beltrana."),
    ]
    vozes = [(7, 23, _voz(1)), (27, 35, _voz(2))]
    out, pessoas, _ = turnos.reatribui(segs, _embed(vozes))
    assert out[2].speaker == out[3].speaker == out[4].speaker   # interjeição junta
    assert "Fulano" in [p.nome for p in pessoas if p.nomeada]


def test_leitor_e_dirigente_nao_viram_turno():
    segs = [
        _seg(0, 3, "D", "Vamos começar o estudo da revista A Sentinela."),
        _seg(4, 6, "D", "Sim, irmão Fulano, por favor. Irmã Beltrana, por favor."),
        _seg(7, 12, "A", "Olá, eu acho que é isso mesmo."),
        _seg(13, 15, "D", "Obrigado, irmão Fulano. Vamos iniciar lendo o parágrafo 1."),
        _seg(16, 40, "B", "Em uma guerra, os soldados precisam confiar em seus comandantes."),
        _seg(41, 43, "D", "Obrigado, irmã Beltrana."),
    ]
    vozes = [(7, 12, _voz(1)), (16, 40, _voz(2))]
    est = turnos.estrutura(segs)
    assert est.leitor == "B"
    out, pessoas, _ = turnos.reatribui(segs, _embed(vozes))
    assert out[4].speaker == "B"                    # leitura continua sendo do leitor
    assert all(p.rotulo != "B" or p.papel == "leitor" for p in pessoas)


def test_nome_do_dirigente_do_anuncio_no_texto():
    segs, _ = _reuniao_basica()
    segs.insert(0, _seg(-30, -28, "D", "O irmão Dirigente, José Eduardo, vai conduzir."))
    est = turnos.estrutura(sorted(segs, key=lambda s: s.start))
    assert est.nome_dirigente == "José Eduardo"


def test_medir_separacao_separa_mesmo_e_diferente():
    segs, emb = _reuniao_basica()
    est = turnos.estrutura(segs)
    ts, idx = turnos._turnos(segs, est)
    turnos._nomeia(ts, segs, idx, est)
    for t in ts:
        t.voz = emb(t.inicio, t.fim)
    mesmo, diferente = turnos.medir_separacao(ts)
    assert mesmo and diferente
    assert min(mesmo) > max(diferente)


def test_caso_carmen_zoom_nao_vira_emacardi():
    """Gabarito do Eduardo: 50:42–51:23 é a irmã Carmen pelo Zoom. A chamada
    ("A irmã Carmen, pelo Zoom. Irmã Carmen, por favor.") vence o agradecimento
    que o Whisper escreveu como "Emacardi"."""
    segs = [
        _seg(0, 3, "D", "Vamos começar o estudo da revista A Sentinela."),
        _seg(4, 6, "D", "Sim, irmão Fulano, por favor. Irmã Beltrana, por favor."),
        _seg(7, 20, "A", "Olá a todos, eu acho que a gente deve orar."),
        _seg(21, 23, "D", "Obrigado, irmão Fulano. Irmã Beltrana, por favor."),
        _seg(24, 32, "B", "Eu acho que Jeová se importa muito conosco."),
        _seg(33, 35, "D", "Muito bem, irmã Beltrana. A irmã Carmen, pelo Zoom. "
                          "Irmã Carmen, por favor."),
        _seg(36, 60, "C", "Eu acho que a oração é importante na minha família."),
        _seg(61, 63, "D", "Muito obrigado, Emacardi."),
        _seg(64, 70, "A", "Vamos prosseguir com o próximo ponto."),
    ]
    vozes = [(7, 20, _voz(1)), (24, 32, _voz(2)), (36, 60, _voz(5))]
    out, pessoas, stats = turnos.reatribui(segs, _embed(vozes))
    nomes = [p.nome for p in pessoas]
    assert "Emacardi" not in nomes
    assert "Carmen" in nomes
    carmen = next(p for p in pessoas if p.nome == "Carmen")
    assert "Zoom" in carmen.origem
    assert out[6].speaker == carmen.rotulo


def test_zoom_nao_se_funde_com_quem_esta_no_salao():
    # comentário pelo Zoom (voz V5) não pode ser juntado a quem está no salão (V1)
    segs = [
        _seg(0, 3, "D", "Vamos começar o estudo da revista A Sentinela."),
        _seg(4, 6, "D", "Sim, irmão Fulano, por favor. Irmã Beltrana, por favor."),
        _seg(7, 20, "A", "Olá a todos, eu acho que a gente deve orar."),
        _seg(21, 23, "D", "Obrigado, irmão Fulano. Irmã Beltrana, por favor."),
        _seg(24, 32, "B", "Eu acho que Jeová se importa muito conosco."),
        _seg(33, 35, "D", "Muito bem, irmã Beltrana. Pelo Zoom, irmã Carmen, por favor."),
        _seg(36, 50, "C", "Pelo Zoom eu concordo com vocês todos."),
        _seg(51, 53, "D", "Vamos seguir."),
    ]
    vozes = [(7, 20, _voz(1)), (24, 32, _voz(2)), (36, 50, _voz(1))]   # mesma voz!
    _, pessoas, _ = turnos.reatribui(segs, _embed(vozes), limiar=0.5)
    carmen = next(p for p in pessoas if p.nome == "Carmen")
    assert carmen.nome != "Fulano"
    assert [p.nome for p in pessoas].count("Fulano") == 1
