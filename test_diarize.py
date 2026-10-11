"""Testes da diarização local (sem modelos: embeddings sintéticos).

Rodar:  python -m pytest test_diarize.py -v
"""
import numpy as np

from app.engines import diarize
from app.engines.diarize import Word, build_sentences, cluster


def _vozes(n_por_voz: list[int], dim: int = 64, ruido: float = 0.35, seed: int = 1):
    rng = np.random.default_rng(seed)
    centros = rng.normal(size=(len(n_por_voz), dim))
    vecs, labels = [], []
    for i, n in enumerate(n_por_voz):
        v = centros[i] + ruido * rng.normal(size=(n, dim))
        vecs.append(v)
        labels += [i] * n
    emb = np.concatenate(vecs)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return emb, np.array(labels)


def _acerto(pred: np.ndarray, real: np.ndarray) -> float:
    """Pureza: cada grupo previsto conta pelo rótulo real majoritário."""
    total = 0
    for c in set(pred):
        _, counts = np.unique(real[pred == c], return_counts=True)
        total += counts.max()
    return total / len(real)


def test_frases_cortam_em_pontuacao_e_pausa():
    w = [Word(0.0, 0.4, " Oi"), Word(0.5, 0.9, " tudo"), Word(1.0, 1.3, " bem?"),
         Word(1.4, 1.8, " Tudo"), Word(3.4, 3.8, " sim")]   # pausa de 1,6 s
    frases = build_sentences(w)
    assert [f.text for f in frases] == ["Oi tudo bem?", "Tudo", "sim"]


def test_fragmento_sem_ponto_segue_com_a_proxima_palavra():
    # "A gente tem que" (sem ponto) + pausa curta de 1,2 s + "ser mais um arrasto."
    w = [Word(0.0, 0.4, " A"), Word(0.5, 0.9, " gente"), Word(1.0, 1.3, " tem"),
         Word(1.4, 1.8, " que"), Word(3.0, 3.4, " ser"), Word(3.5, 3.9, " mais"),
         Word(4.0, 4.3, " um"), Word(4.4, 4.9, " arrasto.")]
    assert [f.text for f in build_sentences(w)] == ["A gente tem que ser mais um arrasto."]


def test_pausa_curta_depois_de_ponto_ainda_corta():
    w = [Word(0.0, 0.4, " Sim."), Word(1.3, 1.7, " Tá")]   # 0,9 s depois do ponto
    assert [f.text for f in build_sentences(w)] == ["Sim.", "Tá"]


def test_frase_longa_sem_pontuacao_e_quebrada():
    w = [Word(i * 0.5, i * 0.5 + 0.4, " palavra") for i in range(60)]  # 30 s
    assert all(f.duration <= diarize.MAX_SENTENCE_SECONDS for f in build_sentences(w))


def test_estima_dois_falantes():
    emb, real = _vozes([40, 25])
    pred = cluster(emb)
    assert len(set(pred)) == 2
    assert _acerto(pred, real) > 0.95


def test_estima_quatro_falantes_desbalanceados():
    emb, real = _vozes([50, 30, 12, 8])
    pred = cluster(emb)
    assert len(set(pred)) == 4
    assert _acerto(pred, real) > 0.9


def test_uma_voz_so():
    emb, _ = _vozes([30])
    assert len(set(cluster(emb))) == 1


def test_respeita_numero_exato_e_maximo():
    emb, _ = _vozes([20, 20, 20])
    assert len(set(cluster(emb, min_speakers=2, max_speakers=2))) == 2
    assert len(set(cluster(emb, max_speakers=2))) <= 2
    assert len(set(cluster(emb, min_speakers=3, max_speakers=3))) == 3


def test_rotulos_estilo_assemblyai():
    assert [diarize.speaker_label(i) for i in (0, 1, 25, 26)] == ["A", "B", "Z", "S27"]


def test_prompt_pontuado_com_participantes():
    from app.engines.local_engine import _hotwords

    assert _hotwords(["Ana", "Bia", "Caio"], "pt") == "Reunião com Ana, Bia e Caio."
    assert _hotwords([], "pt") == "Reunião de trabalho."
    assert _hotwords(["Ann"], "en") == "Ann"


# --------------------------------------------------------------------------
# Pós-processamento: fusão de vozes duplicadas, absorção, teto e limites
# --------------------------------------------------------------------------
def test_funde_clusters_que_sao_a_mesma_voz():
    # um orador separado em 3 grupos (eco/PA): centróides quase iguais
    emb, real = _vozes([30, 30, 30], ruido=0.02)
    base = np.array(emb[0])
    emb = np.stack([base + 0.01 * np.random.default_rng(i).normal(size=base.size)
                    for i in range(len(emb))])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    labels = diarize.refine(np.array([0] * 30 + [1] * 30 + [2] * 30), emb,
                            np.full(90, 5.0), lo=1, absorb_secs=0)
    assert len(set(labels)) == 1


def test_nao_funde_vozes_diferentes():
    emb, real = _vozes([30, 30], ruido=0.3)
    labels = diarize.refine(real, emb, np.full(60, 5.0), lo=1, absorb_secs=0)
    assert len(set(labels)) == 2


def test_absorve_cluster_com_pouco_tempo_no_vizinho_mais_parecido():
    emb, real = _vozes([40, 40, 3], ruido=0.3)
    secs = np.array([5.0] * 80 + [0.5] * 3)          # o pequeno fala 1,5 s
    labels = diarize.refine(real, emb, secs, lo=1, absorb_secs=45.0)
    assert len(set(labels)) == 2


def test_refine_nao_passa_do_minimo():
    emb, real = _vozes([30, 30, 30], ruido=0.3)
    labels = diarize.refine(real, emb, np.full(90, 0.2), lo=3, absorb_secs=45.0)
    assert len(set(labels)) == 3


def test_teto_automatico_limita_a_doze():
    emb, _ = _vozes([10] * 15, dim=96, ruido=0.2)
    assert len(set(cluster(emb))) <= diarize.MAX_SPEAKERS_AUTO


def test_frase_curta_entre_dois_do_mesmo_falante_herda():
    frases = [diarize.Sentence([Word(0, 3, " a")], speaker=0),
              diarize.Sentence([Word(3.2, 3.5, " é")]),          # 0,3 s, sem embedding
              diarize.Sentence([Word(3.7, 8, " b")], speaker=0)]
    diarize._smooth_short(frases)
    assert frases[1].speaker == 0


def test_limites_sem_teto_viram_dica_e_nao_forcam():
    from app.engines.local_engine import _limites

    assert _limites(20, 20) == (None, None)     # "20 e 20" digitado na tela
    assert _limites(2, 2) == (2, 2)
    assert _limites(3, None) == (3, None)
    assert _limites(5, 3) == (None, None)       # limites contraditórios


def test_laco_de_cantico_tira_a_terceira_repeticao():
    from app.engines.local_engine import _DetectorDeLaco

    d = _DetectorDeLaco()
    frase = "A gente tem que ser mais um arrasto."
    saida = [d.repetido(frase, t) for t in (5.0, 10.0, 15.0, 20.0)]
    assert saida == [False, False, True, True]
    # fala normal com frase curta repetida ("Amém.") nunca é laço
    assert [d.repetido("Amém.", t) for t in (30.0, 31.0, 32.0)] == [False, False, False]
    # depois de 60 s a frase volta a valer
    assert d.repetido(frase, 200.0) is False
