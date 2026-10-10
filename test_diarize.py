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
         Word(1.4, 1.8, " Tudo"), Word(3.0, 3.4, " sim")]   # pausa de 1,2 s
    frases = build_sentences(w)
    assert [f.text for f in frases] == ["Oi tudo bem?", "Tudo", "sim"]


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
