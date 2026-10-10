"""Diarização local (quem falou o quê) sem GPU, sem PyTorch e sem internet.

Segue a mesma receita dos serviços pagos (AssemblyAI e afins):

  1. Transcrição com tempo por palavra (feita antes, pelo Whisper).
  2. As palavras viram "frases": cortadas em pontuação final ou pausa.
     Troca de falante quase sempre cai numa dessas fronteiras.
  3. Cada frase longa o bastante vira um *embedding* de voz (vetor que
     descreve o timbre), com um modelo ONNX pequeno via sherpa-onnx.
  4. Agrupamento espectral dos embeddings. O número de falantes é estimado
     pelo maior "salto" de autovalores (eigengap), sempre dentro do intervalo
     mínimo/máximo informado — o equivalente ao `speakers_expected` /
     `speaker_options` do AssemblyAI. Padrão: 1 a 10 falantes, como lá.
  5. Frases curtas demais para um embedding confiável ("Sim.", "Tá.") herdam
     o falante mais parecido; empate apertado fica com o falante anterior.

Tudo aqui é numpy puro (exceto a extração do embedding), então é rápido e
fácil de empacotar no executável.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

SAMPLE_RATE = 16000

# Frases mais curtas que isso não geram embedding próprio (pouco timbre).
MIN_EMBED_SECONDS = 1.0
# Frases longas são cortadas em janelas para o embedding (média das janelas).
EMBED_WINDOW_SECONDS = 6.0
# Pausa entre palavras que encerra uma frase mesmo sem pontuação.
SENTENCE_GAP_SECONDS = 0.8
# Frase muito longa sem pontuação é quebrada para não esconder trocas.
MAX_SENTENCE_SECONDS = 15.0
_SENTENCE_END = (".", "?", "!", "…")


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Sentence:
    words: list[Word] = field(default_factory=list)
    speaker: int = -1

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def text(self) -> str:
        return "".join(w.text for w in self.words).strip()


# --------------------------------------------------------------------------
# 2) Palavras -> frases
# --------------------------------------------------------------------------
def build_sentences(words: Sequence[Word]) -> list[Sentence]:
    sentences: list[Sentence] = []
    cur = Sentence()
    for w in words:
        if cur.words:
            gap = w.start - cur.words[-1].end
            too_long = w.end - cur.start > MAX_SENTENCE_SECONDS
            if gap >= SENTENCE_GAP_SECONDS or too_long:
                sentences.append(cur)
                cur = Sentence()
        cur.words.append(w)
        if w.text.strip().endswith(_SENTENCE_END):
            sentences.append(cur)
            cur = Sentence()
    if cur.words:
        sentences.append(cur)
    return sentences


# --------------------------------------------------------------------------
# 3) Embeddings de voz
# --------------------------------------------------------------------------
class EmbeddingExtractor:
    def __init__(self, model_path: Path, num_threads: int = 4) -> None:
        import sherpa_onnx  # import tardio: só carrega quando for diarizar

        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(model_path), num_threads=num_threads)
        if not cfg.validate():
            raise RuntimeError(f"Modelo de voz inválido ou ausente: {model_path}")
        self._ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)

    def embed(self, samples: np.ndarray) -> np.ndarray:
        stream = self._ex.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples)
        stream.input_finished()
        vec = np.asarray(self._ex.compute(stream), dtype=np.float32)
        return vec / (np.linalg.norm(vec) + 1e-9)


def _embed_span(ex: EmbeddingExtractor, audio: np.ndarray,
                start: float, end: float) -> np.ndarray:
    win = int(EMBED_WINDOW_SECONDS * SAMPLE_RATE)
    a, b = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
    chunk = audio[a:b]
    parts = [chunk[i:i + win] for i in range(0, len(chunk), win)]
    # descarta sobra final curta demais (só se houver outras janelas)
    parts = [p for p in parts if len(p) >= SAMPLE_RATE * MIN_EMBED_SECONDS] or parts
    vecs = np.stack([ex.embed(p) for p in parts])
    mean = vecs.mean(axis=0)
    return mean / (np.linalg.norm(mean) + 1e-9)


# --------------------------------------------------------------------------
# 4) Agrupamento espectral com estimativa do número de falantes
# --------------------------------------------------------------------------
def _affinity(emb: np.ndarray) -> np.ndarray:
    sim = np.clip(emb @ emb.T, -1.0, 1.0)
    n = len(sim)
    # Poda: cada frase só "enxerga" os vizinhos mais parecidos (NME-SC).
    # Remove o ruído de similaridades médias que borram os grupos.
    keep = max(2, min(n - 1, int(round(n * 0.15))))
    aff = np.zeros_like(sim)
    idx = np.argsort(-sim, axis=1)[:, :keep + 1]
    rows = np.arange(n)[:, None]
    aff[rows, idx] = sim[rows, idx]
    aff = np.maximum(aff, aff.T)
    np.fill_diagonal(aff, 0.0)
    return np.clip(aff, 0.0, None)


def _kmeans(x: np.ndarray, k: int, seed: int = 0, n_init: int = 10,
            iters: int = 100) -> np.ndarray:
    rng = np.random.default_rng(seed)
    best_labels, best_inertia = None, np.inf
    for _ in range(n_init):
        # k-means++
        centers = [x[rng.integers(len(x))]]
        for _ in range(1, k):
            d2 = np.min(((x[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
            p = d2 / d2.sum() if d2.sum() > 0 else None
            centers.append(x[rng.choice(len(x), p=p)])
        c = np.array(centers)
        for _ in range(iters):
            labels = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
            new_c = np.array([x[labels == j].mean(0) if np.any(labels == j) else c[j]
                              for j in range(k)])
            if np.allclose(new_c, c):
                break
            c = new_c
        inertia = ((x - c[labels]) ** 2).sum()
        if inertia < best_inertia:
            best_labels, best_inertia = labels, inertia
    return best_labels


def cluster(emb: np.ndarray, min_speakers: Optional[int] = None,
            max_speakers: Optional[int] = None) -> np.ndarray:
    """Rótulo (0..k-1) para cada embedding. k estimado por eigengap."""
    n = len(emb)
    lo = max(1, min_speakers or 1)
    hi = max(lo, max_speakers or 10)
    hi = min(hi, n)
    lo = min(lo, hi)
    if n <= 1 or hi == 1:
        return np.zeros(n, dtype=int)

    aff = _affinity(emb)
    deg = aff.sum(1)
    d = 1.0 / np.sqrt(deg + 1e-9)
    lap = np.eye(n) - d[:, None] * aff * d[None, :]
    evals, evecs = np.linalg.eigh(lap)

    if lo == hi:
        k = lo
    else:
        # k = posição do maior salto entre autovalores consecutivos
        gaps = np.diff(evals[: hi + 1])
        k = int(np.argmax(gaps[lo - 1: hi])) + lo
        # Salto pequeno = estrutura fraca: provavelmente uma voz só
        if lo == 1 and gaps.max() < 0.05:
            k = 1
    if k == 1:
        return np.zeros(n, dtype=int)

    u = evecs[:, :k]
    u = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-9)
    return _kmeans(u, k)


# --------------------------------------------------------------------------
# 1ª rodada: quantas vozes existem? (só pelo som, antes de transcrever)
# --------------------------------------------------------------------------
SCAN_WINDOW_SECONDS = 3.0


@dataclass
class VoiceScan:
    count: int                              # nº de vozes encontradas
    centroids: dict[int, np.ndarray]        # assinatura de cada voz
    seconds: dict[int, float]               # quanto cada voz falou


def scan_voices(audio: np.ndarray, extractor: EmbeddingExtractor,
                min_speakers: Optional[int] = None,
                max_speakers: Optional[int] = None) -> VoiceScan:
    """Detecta a fala (VAD), corta em janelas de 3 s e agrupa os timbres.

    Medido numa reunião real de 22 min: 17 s de processamento, 2 vozes
    encontradas; a mesma voz comparada entre as duas metades da reunião deu
    similaridade 0,97–0,98, e vozes diferentes, 0,21.
    """
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    spans = get_speech_timestamps(
        audio, VadOptions(min_silence_duration_ms=300, speech_pad_ms=100))
    step = int(SCAN_WINDOW_SECONDS * SAMPLE_RATE)
    windows = []
    for sp in spans:
        for a in range(sp["start"], sp["end"], step):
            b = min(a + step, sp["end"])
            if b - a >= int(1.5 * SAMPLE_RATE):
                windows.append((a, b))
    if not windows:
        return VoiceScan(0, {}, {})

    emb = np.stack([extractor.embed(audio[a:b]) for a, b in windows])
    labels = cluster(emb, min_speakers, max_speakers)
    k = int(labels.max()) + 1
    centroids, seconds = {}, {}
    for j in range(k):
        v = emb[labels == j].mean(0)
        centroids[j] = v / (np.linalg.norm(v) + 1e-9)
        seconds[j] = float(sum(b - a for (a, b), lab in zip(windows, labels)
                               if lab == j)) / SAMPLE_RATE
    return VoiceScan(k, centroids, seconds)


# --------------------------------------------------------------------------
# Pipeline completo
# --------------------------------------------------------------------------
def diarize(audio: np.ndarray, words: Sequence[Word], extractor: EmbeddingExtractor,
            min_speakers: Optional[int] = None,
            max_speakers: Optional[int] = None,
            ) -> tuple[list[Sentence], dict[int, np.ndarray]]:
    """Frases com falante (0, 1, ... na ordem de aparição) + a assinatura
    de voz (centróide normalizado) de cada falante."""
    sentences = build_sentences(words)
    if not sentences:
        return [], {}

    long_idx = [i for i, s in enumerate(sentences) if s.duration >= MIN_EMBED_SECONDS]
    if len(long_idx) < 2:
        for s in sentences:
            s.speaker = 0
        if long_idx:
            i = long_idx[0]
            vec = _embed_span(extractor, audio, sentences[i].start, sentences[i].end)
            return sentences, {0: vec}
        return sentences, {}

    emb = np.stack([_embed_span(extractor, audio, sentences[i].start, sentences[i].end)
                    for i in long_idx])
    labels = cluster(emb, min_speakers, max_speakers)
    for i, lab in zip(long_idx, labels):
        sentences[i].speaker = int(lab)

    k = int(labels.max()) + 1
    centroids = np.stack([emb[labels == j].mean(0) for j in range(k)])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9

    # 5) frases curtas: falante mais parecido; empate -> mantém o anterior
    prev = int(labels[0])
    for s in sentences:
        if s.speaker >= 0:
            prev = s.speaker
            continue
        if k == 1:
            s.speaker = 0
            continue
        try:
            vec = _embed_span(extractor, audio, s.start, max(s.end, s.start + 0.5))
            sims = centroids @ vec
            order = np.argsort(-sims)
            s.speaker = int(order[0]) if sims[order[0]] - sims[order[1]] > 0.05 else prev
        except Exception:  # noqa: BLE001 — trecho curto demais p/ o modelo
            s.speaker = prev
        prev = s.speaker

    # Rótulos na ordem em que cada voz aparece: A, B, C... (igual ao AssemblyAI)
    order: dict[int, int] = {}
    for s in sentences:
        order.setdefault(s.speaker, len(order))
    for s in sentences:
        s.speaker = order[s.speaker]
    return sentences, {order[j]: centroids[j] for j in range(k) if j in order}


def speaker_label(idx: int) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return letters[idx] if idx < len(letters) else f"S{idx + 1}"
