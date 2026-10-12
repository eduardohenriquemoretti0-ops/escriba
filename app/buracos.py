"""Retranscrição dos trechos que o Whisper pulou ("buracos").

Problema medido no salão (10/10): em 1:19:32, 1:23:07 e 1:24:56 o VAD acha de
17 a 27 s de fala, o áudio está baixo (-31 a -36 dBFS, contra cerca de -24 no
resto) e o Whisper não devolve nenhuma palavra. São 220 palavras do gabarito
sem transcrição.

Receita (protótipo F4 "c", medida no laboratório do Escriba v5):

1. buraco = intervalo de pelo menos ``GAP_MIN`` s entre palavras do Whisper,
   fora dos trechos de coro;
2. VAD Silero com limiar mais baixo (``VAD_LIMIAR``) dentro do buraco: precisa
   de pelo menos ``FALA_MIN`` s de fala;
3. ganho: o trecho é levado a ``RMS_ALVO`` dBFS na fala (teto ``GANHO_MAX``
   dB), sem estourar (pico <= 0,99);
4. Whisper só no trecho, com os MESMOS parâmetros da transcrição principal
   (hotwords em frase pontuada, ``condition_on_previous_text=False``, limiares
   anti-alucinação), mais o limiar de VAD baixo;
5. filtros: lista de alucinações do produto, ``avg_logprob >= LOGPROB_MIN``,
   ``no_speech_prob < NO_SPEECH_MAX`` e palavras dentro do buraco.

Custo medido (CPU extra sobre a transcrição inteira): +10,6% no salão, +28,5%
na conversa de 2 pessoas, +36% a +113% no AMI. Por isso há um teto de CPU
configurável (``config.BURACOS_TETO_CPU``, CPU-segundos por segundo de áudio):
os buracos com mais fala vão primeiro, e o que passar do teto fica de fora,
com registro no log e no progresso.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

from .engines.diarize import SAMPLE_RATE, Word

log = logging.getLogger(__name__)

GAP_MIN = 2.0  # s sem palavra para ser candidato a buraco
VAD_LIMIAR = 0.35  # limiar do VAD Silero no buraco (o padrão é 0,5)
FALA_MIN = 1.0  # s de fala (VAD) dentro do buraco
RMS_ALVO = -23.0  # dBFS da fala depois do ganho
GANHO_MAX = 18.0  # dB
LOGPROB_MIN = -1.0  # segmento com avg_logprob abaixo disso é recusado
NO_SPEECH_MAX = 0.6  # segmento com no_speech_prob a partir disso é recusado
_MARGEM = 0.05  # s de folga nas bordas do buraco

Intervalo = tuple[float, float]
Transcritor = Callable[[np.ndarray], Iterable[Any]]


@dataclass
class Buraco:
    """Trecho sem palavra transcrita, com fala no VAD."""

    inicio: float
    fim: float
    fala_s: float
    falas: list[Intervalo] = field(default_factory=list)


@dataclass
class Relatorio:
    """Resumo da retranscrição (vai para o log e para o resumo do job)."""

    buracos: int = 0
    retranscritos: int = 0
    pulados_teto: int = 0
    palavras: int = 0
    cpu_s: float = 0.0
    teto_cpu_s: float = 0.0
    detalhes: list[dict] = field(default_factory=list)

    def como_dict(self) -> dict:
        """Versão serializável (sem os detalhes de cada buraco)."""
        return {
            "buracos": self.buracos,
            "retranscritos": self.retranscritos,
            "pulados_teto": self.pulados_teto,
            "palavras": self.palavras,
            "cpu_s": round(self.cpu_s, 1),
            "teto_cpu_s": round(self.teto_cpu_s, 1),
        }


def dbfs(x: np.ndarray) -> float:
    """Nível RMS do trecho em dBFS (sinal float em -1..1)."""
    if x.size == 0:
        return -200.0
    return float(20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-9))


def falas_vad(audio: np.ndarray) -> list[Intervalo]:
    """Trechos de fala (s) pelo VAD Silero do faster-whisper, com limiar baixo."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    opcoes = VadOptions(
        threshold=VAD_LIMIAR, min_silence_duration_ms=500, speech_pad_ms=100
    )
    return [
        (s["start"] / SAMPLE_RATE, s["end"] / SAMPLE_RATE)
        for s in get_speech_timestamps(audio, opcoes)
    ]


def acha_buracos(
    duracao: float,
    palavras: Sequence[Intervalo],
    coro: Sequence[Intervalo],
    falas: Sequence[Intervalo],
) -> list[Buraco]:
    """Intervalos sem palavra, fora do coro, com fala suficiente no VAD.

    Args:
        duracao: duração do áudio (s).
        palavras: (início, fim) de cada palavra transcrita.
        coro: trechos de cântico (não são buraco).
        falas: trechos de fala do VAD (limiar baixo).

    Returns:
        Buracos em ordem de tempo.
    """
    out: list[Buraco] = []
    prev = 0.0
    for a, b in sorted(palavras) + [(duracao, duracao)]:
        if a - prev >= GAP_MIN:
            lo, hi = prev + _MARGEM, a - _MARGEM
            if not any(min(hi, y) - max(lo, x) > 0 for x, y in coro):
                dentro = [
                    (max(lo, s), min(hi, e))
                    for s, e in falas
                    if min(hi, e) > max(lo, s)
                ]
                total = sum(e - s for s, e in dentro)
                if total >= FALA_MIN:
                    out.append(
                        Buraco(round(lo, 3), round(hi, 3), round(total, 2), dentro)
                    )
        prev = max(prev, b)
    return out


def aplica_ganho(
    trecho: np.ndarray, falas: Sequence[Intervalo], inicio: float
) -> tuple[np.ndarray, float]:
    """Leva a fala do trecho a RMS_ALVO dBFS, sem passar de GANHO_MAX nem estourar.

    Args:
        trecho: amostras do buraco.
        falas: trechos de fala (tempo absoluto) dentro do buraco.
        inicio: tempo absoluto da 1ª amostra do trecho.

    Returns:
        (amostras com ganho, ganho aplicado em dB).
    """
    partes = [
        trecho[int((s - inicio) * SAMPLE_RATE) : int((e - inicio) * SAMPLE_RATE)]
        for s, e in falas
    ]
    partes = [p for p in partes if p.size]
    voz = np.concatenate(partes) if partes else trecho
    ganho_db = min(GANHO_MAX, max(0.0, RMS_ALVO - dbfs(voz)))
    y = trecho * (10 ** (ganho_db / 20))
    pico = float(np.max(np.abs(y))) if y.size else 0.0
    if pico > 0.99:
        y = y * (0.99 / pico)
        ganho_db -= 20 * np.log10(pico / 0.99)
    return y.astype(np.float32), round(float(ganho_db), 1)


def _motivo_recusa(
    seg: Any, palavras: list[Word], e_alucinacao: Callable[[str], bool]
) -> Optional[str]:
    texto = (seg.text or "").strip()
    if e_alucinacao(texto):
        return "alucinação conhecida"
    if seg.avg_logprob < LOGPROB_MIN:
        return f"avg_logprob {seg.avg_logprob:.2f}"
    if seg.no_speech_prob >= NO_SPEECH_MAX:
        return f"no_speech {seg.no_speech_prob:.2f}"
    if not palavras:
        return "sem palavras"
    return None


def retranscreve(
    audio: np.ndarray,
    palavras: Sequence[Intervalo],
    coro: Sequence[Intervalo],
    transcrever: Transcritor,
    *,
    teto_cpu_s: float,
    e_alucinacao: Callable[[str], bool],
    progress: Callable[[str], None] = lambda _m: None,
    falas: Optional[Sequence[Intervalo]] = None,
) -> tuple[list[list[Word]], Relatorio]:
    """Acha e retranscreve os buracos, dentro do teto de CPU.

    Args:
        audio: áudio 16 kHz mono (float32).
        palavras: (início, fim) das palavras já transcritas.
        coro: trechos de cântico.
        transcrever: função que recebe as amostras do trecho e devolve os
            segmentos do Whisper (com ``start``, ``end``, ``text``,
            ``avg_logprob``, ``no_speech_prob`` e ``words``), em tempo relativo.
        teto_cpu_s: CPU máxima (s) gasta aqui. Zero ou negativo desliga.
        e_alucinacao: filtro de frases inventadas do motor.
        progress: recebe a fase atual (aparece na tela).
        falas: trechos de fala do VAD; se None, roda o VAD aqui.

    Returns:
        (palavras novas agrupadas por segmento aceito, em ordem de tempo; relatório).
    """
    rel = Relatorio(teto_cpu_s=max(0.0, teto_cpu_s))
    if teto_cpu_s <= 0:
        log.info("Buracos: desligado (teto de CPU = 0)")
        return [], rel
    c0 = time.process_time()
    duracao = len(audio) / SAMPLE_RATE
    if falas is None:
        falas = falas_vad(audio)
    lista = acha_buracos(duracao, palavras, coro, falas)
    rel.buracos = len(lista)
    if not lista:
        rel.cpu_s = time.process_time() - c0
        return [], rel
    # mais fala primeiro: se o teto cortar, corta o que vale menos
    ordem = sorted(lista, key=lambda b: (-b.fala_s, b.inicio))
    novos: list[tuple[float, list[Word]]] = []
    for k, b in enumerate(ordem, 1):
        gasto = time.process_time() - c0
        if gasto >= teto_cpu_s:
            rel.pulados_teto = len(ordem) - k + 1
            progress(
                f"Fase 2/3 · Trechos pulados: teto de CPU atingido "
                f"({rel.pulados_teto} de {len(ordem)} ficaram de fora)"
            )
            log.warning(
                "Buracos: teto de CPU (%.0f s) atingido; %d de %d não " "retranscritos",
                teto_cpu_s,
                rel.pulados_teto,
                len(ordem),
            )
            break
        progress(f"Fase 2/3 · Recuperando trechos pulados — {k}/{len(ordem)}")
        trecho = audio[int(b.inicio * SAMPLE_RATE) : int(b.fim * SAMPLE_RATE)]
        y, ganho = aplica_ganho(trecho, b.falas, b.inicio)
        aceitos, recusados = 0, []
        for seg in transcrever(y):
            ws = [
                Word(w.start + b.inicio, w.end + b.inicio, w.word)
                for w in (seg.words or [])
                if w.start + b.inicio <= b.fim
            ]
            motivo = _motivo_recusa(seg, ws, e_alucinacao)
            if motivo:
                recusados.append(
                    {"texto": (seg.text or "").strip()[:120], "motivo": motivo}
                )
                continue
            novos.append((ws[0].start, ws))
            aceitos += len(ws)
        rel.retranscritos += 1
        rel.palavras += aceitos
        rel.detalhes.append(
            {
                "inicio": b.inicio,
                "fim": b.fim,
                "fala_s": b.fala_s,
                "dbfs": round(dbfs(trecho), 1),
                "ganho_db": ganho,
                "palavras": aceitos,
                "recusados": recusados,
            }
        )
        log.info(
            "Buraco %.1f–%.1f s: ganho %.1f dB, %d palavras, %d recusados",
            b.inicio,
            b.fim,
            ganho,
            aceitos,
            len(recusados),
        )
    rel.cpu_s = time.process_time() - c0
    novos.sort(key=lambda x: x[0])
    return [ws for _, ws in novos], rel
