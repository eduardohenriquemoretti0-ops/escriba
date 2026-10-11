"""Motor local (padrão): Faster-Whisper + diarização ONNX. 100% offline.

Configurado para chegar o mais perto possível do AssemblyAI (Universal +
speaker_labels) numa máquina sem GPU:

  AssemblyAI                         Aqui
  ---------------------------------  -----------------------------------------
  modelo Universal                   Whisper large-v3-turbo (int8 na CPU)
  punctuate / format_text            pontuação e maiúsculas nativas do Whisper
  keyterms_prompt / word_boost       `hotwords` do Whisper com os participantes
                                     e termos informados (reaplicado em toda
                                     janela de 30 s, não só na primeira)
  speaker_labels                     diarize.py (embedding de voz por frase +
                                     agrupamento espectral)
  speakers_expected / speaker_options mín./máx. de falantes (estimativa 1–12)
  utterances                         frases do mesmo falante unidas em blocos
  filtro de silêncio/ruído           VAD Silero + limiares anti-alucinação

Nada sai da máquina: sem internet a transcrição funciona igual, desde que
os modelos estejam em models/ (o instalador/build já deixa lá).
"""
from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Optional

from .. import config
from . import diarize
from .base import ProgressCB, Segment

_lock = threading.Lock()

# Silêncio (s) a partir do qual o mesmo falante começa um bloco novo.
UTTERANCE_MAX_GAP = 10.0
_whisper_cache: dict = {}
_extractor_cache: dict = {}

# Frases que o Whisper "inventa" em silêncio/ruído (vêm dos dados de treino,
# legendas de vídeos). Removidas se forem o conteúdo inteiro de um trecho.
_HALLUCINATIONS = {
    "legendas pela comunidade amara.org",
    "obrigado por assistir",
    "obrigada por assistir",
    "inscreva-se no canal",
    "legenda adriana zanotto",
    "sous-titres réalisés para la communauté d'amara.org",
}
_NORM = re.compile(r"[^\w\s.-]")


def _is_hallucination(text: str) -> bool:
    t = _NORM.sub("", text.casefold()).strip(" .")
    return t in _HALLUCINATIONS


def _whisper_model():
    """Carrega (uma vez) o modelo. Usa a pasta local; nunca baixa sozinho
    durante uma transcrição se a pasta existir."""
    key = (config.WHISPER_MODEL, config.WHISPER_DEVICE, config.WHISPER_COMPUTE)
    with _lock:
        if key not in _whisper_cache:
            from faster_whisper import WhisperModel

            local = config.MODELS_DIR / "whisper" / config.WHISPER_MODEL
            source = str(local) if (local / "model.bin").exists() else config.WHISPER_MODEL
            device = config.WHISPER_DEVICE
            if device == "auto":
                device = "cuda" if _cuda_ok() else "cpu"
            compute = config.WHISPER_COMPUTE
            if compute == "auto":
                compute = "int8_float16" if device == "cuda" else "int8"
            _whisper_cache[key] = WhisperModel(
                source, device=device, compute_type=compute,
                cpu_threads=config.CPU_THREADS,
                download_root=str(config.MODELS_DIR / "whisper"))
        return _whisper_cache[key]


def _cuda_ok() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0 and bool(
            {"int8_float16", "float16"} & ctranslate2.get_supported_compute_types("cuda"))
    except Exception:  # noqa: BLE001 — sem CUDA/driver: CPU
        return False


def _extractor() -> diarize.EmbeddingExtractor:
    path = config.DIARIZATION_MODEL
    with _lock:
        if path not in _extractor_cache:
            if not Path(path).exists():
                raise RuntimeError(
                    f"Modelo de identificação de falantes não encontrado: {path}. "
                    "Rode: python baixar_modelos.py")
            _extractor_cache[path] = diarize.EmbeddingExtractor(
                Path(path), num_threads=config.CPU_THREADS)
        return _extractor_cache[path]


def _hotwords(vocabulary: Optional[list[str]], language: Optional[str]) -> Optional[str]:
    """Texto reaplicado como contexto em toda janela de 30 s do Whisper.

    Escrito como FRASE PONTUADA, não como lista solta: no teste com uma
    reunião real de 22 min, isso levou o Whisper a pontuar o texto todo
    (de ~30 para ~6 palavras por frase) sem perder palavras. A pontuação é
    o que permite à diarização achar onde cada falante começa e termina.
    """
    names = [n.strip() for n in vocabulary or [] if n.strip()]
    terms = [t for t in config.EXTRA_TERMS if t not in names]
    if language in (None, "pt"):
        if len(names) > 1:
            text = f"Reunião com {', '.join(names[:-1])} e {names[-1]}."
        elif names:
            text = f"Reunião com {names[0]}."
        else:
            text = "Reunião de trabalho."
        if terms:
            text += f" Assuntos: {', '.join(terms)}."
    else:
        text = ", ".join(names + terms)
    # O prompt do Whisper tem limite (~224 tokens, metade p/ hotwords)
    return text[:400] or None


LACO_JANELA_SECONDS = 60.0
LACO_MIN_PALAVRAS = 4
LACO_MAX_REPETICOES = 2


class _DetectorDeLaco:
    """Descarta o trecho repetido de novo pelo Whisper (cântico, ruído).

    Frase com 4 palavras ou mais que já apareceu 2 vezes nos últimos 60 s é
    tratada como alucinação de laço, e a 3ª ocorrência sai da transcrição.
    Cânticos viram "a gente tem que ser mais um arrasto" repetido; a fala
    real quase nunca repete a mesma frase longa em tão pouco tempo.
    """

    def __init__(self) -> None:
        self._vistas: list[tuple[float, str]] = []

    def repetido(self, texto: str, fim: float) -> bool:
        chave = " ".join(_NORM.sub("", texto.casefold()).split())
        if len(chave.split()) < LACO_MIN_PALAVRAS:
            return False
        self._vistas = [(t, c) for t, c in self._vistas if fim - t <= LACO_JANELA_SECONDS]
        if sum(1 for _, c in self._vistas if c == chave) >= LACO_MAX_REPETICOES:
            return True
        self._vistas.append((fim, chave))
        return False


def _limites(min_speakers: Optional[int],
             max_speakers: Optional[int]) -> tuple[Optional[int], Optional[int]]:
    """Limites informados pelo usuário viram dica para o agrupamento.

    Acima de diarize.MAX_SPEAKERS_AUTO não são usados: um "20 e 20" digitado
    na tela forçava o agrupamento a 20 vozes numa reunião de 1 pessoa, e o
    eigengap não tem como corrigir isso.
    """
    teto = diarize.MAX_SPEAKERS_AUTO
    lo = min_speakers if min_speakers and 1 <= min_speakers <= teto else None
    hi = max_speakers if max_speakers and 1 <= max_speakers <= teto else None
    if lo and hi and lo > hi:
        lo = hi = None
    return lo, hi


def _load_audio(path: Path):
    import numpy as np
    from faster_whisper.audio import decode_audio

    return np.asarray(decode_audio(str(path), sampling_rate=diarize.SAMPLE_RATE),
                      dtype=np.float32)


class LocalEngine:
    """Três fases:
      1. Reconhecer vozes (só pelo som): quantas pessoas falam e se alguma
         já é conhecida da memória de vozes.
      2. Transcrever, já separando as falas no nº de vozes da fase 1.
      3. (no worker) Dar nome às vozes: memória de vozes + quem é chamado
         pelo nome na conversa.
    """
    name = "local"

    def __init__(self) -> None:
        # preenchidos a cada transcrição, lidos pelo worker
        self.voices: dict[str, dict] = {}       # rótulo -> {vetor, segundos}
        self.scan: dict = {}                    # resumo da fase 1

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: Optional[str],
        min_speakers: Optional[int],
        max_speakers: Optional[int],
        progress: ProgressCB,
        vocabulary: Optional[list[str]] = None,
    ) -> list[Segment]:
        from .. import vozes

        audio = _load_audio(audio_path)

        # ---- Fase 1: reconhecimento de vozes (antes de transcrever) --------
        progress("Fase 1/3 · Reconhecendo as vozes")
        extractor = _extractor()
        min_speakers, max_speakers = _limites(min_speakers, max_speakers)
        scan = diarize.scan_voices(audio, extractor, min_speakers, max_speakers)
        known = vozes.match({j: c for j, c in scan.centroids.items()})
        names = [known[j][0] for j in sorted(known)]
        self.scan = {
            "vozes": scan.count,
            "conhecidas": [{"nome": known[j][0], "similaridade": round(known[j][1], 2)}
                           for j in sorted(known)],
        }
        if scan.count == 1:
            resumo = "1 voz (só uma pessoa falando)"
        else:
            resumo = f"{scan.count} vozes"
        if names:
            resumo += " · reconhecida(s): " + ", ".join(names)
        progress(f"Fase 1/3 · {resumo}")

        # Os nomes reconhecidos pela voz também ajudam o Whisper na grafia
        vocabulary = list(dict.fromkeys([*(vocabulary or []), *names]))
        # ---- Fase 2: transcrição -----------------------------------------
        progress(f"Fase 2/3 · Carregando modelo de transcrição ({config.WHISPER_MODEL})")
        model = _whisper_model()

        segments_iter, info = model.transcribe(
            audio,
            language=language or None,
            beam_size=config.WHISPER_BEAM_SIZE,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 300},
            # Sem herdar o texto anterior: evita o Whisper entrar em laço
            # repetindo a mesma frase (falha clássica em reuniões longas).
            condition_on_previous_text=False,
            hallucination_silence_threshold=2.0,
            compression_ratio_threshold=2.4,
            no_speech_threshold=0.6,
            hotwords=_hotwords(vocabulary, language),
        )

        words: list[diarize.Word] = []
        total = info.duration or 1.0
        last_pct = -1
        laco = _DetectorDeLaco()
        for seg in segments_iter:
            if _is_hallucination(seg.text) or laco.repetido(seg.text, seg.end):
                continue
            for w in seg.words or []:
                words.append(diarize.Word(w.start, w.end, w.word))
            pct = int(min(seg.end / total, 1.0) * 100)
            if pct >= last_pct + 2:
                last_pct = pct
                progress(f"Fase 2/3 · Transcrevendo — {pct}%")

        if not words:
            return []

        progress("Fase 2/3 · Separando as falas por pessoa")
        sentences, centroids = diarize.diarize(
            audio, words, extractor,
            min_speakers=min_speakers, max_speakers=max_speakers)

        talk = {}
        for s in sentences:
            talk[s.speaker] = talk.get(s.speaker, 0.0) + s.duration
        self.voices = {
            diarize.speaker_label(j): {"vetor": [round(float(x), 6) for x in vec],
                                       "segundos": round(talk.get(j, 0.0), 1)}
            for j, vec in centroids.items()
        }

        # "Utterances" como no AssemblyAI: o bloco de um falante só termina
        # quando outra pessoa fala (ou após um silêncio longo).
        out: list[Segment] = []
        for s in sentences:
            label = diarize.speaker_label(s.speaker)
            if out and out[-1].speaker == label and s.start - out[-1].end <= UTTERANCE_MAX_GAP:
                out[-1].text = f"{out[-1].text} {s.text}"
                out[-1].end = s.end
            else:
                out.append(Segment(start=s.start, end=s.end, speaker=label, text=s.text))
        return out
