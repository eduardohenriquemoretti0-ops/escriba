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

import dataclasses
import logging
import os
import re
import threading

import numpy as np
from pathlib import Path
from typing import Optional, Sequence

from .. import config
from .. import irmaos as _irmaos
from . import diarize
from .base import CORO_LABEL, CORO_TEXT, ProgressCB, Segment

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


HOTWORDS_MAX_CHARS = 400          # teto do texto de hotwords (antes cortava no meio)
HOTWORDS_IRMAOS_CHARS = 160       # frase da lista de irmãos: nomes mais recentes que cabem
_log = logging.getLogger(__name__)


def _corta(text: str, limite: int) -> str:
    """Corta no fim de uma frase, nunca no meio de uma palavra."""
    if len(text) <= limite:
        return text
    cortado = text[:limite]
    fim = cortado.rfind(". ")
    return cortado[:fim + 1] if fim > 0 else cortado


def _nome_falado(nome: str, todos: Sequence[str]) -> str:
    """Como o nome é dito na reunião: o 1º nome ("Carmem Carlotti" -> "Carmem").

    Se outro nome da lista começa igual ("José Eduardo" e "José Carlos"), usa
    os dois primeiros. Medido no salão 10/10: o nome completo gasta o orçamento
    do hotwords, a frase dos irmãos some e a separação piora (DER 22,4 -> 27,4)."""
    partes = nome.split()
    if len(partes) < 2:
        return nome
    primeiro = _irmaos.fold(partes[0])
    outros = {_irmaos.fold(t) for t in todos
              if t.split() and _irmaos.fold(t.split()[0]) == primeiro}
    repetido = len(outros - {_irmaos.fold(nome)}) > 0
    return " ".join(partes[:2]) if repetido else partes[0]


def _hotwords(vocabulary: Optional[list[str]], language: Optional[str],
              irmaos: Sequence = ()) -> Optional[str]:
    """Texto reaplicado como contexto em toda janela de 30 s do Whisper.

    Escrito como FRASE PONTUADA, não como lista solta: no teste com uma
    reunião real de 22 min, isso levou o Whisper a pontuar o texto todo
    (de ~30 para ~6 palavras por frase) sem perder palavras. A pontuação é
    o que permite à diarização achar onde cada falante começa e termina.
    """
    names = [n.strip() for n in vocabulary or [] if n.strip()]
    terms = [t for t in config.EXTRA_TERMS if t not in names]
    if language in (None, "pt"):
        todos = [*names, *(e.nome for e in irmaos)]
        names = list(dict.fromkeys(_nome_falado(n, todos) for n in names))
        if len(names) > 1:
            text = f"Reunião com {', '.join(names[:-1])} e {names[-1]}."
        elif names:
            text = f"Reunião com {names[0]}."
        else:
            text = "Reunião de trabalho."
        # Irmãos da lista (mais recentes primeiro), em frase pontuada à parte
        frase = _irmaos.frase_irmaos(list(irmaos), HOTWORDS_IRMAOS_CHARS, excluir=names,
                                     forma=lambda n: _nome_falado(n, todos))
        if frase:
            text += f" {frase}"
        if terms:
            text += f" Assuntos: {', '.join(terms)}."
    else:
        extra = [e.nome for e in irmaos if e.nome not in names]
        text = ", ".join(names + extra + terms)
    # O prompt do Whisper tem limite (~224 tokens, metade p/ hotwords)
    return _corta(text, HOTWORDS_MAX_CHARS) or None


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


# Cântico: o Whisper quase não escreve nada enquanto a congregação canta (o VAD
# trata como não-fala). Medido no salão: silêncio de fala de 15 s ou mais com
# som acima de -20 dBFS = cântico; pausa normal fica em -20 a -32 dBFS.
CORO_GAP_SECONDS = 15.0
CORO_MIN_DBFS = -20.0


def _e_coro(seg, laco: "_DetectorDeLaco") -> bool:
    """Trecho de texto em laço (o Whisper repetiu a mesma frase): não é fala."""
    return laco.repetido(seg.text, seg.end)


CORO_WINDOW_SECONDS = 5.0


def _som_alto(audio: np.ndarray, a: float, b: float) -> bool:
    sr = diarize.SAMPLE_RATE
    x = audio[int(a * sr):int(b * sr)].astype(np.float64)
    if x.size == 0:
        return False
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9) >= CORO_MIN_DBFS


def _lacunas_de_coro(audio: np.ndarray, words: list) -> list[tuple[float, float]]:
    """Trechos sem nenhuma palavra transcrita com som alto por CORO_GAP_SECONDS
    ou mais: cântico da congregação, marcado "em coro".

    Dentro de cada intervalo entre palavras, o som é medido em janelas de
    5 s; só uma sequência contínua de janelas altas com 15 s ou mais conta.
    Assim um cântico seguido de silêncio não some na média.
    """
    fim = len(audio) / diarize.SAMPLE_RATE
    trechos: list[tuple[float, float]] = []
    cursor = 0.0
    for w in sorted(words, key=lambda w: w.start) + [diarize.Word(fim, fim, "")]:
        if w.start - cursor >= CORO_GAP_SECONDS:
            corrida: list[tuple[float, float]] = []   # janelas altas seguidas
            t = cursor
            while t < w.start:
                t2 = min(t + CORO_WINDOW_SECONDS, w.start)
                if _som_alto(audio, t, t2):
                    corrida.append((t, t2))
                else:
                    _fecha_corrida(corrida, trechos)
                    corrida = []
                t = t2
            _fecha_corrida(corrida, trechos)
        cursor = max(cursor, w.end)
    return trechos


def _fecha_corrida(corrida: list[tuple[float, float]],
                   trechos: list[tuple[float, float]]) -> None:
    if corrida and corrida[-1][1] - corrida[0][0] >= CORO_GAP_SECONDS:
        trechos.append((corrida[0][0], corrida[-1][1]))


def _agrupa_coro(trechos: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Junta trechos de coro separados por até UTTERANCE_MAX_GAP segundos."""
    out: list[list[float]] = []
    for a, b in sorted(trechos):
        if out and a - out[-1][1] <= UTTERANCE_MAX_GAP:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


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


def _whisper_kwargs(language: Optional[str]) -> dict:
    """Parâmetros do Whisper (os mesmos na transcrição e nos buracos).

    Hotwords vão à parte, como frase pontuada; nunca `initial_prompt` junto
    (perdeu palavras no teste: 313 para 280)."""
    return dict(
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
    )


def _junta_buracos(sentences: list, centroids: dict, buracos: list,
                   extractor: diarize.EmbeddingExtractor,
                   audio: np.ndarray) -> list:
    """Frases das palavras recuperadas nos buracos, com a voz mais parecida.

    O agrupamento já foi feito sem elas; cada frase nova vai para o falante
    de centróide mais próximo (cosseno)."""
    chaves = sorted(centroids)
    cen = np.stack([centroids[j] for j in chaves]) if chaves else None
    novas = []
    for ws in buracos:
        for sen in diarize.build_sentences(ws):
            vec = diarize._embed_span(extractor, audio, sen.start,
                                      max(sen.end, sen.start + 0.5))
            sen.speaker = int(chaves[int(np.argmax(cen @ vec))]) if cen is not None else 0
            sen.embedding = vec if sen.duration >= diarize.MIN_EMBED_SECONDS else None
            novas.append(sen)
    return sorted(sentences + novas, key=lambda s: s.start)


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
        self.estudo: list = []                  # pessoas do estudo (turnos)
        self.estudo_info: dict = {}             # estrutura detectada (dirigente...)
        self.buracos_info: dict = {}            # trechos pulados retranscritos

    def _buracos(self, audio: np.ndarray, words: list, coro: list, model,
                 hot: Optional[str], language: Optional[str],
                 progress: ProgressCB) -> list[list[diarize.Word]]:
        """Retranscreve os trechos pulados, dentro do teto de CPU (config)."""
        from .. import buracos

        self.buracos_info = {}
        if not config.BURACOS_ATIVO:
            return []
        duracao = len(audio) / diarize.SAMPLE_RATE
        kw = _whisper_kwargs(language)
        kw["vad_parameters"] = {**kw["vad_parameters"], "threshold": buracos.VAD_LIMIAR}

        def transcrever(y: np.ndarray):
            segs, _ = model.transcribe(y, hotwords=hot, **kw)
            return segs

        novos, rel = buracos.retranscreve(
            audio, [(w.start, w.end) for w in words], coro, transcrever,
            teto_cpu_s=config.BURACOS_TETO_CPU * duracao,
            e_alucinacao=_is_hallucination, progress=progress)
        self.buracos_info = rel.como_dict()
        _log.info("Buracos: %s", self.buracos_info)
        return novos

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
        entradas = _irmaos.listar_seguro(_log)     # lista de irmãos (só nomes)

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
            resumo = "1 voz detectada (só uma pessoa falando)"
        else:
            resumo = f"{scan.count} vozes detectadas"
        if names:
            resumo += " · reconhecida(s): " + ", ".join(names)
        progress(f"Fase 1/3 · {resumo}")

        # Os nomes reconhecidos pela voz também ajudam o Whisper na grafia
        vocabulary = list(dict.fromkeys([*(vocabulary or []), *names]))
        # Separação de falantes pelo pyannote (se a máquina aguenta): roda na
        # GPU em paralelo com o Whisper; se falhar, segue o caminho ONNX.
        from . import pyannote_diar
        usar_pya, motivo_pya = pyannote_diar.disponivel()
        rodada_pya = pyannote_diar.Rodada(audio, diarize.SAMPLE_RATE) if usar_pya else None
        self.scan["diarizador"] = ("pyannote community-1 · " if usar_pya else "ONNX · ") + motivo_pya
        _log.info("separação de falantes: %s", self.scan["diarizador"])

        # ---- Fase 2: transcrição -----------------------------------------
        progress(f"Fase 2/3 · Carregando modelo de transcrição ({config.WHISPER_MODEL})")
        model = _whisper_model()

        hot = _hotwords(vocabulary, language, entradas)
        segments_iter, info = model.transcribe(
            audio, hotwords=hot, **_whisper_kwargs(language))

        words: list[diarize.Word] = []
        coro: list[tuple[float, float]] = []      # trechos de cântico/coro
        total = info.duration or 1.0
        last_pct = -1
        laco = _DetectorDeLaco()
        for seg in segments_iter:
            if _is_hallucination(seg.text):
                continue
            if _e_coro(seg, laco):
                coro.append((seg.start, seg.end))
                continue
            for w in seg.words or []:
                words.append(diarize.Word(w.start, w.end, w.word))
            pct = int(min(seg.end / total, 1.0) * 100)
            if pct >= last_pct + 2:
                last_pct = pct
                progress(f"Fase 2/3 · Transcrevendo — {pct}%")

        if not words and not coro:
            return []
        coro.extend(_lacunas_de_coro(audio, words))
        coro = _agrupa_coro(coro)

        # Trechos que o Whisper pulou (áudio baixo): só depois do coro, para
        # cântico não virar buraco. Entram depois da separação de falantes.
        buracos = self._buracos(audio, words, coro, model, hot, language, progress)

        progress("Fase 2/3 · Separando as falas por pessoa")
        turnos_pya = rodada_pya.resultado() if rodada_pya else None
        if turnos_pya:
            sentences, centroids = diarize.diarize_por_turnos(
                audio, words, extractor, turnos_pya)
        else:
            if rodada_pya:
                self.scan["diarizador"] = f"ONNX · pyannote falhou ({rodada_pya.erro})"
            sentences, centroids = diarize.diarize(
                audio, words, extractor,
                min_speakers=min_speakers, max_speakers=max_speakers)
        if buracos:
            sentences = _junta_buracos(sentences, centroids, buracos, extractor, audio)

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
        # Estudo bíblico: comentários separados por turno, guiados pela estrutura
        # (dirigente chama e agradece). Sem a estrutura, nada muda.
        from .. import estudo, turnos

        def embed(a: float, b: float) -> np.ndarray:
            return diarize._embed_span(extractor, audio, a, b)

        # Com o pyannote, os falantes já vêm separados: `turnos.reatribui` e
        # `estudo.aplica` foram feitos para partir os grupos grossos do ONNX e,
        # por cima do pyannote, pioram (salão 10/10: DER 30,6 com eles, 23,6
        # sem). Os nomes vêm do motor de estrutura, depois. ESCRIBA_PYANNOTE_POS
        # (tudo | sem_estudo | sem_reatribui | nada) só para laboratório.
        pos = os.getenv("ESCRIBA_PYANNOTE_POS", "nada") if turnos_pya else "tudo"
        pessoas = []
        self.estudo_info = {"ativo": False}
        if pos in ("tudo", "sem_estudo"):
            out, pessoas, self.estudo_info = turnos.reatribui(
                out, embed, irmaos=entradas, participantes=list(vocabulary or []))
        # Estudo por palavra (dirigente ↔ chamado, nome só com evidência).
        # A guarda deixa a conversa comum como está.
        progress("Fase 2/3 · Conferindo a estrutura da reunião")
        if pos in ("tudo", "sem_reatribui"):
            res = estudo.aplica(sentences, out, pessoas, embed, entradas,
                                list(vocabulary or []))
            if res.ativa:
                out, pessoas = res.segmentos, res.pessoas
            self.estudo_info = {**self.estudo_info, "estudo_por_palavra": res.info}
        if self.buracos_info:
            self.estudo_info["buracos"] = self.buracos_info
        self.estudo = [dataclasses.asdict(p) for p in pessoas]

        # Cânticos: um bloco "em coro" por trecho, fora da separação de falantes
        for a, b in coro:
            out.append(Segment(start=a, end=b, speaker=CORO_LABEL, text=CORO_TEXT))
        out.sort(key=lambda sg: sg.start)
        return out
