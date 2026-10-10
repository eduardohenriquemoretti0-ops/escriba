"""Motor AssemblyAI (API na nuvem).

Por que ele é o padrão: diarização nativa e madura, aceita arquivos longos
(até ~10 h), suporta português e não exige GPU na sua máquina. O SDK cuida
de upload + fila + polling. Custo na casa de US$ 0,17/h de áudio
(Universal + speaker labels) — confira valores atuais em assemblyai.com/pricing.

Nomes dos participantes: quando informados, vão como `word_boost`, que faz o
modelo preferir a grafia certa de nomes próprios. Se a API recusar esse
parâmetro (o recurso varia por modelo/conta), a transcrição é refeita sem ele
— o cliente corrige o que sobrar na tela de revisão.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .. import config
from .base import ProgressCB, Segment


def _boost_terms(vocabulary: Optional[list[str]]) -> list[str]:
    """Nome completo e cada parte (>= 3 letras), sem repetições."""
    terms: list[str] = []
    for full in vocabulary or []:
        for term in [full, *full.split()]:
            term = term.strip()
            if len(term) >= 3 and term not in terms:
                terms.append(term)
    return terms[:200]          # limite razoável da API


class AssemblyAIEngine:
    name = "assemblyai"

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
        if not config.ASSEMBLYAI_API_KEY:
            raise RuntimeError(
                "ASSEMBLYAI_API_KEY não configurada no .env "
                "(crie uma chave gratuita em assemblyai.com)."
            )

        import assemblyai as aai  # import tardio: só carrega se usado

        aai.settings.api_key = config.ASSEMBLYAI_API_KEY

        kwargs: dict = {"speaker_labels": True}
        if language:
            kwargs["language_code"] = language
        else:
            kwargs["language_detection"] = True
        # Dica de qtde. de falantes melhora a diarização quando você sabe
        # quantas pessoas estavam na reunião.
        if min_speakers and max_speakers and min_speakers == max_speakers:
            kwargs["speakers_expected"] = min_speakers
        elif max_speakers:
            kwargs["speakers_expected"] = max_speakers

        boost = _boost_terms(vocabulary)

        def run(with_boost: bool):
            cfg_kwargs = dict(kwargs)
            if with_boost and boost:
                cfg_kwargs["word_boost"] = boost
                cfg_kwargs["boost_param"] = "high"
            cfg = aai.TranscriptionConfig(**cfg_kwargs)
            return aai.Transcriber(config=cfg).transcribe(str(audio_path))

        progress("Enviando áudio e transcrevendo na nuvem")
        try:
            transcript = run(with_boost=True)
            if transcript.status == aai.TranscriptStatus.error and boost:
                # Reforço de nomes é um extra: se a API recusar (qualquer
                # motivo), tenta de novo sem ele em vez de perder a reunião.
                progress("Refazendo sem o reforço de nomes (a API recusou)")
                transcript = run(with_boost=False)
        except (TypeError, ValueError):
            if not boost:
                raise
            # versão do SDK/modelo sem word_boost
            transcript = run(with_boost=False)

        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(f"AssemblyAI retornou erro: {transcript.error}")

        segments: list[Segment] = []
        utterances = transcript.utterances or []
        for utt in utterances:
            segments.append(
                Segment(
                    start=(utt.start or 0) / 1000.0,
                    end=(utt.end or 0) / 1000.0,
                    speaker=str(utt.speaker or "?"),
                    text=utt.text or "",
                )
            )

        # Sem utterances (ex.: 1 falante só e diarização vazia) -> texto único
        if not segments and transcript.text:
            segments.append(Segment(0.0, 0.0, "A", transcript.text))
        return segments
