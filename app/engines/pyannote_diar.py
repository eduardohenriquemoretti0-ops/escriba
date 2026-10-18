"""Separação de falantes com o pyannote community-1 (PyTorch), quando a máquina aguenta.

Medido no laboratório contra o gabarito do salão 10/10: DER 18,7 contra 25,0 do
caminho ONNX. Precisa de PyTorch; com GPU NVIDIA roda em paralelo com o Whisper
(que fica na CPU), então quase não soma tempo. Sem GPU, só com 16 GB+ de RAM.

Qualquer falha (sem torch, sem modelo, falta de memória na GPU) devolve None e o
motor segue pelo caminho ONNX de sempre: o pyannote nunca derruba a transcrição.

Pesos: pyannote/speaker-diarization-community-1, licença CC-BY-4.0 (exige
atribuição; ver models/pyannote-community-1/README.md).
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Optional

import numpy as np

from .. import config

log = logging.getLogger(__name__)

RAM_MINIMA_GB = 15.0      # máquina "de 16 GB" reporta ~15,5 GB ao sistema

Turno = tuple[float, float, str]


def _ram_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):   # Windows não tem sysconf
        try:
            import psutil
            return psutil.virtual_memory().total / 2**30
        except Exception:  # noqa: BLE001 — sem como medir: não arrisca
            return 0.0


def disponivel() -> tuple[bool, str]:
    """(usar?, motivo). Barato: só importa o torch se o resto já permitir."""
    modo = config.DIARIZADOR
    if modo == "onnx":
        return False, "desligado (ESCRIBA_DIARIZADOR=onnx)"
    if not (config.PYANNOTE_DIR / "config.yaml").exists():
        return False, f"modelo ausente em {config.PYANNOTE_DIR}"
    try:
        import torch
        import pyannote.audio  # noqa: F401
    except Exception as exc:  # noqa: BLE001 — pacote sem torch/pyannote
        return False, f"PyTorch/pyannote indisponível ({exc.__class__.__name__})"
    if torch.cuda.is_available():
        return True, f"GPU {torch.cuda.get_device_name(0)}"
    ram = _ram_gb()
    if modo == "pyannote" or ram >= RAM_MINIMA_GB:
        return True, f"CPU, {ram:.0f} GB de RAM"
    return False, f"sem GPU e só {ram:.0f} GB de RAM"


def _rodar(audio: np.ndarray, sr: int) -> list[Turno]:
    import torch
    from pyannote.audio import Pipeline

    pipe = Pipeline.from_pretrained(str(config.PYANNOTE_DIR))
    pipe.segmentation_batch_size = config.PYANNOTE_LOTE
    pipe.embedding_batch_size = config.PYANNOTE_LOTE
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe.to(dev)
    onda = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)
    try:
        saida = pipe({"waveform": onda, "sample_rate": sr})
    finally:
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    # exclusive: um falante por instante (é o que a transcrição precisa)
    diar = getattr(saida, "exclusive_speaker_diarization", None) or saida.speaker_diarization
    return [(float(t.start), float(t.end), str(spk))
            for t, _, spk in diar.itertracks(yield_label=True)]


class Rodada:
    """Roda o pyannote numa thread, em paralelo com o Whisper."""

    def __init__(self, audio: np.ndarray, sr: int) -> None:
        self.turnos: Optional[list[Turno]] = None
        self.erro: Optional[str] = None
        self._t = threading.Thread(target=self._run, args=(audio, sr),
                                   name="pyannote", daemon=True)
        self._t.start()

    def _run(self, audio: np.ndarray, sr: int) -> None:
        # Só laboratório: ESCRIBA_PYANNOTE_CACHE guarda os turnos por áudio
        # (a mesma gravação medida várias vezes não roda a GPU de novo).
        cache_dir = os.getenv("ESCRIBA_PYANNOTE_CACHE", "")
        cache = None
        if cache_dir:
            import hashlib
            import json
            from pathlib import Path
            chave = hashlib.sha1(np.ascontiguousarray(audio).tobytes()).hexdigest()[:16]
            cache = Path(cache_dir) / f"{chave}.json"
            if cache.exists():
                self.turnos = [tuple(t) for t in json.loads(cache.read_text())]
                return
        try:
            self.turnos = _rodar(audio, sr)
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(self.turnos))
        except Exception as exc:  # noqa: BLE001 — qualquer falha: volta ao ONNX
            self.erro = f"{exc.__class__.__name__}: {exc}"
            log.warning("pyannote falhou; seguindo pelo caminho ONNX: %s", self.erro)

    def resultado(self) -> Optional[list[Turno]]:
        self._t.join()
        return self.turnos or None
