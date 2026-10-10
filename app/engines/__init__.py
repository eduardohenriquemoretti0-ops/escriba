from .assemblyai_engine import AssemblyAIEngine
from .base import Segment, merge_segments
from .whisperx_engine import WhisperXEngine

_ENGINES = {
    "assemblyai": AssemblyAIEngine,
    "whisperx": WhisperXEngine,
}


def get_engine(name: str):
    try:
        return _ENGINES[name]()
    except KeyError:
        raise ValueError(
            f"Engine '{name}' desconhecido. Disponíveis: {', '.join(_ENGINES)}"
        ) from None


__all__ = ["get_engine", "Segment", "merge_segments"]
