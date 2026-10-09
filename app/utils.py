"""Utilitários: escrita atômica, logging, slug de arquivo."""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("escriba")


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Escreve em arquivo temporário e faz rename atômico.

    FIX #3 (escriba): nunca sobrescrever o arquivo de destino com conteúdo
    parcial — escreve em `.tmp`, dá flush/fsync e renomeia (operação atômica
    no mesmo filesystem).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding=encoding) as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)  # atômico


_SAFE = re.compile(r"[^a-zA-Z0-9_.-]+")


def safe_slug(name: str, maxlen: int = 80) -> str:
    """Nome de arquivo seguro (sem path traversal nem caracteres estranhos)."""
    base = os.path.basename(name)
    slug = _SAFE.sub("-", base).strip("-.") or "arquivo"
    return slug[:maxlen]
