"""Perfil do cliente que adquiriu o software + imagem da assinatura.

Fica em data/profile/ (perfil.json e assinatura.png). A assinatura entra nos
PDFs da ata e da transcrição; os participantes assinam à parte, nas linhas em
branco que o PDF reserva para cada um.

Atenção: é a IMAGEM da assinatura, não uma assinatura digital certificada
(ICP-Brasil / gov.br). Serve para documentos internos e para imprimir/colher
as demais assinaturas; para ata que precise de registro em cartório ou de
certificação, use a assinatura digital apropriada.
"""
from __future__ import annotations

import io
import json
from typing import Optional

from PIL import Image, ImageOps, UnidentifiedImageError

from . import config
from .review import clean_name

PROFILE_JSON = config.PROFILE_DIR / "perfil.json"
SIGNATURE_PNG = config.PROFILE_DIR / "assinatura.png"

FIELDS = ("nome", "cargo", "organizacao", "email")
MAX_SIDE = 700               # px — o suficiente para sair nítida no PDF
INK = (16, 24, 40)           # tinta escura, aplicada em fotos de assinatura
Image.MAX_IMAGE_PIXELS = 40_000_000


class SignatureError(ValueError):
    pass


# --------------------------------------------------------------------------
# Dados do perfil
# --------------------------------------------------------------------------
def load() -> dict:
    data = {k: "" for k in FIELDS}
    if PROFILE_JSON.exists():
        try:
            saved = json.loads(PROFILE_JSON.read_text(encoding="utf-8-sig"))
            data.update({k: str(saved.get(k, "")) for k in FIELDS})
        except (OSError, ValueError):
            pass
    return data


def save(values: dict) -> dict:
    data = {k: clean_name(values.get(k), 120) for k in FIELDS}
    PROFILE_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    return data


# --------------------------------------------------------------------------
# Assinatura
# --------------------------------------------------------------------------
def has_signature() -> bool:
    return SIGNATURE_PNG.exists()


def signature_bytes() -> Optional[bytes]:
    return SIGNATURE_PNG.read_bytes() if SIGNATURE_PNG.exists() else None


def clear_signature() -> None:
    SIGNATURE_PNG.unlink(missing_ok=True)


def _ink_from_photo(img: Image.Image) -> Image.Image:
    """Foto/scan (fundo claro, traço escuro) -> PNG com fundo transparente."""
    gray = ImageOps.autocontrast(img.convert("L"), cutoff=1)
    hist = gray.histogram()
    total, acc, background = sum(hist), 0, 255
    for level, n in enumerate(hist):        # nível de cinza do "papel"
        acc += n
        if acc >= total * 0.85:
            background = level
            break
    span = max(background - 60, 1)
    alpha = gray.point(
        lambda v: max(0, min(255, int((background - v) * 255 / span))))
    # limpa a textura/sombra do papel: o que for bem fraco vira transparente
    alpha = alpha.point(lambda a: 0 if a < 40 else int((a - 40) * 255 / 215))
    out = Image.new("RGBA", img.size, INK + (0,))
    out.putalpha(alpha)
    return out


def set_signature(raw: bytes) -> None:
    """Valida, limpa (fundo transparente, recorte, tamanho) e grava a imagem."""
    if not raw or len(raw) > config.MAX_SIGNATURE_BYTES:
        raise SignatureError("Imagem vazia ou maior que 5 MB.")
    try:
        img = Image.open(io.BytesIO(raw))
        if img.format not in ("PNG", "JPEG", "WEBP"):
            raise SignatureError("Use uma imagem PNG, JPG ou WEBP.")
        img = ImageOps.exif_transpose(img)
        img.load()
    except SignatureError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise SignatureError("Não consegui abrir essa imagem.") from None

    has_alpha = img.mode in ("RGBA", "LA") or "transparency" in img.info
    rgba = img.convert("RGBA")
    if has_alpha and rgba.getchannel("A").getextrema()[0] < 250:
        out = rgba                        # já tem fundo transparente (desenho)
    else:
        out = _ink_from_photo(rgba)

    # recorta no traço (ignora ruído bem fraco) e dá uma folga
    bbox = out.getchannel("A").point(lambda a: 255 if a > 24 else 0).getbbox()
    if not bbox:
        raise SignatureError("Não encontrei nenhum traço de assinatura na imagem.")
    pad = 8
    bbox = (max(bbox[0] - pad, 0), max(bbox[1] - pad, 0),
            min(bbox[2] + pad, out.width), min(bbox[3] + pad, out.height))
    out = out.crop(bbox)
    if max(out.size) > MAX_SIDE:
        out.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    out.save(SIGNATURE_PNG, format="PNG", optimize=True)
