"""Geração de ata a partir da transcrição, via Claude.

Decisões de arquitetura do escriba:
- Auto-seleção de modelo por tamanho: Haiku (< threshold) → Sonnet (≥ threshold).
- MAX_CHARS limita o input (margem segura da janela de contexto).
- Lazy-init do client Anthropic: instanciado DENTRO da função, nunca no nível
  de módulo — senão o `import` quebra sem ANTHROPIC_API_KEY mesmo sem usar.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import settings
from .utils import log

SYSTEM_PROMPT = (
    "Você é um secretário de reuniões. A partir da transcrição fornecida, "
    "gere uma ATA de reunião clara e objetiva em português brasileiro, em Markdown, com:\n"
    "1. **Resumo executivo** (3-5 linhas).\n"
    "2. **Pauta / tópicos discutidos** (bullets).\n"
    "3. **Decisões tomadas**.\n"
    "4. **Ações / próximos passos** (responsável quando identificável, prazo se citado).\n"
    "5. **Pontos em aberto**.\n"
    "Não invente nomes, números ou compromissos que não estejam na transcrição."
)


class SummaryUnavailableError(Exception):
    """Sumarização indisponível (desligada, sem chave, ou SDK ausente)."""


@dataclass
class AtaResult:
    markdown: str
    model: str
    warnings: list[str]


def _pick_model(n_chars: int) -> str:
    """Auto-seleção por tamanho (decisão escriba)."""
    if n_chars >= settings.summary_threshold_chars:
        return settings.model_large
    return settings.model_small


def generate_ata(transcript: str) -> AtaResult:
    """Gera a ata. Lazy-init do client; trunca com AVISO (não silenciosamente)."""
    if not settings.summary_enabled:
        raise SummaryUnavailableError(
            "Sumarização desligada (defina ESCRIBA_SUMMARY_ENABLED=true)."
        )
    try:
        import anthropic
    except Exception as e:
        raise SummaryUnavailableError("SDK anthropic não instalado.") from e

    warnings: list[str] = []
    text = transcript
    if len(text) > settings.max_chars:
        warnings.append(
            f"Transcrição ({len(text)} chars) excede MAX_CHARS "
            f"({settings.max_chars}); truncada para a ata."
        )
        log.warning(warnings[-1])
        text = text[: settings.max_chars]

    model = _pick_model(len(text))
    log.info("Gerando ata com modelo=%s (%d chars).", model, len(text))

    try:
        client = anthropic.Anthropic()  # lazy-init — resolve credenciais do ambiente
        # Streaming evita timeout em entradas/saídas longas.
        with client.messages.stream(
            model=model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Transcrição:\n\n{text}"}],
        ) as stream:
            message = stream.get_final_message()
    except anthropic.AuthenticationError as e:
        raise SummaryUnavailableError("ANTHROPIC_API_KEY inválida/ausente.") from e
    except Exception as e:  # erro de API → reporta, não trava a transcrição
        raise SummaryUnavailableError(f"Falha ao chamar a API: {e}") from e

    md = "".join(b.text for b in message.content if b.type == "text").strip()
    return AtaResult(markdown=md, model=model, warnings=warnings)
