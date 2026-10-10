"""Ata automática da reunião (opcional).

Se ANTHROPIC_API_KEY estiver no .env e o job pedir ata, o transcript vai para
a API da Anthropic e volta um ata.md com resumo executivo, decisões, ações e
pontos em aberto — o equivalente ao "AI Summary" do Otter.

Os nomes dos participantes informados pelo cliente entram no pedido, para a
ata sair com a grafia certa. Custo aproximado: uma reunião de 2 h (~40k tokens
de entrada) fica na casa de poucos centavos de dólar com Sonnet. Ajuste
ESCRIBA_SUMMARY_MODEL se quiser outro modelo.
"""
from __future__ import annotations

from typing import Optional

from . import config

MAX_CHARS = 350_000  # margem de segurança p/ reuniões muito longas

PROMPT = """A seguir está a transcrição de uma reunião, com falantes \
identificados. Gere a ata em Markdown, em português, com exatamente estas \
seções:

# Ata da reunião

## Resumo executivo
(1 parágrafo objetivo)

## Principais tópicos
(um bullet por tópico, com 1-2 frases cada)

## Decisões tomadas
(somente decisões explícitas; se não houver, escreva "Nenhuma decisão \
explícita registrada")

## Ações e pendências
(tabela: Ação | Responsável | Prazo — use "não definido" quando a transcrição \
não disser)

## Pontos em aberto
(dúvidas ou temas adiados)

Seja fiel à transcrição: não invente responsáveis, prazos nem decisões. \
Quando os falantes se identificarem pelo nome, use os nomes em vez de \
"Falante N".
{participantes}
TRANSCRIÇÃO:
"""


def build_prompt(transcript_txt: str,
                 participants: Optional[list[str]] = None) -> str:
    bloco = ""
    if participants:
        lista = "\n".join(f"- {p}" for p in participants)
        bloco = (
            "\nNomes dos participantes, com a grafia correta (use exatamente "
            "esta grafia sempre que citar essas pessoas):\n" + lista + "\n"
        )
    return PROMPT.format(participantes=bloco) + transcript_txt[:MAX_CHARS]


def generate_ata(transcript_txt: str,
                 participants: Optional[list[str]] = None) -> str:
    if not config.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY não configurada no .env.")

    import anthropic  # import tardio

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=config.SUMMARY_MODEL,
        max_tokens=4000,
        messages=[{"role": "user",
                   "content": build_prompt(transcript_txt, participants)}],
    )
    return "".join(
        block.text for block in message.content if block.type == "text"
    )
