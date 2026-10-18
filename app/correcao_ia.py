"""Passe de IA ANCORADA para corrigir o texto da transcrição (Track 2).

Corrige o que a lógica fonética não alcança: garble severo de nome ("Palpino",
"Módulo Vir" -> "Ademir Baldino") e erro óbvio de palavra do ASR ("boazinha
dessa lei" -> "damos boas-vindas"), SEM mudar o sentido e SEM inventar.

Regra do Eduardo (precisão): nunca inventar. Por isso:
- O modelo recebe a LISTA de nomes permitidos (participantes + papéis) e só pode
  trocar um nome por um desses; não pode criar nome novo.
- Corrige só erro CLARO de reconhecimento; na dúvida, mantém o original.
- Não resume, não acrescenta, não remove frase. A saída tem de ter o MESMO número
  de blocos, na mesma ordem.
- Guarda dura: se a resposta não casar (contagem/formato), devolve o texto
  ORIGINAL do bloco — nunca piora.

Custo: usa a API Anthropic (gasto). Não roda no pipeline automático; é passo
sob demanda (o worker/menu chama quando o Eduardo pedir).
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from . import config

log = logging.getLogger("escriba.correcao_ia")

# orçamento de caracteres por requisição (chunk de blocos)
CHUNK_CHARS = 8000

_PROMPT = """Você corrige erros de RECONHECIMENTO DE FALA (ASR) em uma transcrição de reunião religiosa em português do Brasil. NÃO resuma, NÃO interprete, NÃO acrescente nem remova conteúdo.

Nomes permitidos (só estes podem ser usados ao corrigir um nome; nunca invente outro):
{nomes}

Regras:
1. Troque um nome mal reconhecido pelo nome permitido correto SÓ quando estiver claro pelo contexto (ex.: o presidente anunciando o orador). Na dúvida, mantenha o original.
2. Corrija palavras claramente mal ouvidas quando o sentido ficar óbvio (ex.: "boazinha dessa lei" -> "damos boas-vindas"). Se não tiver certeza, mantenha.
3. Não mude o sentido, não reescreva estilo, não junte nem divida frases.
4. Responda APENAS com um array JSON de strings, uma por bloco, na MESMA ordem e MESMA quantidade dos blocos de entrada. Nada fora do JSON.

Blocos (JSON de entrada, cada item tem "quem" e "texto"):
{blocos}"""


def build_prompt(blocos: list[dict], nomes: list[str]) -> str:
    return _PROMPT.format(
        nomes="\n".join(f"- {n}" for n in nomes) or "(nenhum)",
        blocos=json.dumps([{"quem": b.get("quem", ""), "texto": b.get("texto", "")}
                           for b in blocos], ensure_ascii=False),
    )


def _parse(resposta: str, n: int) -> Optional[list[str]]:
    """Extrai o array JSON de strings; None se não casar a contagem."""
    s = resposta.strip()
    i, j = s.find("["), s.rfind("]")
    if i < 0 or j <= i:
        return None
    try:
        arr = json.loads(s[i:j + 1])
    except ValueError:
        return None
    if not isinstance(arr, list) or len(arr) != n or not all(isinstance(x, str) for x in arr):
        return None
    return arr


def _chunks(blocos: list[dict]) -> list[list[dict]]:
    out, cur, tam = [], [], 0
    for b in blocos:
        t = len(b.get("texto", ""))
        if cur and tam + t > CHUNK_CHARS:
            out.append(cur); cur, tam = [], 0
        cur.append(b); tam += t
    if cur:
        out.append(cur)
    return out


def corrige_blocos(blocos: list[dict], nomes: list[str], client=None) -> list[str]:
    """Devolve o texto corrigido de cada bloco (mesma ordem). Bloco que o modelo
    não corrigir com segurança volta com o texto ORIGINAL."""
    if client is None:
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY não configurada no .env.")
        import anthropic  # import tardio
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    saida: list[str] = []
    for chunk in _chunks(blocos):
        originais = [b.get("texto", "") for b in chunk]
        try:
            msg = client.messages.create(
                model=config.SUMMARY_MODEL, max_tokens=8000,
                messages=[{"role": "user", "content": build_prompt(chunk, nomes)}])
            txt = "".join(b.text for b in msg.content if b.type == "text")
            corr = _parse(txt, len(chunk))
        except Exception:                      # noqa: BLE001 — nunca derruba
            log.exception("passe de IA falhou no chunk; mantém original")
            corr = None
        saida.extend(corr if corr is not None else originais)
    return saida
