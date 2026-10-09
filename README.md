# escriba

App de **transcrição de áudio → ata de reunião**. Transcrição **local e grátis**
(Faster-Whisper) por padrão, com provider de API opcional; geração de ata via
Claude (opcional). Expõe **CLI** (uso imediato) e **API FastAPI**.

> ⚠️ Versão de reconstrução (2026-10-08) feita a partir do blueprint documentado.
> O código-fonte original será reconciliado quando chegar no flashdrive.
> Incorpora os 5 fixes críticos + 7 médios documentados do escriba.

## Instalação

Pré-requisitos do sistema (precisa de `sudo` uma vez):
```bash
sudo apt install -y python3-venv python3-pip ffmpeg
```
Depois, no diretório do projeto:
```bash
cd ~/escriba
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # ajuste conforme necessário
```

## Uso rápido (CLI) — transcrever um áudio

```bash
source ~/escriba/venv/bin/activate
python -m app.cli transcrever /caminho/reuniao.m4a --idioma pt
# também gerar a ata (precisa de ANTHROPIC_API_KEY e ESCRIBA_SUMMARY_ENABLED):
python -m app.cli transcrever /caminho/reuniao.m4a --ata
```
Saídas salvas em `ESCRIBA_DATA_DIR` (padrão `~/escriba/data/`):
`<arquivo>.transcript.txt` e, se pedida, `<arquivo>.ata.md`.

> A primeira execução baixa o modelo Whisper (`small` ≈ 460 MB). Para mais
> precisão, `--modelo medium` ou `large-v3` (mais lento). CPU usa `int8`.

## Uso como API

```bash
source ~/escriba/venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```
- `POST /transcrever` (multipart: `arquivo`, query `ata=true|false`)
- `GET /jobs/{id}` · `GET /jobs/{id}/transcript` · `DELETE /jobs/{id}`
- `GET /health`

Auth: defina `ESCRIBA_API_TOKEN` e envie `Authorization: Bearer <token>`.
Rate limit por IP+rota (config via `.env`).

## Testes
```bash
source ~/escriba/venv/bin/activate && pytest -v
```

## Segurança embutida (fixes do escriba)
- Rate limit **antes** do auth, por IP, em todos os métodos (C4).
- `ffprobe` com `-protocol_whitelist file` — anti-SSRF (C5).
- Escrita atômica (temp + rename) — nunca sobrescreve entrada (C3).
- Token constant-time (`hmac.compare_digest`) (M1); `job_id` por whitelist (M2).
- `warnings` (não-fatal) separado de `error` (fatal) (M3); sem falha silenciosa (M4);
  rate-limit com limpeza/TTL (M5).
- Client de API paga **lazy-init** (nunca no nível de módulo).

## Estrutura
```
app/
  config.py        settings via .env
  audio.py         validação/ffprobe (anti-SSRF)
  security.py      token, job_id, rate limiter
  jobs.py          job model + store TTL
  transcribe/      base, local_whisper, engine (fallback)
  summary.py       ata via Claude (auto-seleção de modelo, lazy)
  pipeline.py      orquestra tudo
  cli.py           CLI
  main.py          API FastAPI
tests/
```
