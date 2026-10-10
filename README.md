# Escriba — transcrição de reuniões, self-hosted

App web local no estilo Otter.ai: recebe áudio ou vídeo, transcreve com
identificação de falantes (diarização) e exporta **TXT, SRT, VTT, JSON e PDF** —
com **ata automática** opcional gerada por IA.

**Novidades da v2** (detalhes na seção [Nomes, assinatura e e-mail](#nomes-assinatura-e-e-mail)):
1. **Nomes dos participantes** — o cliente informa a grafia correta logo após a
   transcrição; ela vale para os arquivos, para a ata e para as linhas de assinatura.
2. **Envio por e-mail** da ata e da transcrição (SMTP), direto da tela.
3. **Assinatura do cliente** — cadastrada uma vez; a ata e a transcrição em PDF
   saem já assinadas por ele, com linhas em branco para cada participante assinar.

Feito para reuniões longas (2 h+), com vários participantes, gravadas no Zoom
ou em qualquer outra fonte.

**Formatos aceitos**
Áudio: `.m4a` `.mp3` `.wav` `.wma` `.m4p` `.aac` `.ogg` `.flac`
Vídeo: `.mp4` `.mov` `.avi` `.mpeg`/`.mpg` `.wmv` `.mkv` `.3gp` `.webm`
(o ffmpeg extrai e normaliza o áudio de todos)

---

## Instalação (5 minutos)

Pré-requisitos: **Python 3.10+** e **ffmpeg** no PATH.

```bash
# Ubuntu (ffmpeg + bibliotecas que o WeasyPrint usa para gerar os PDFs)
sudo apt install ffmpeg libpango-1.0-0 libpangoft2-1.0-0 fonts-liberation

cd escriba
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edite o .env e cole sua ASSEMBLYAI_API_KEY (chave gratuita em assemblyai.com)

python run.py
# abra http://localhost:8000
```

No Windows/macOS o WeasyPrint precisa de passos extras de instalação (GTK/Pango);
siga o guia oficial: https://doc.courtbouillon.org/weasyprint/stable/first_steps.html
— ou rode o Escriba dentro do WSL/Ubuntu. Sem essas bibliotecas tudo funciona,
**menos os PDFs** (e o anexo em PDF do e-mail).

Para acessar de outro dispositivo da rede, troque o host no `run.py` para
`0.0.0.0` (o app **não tem autenticação**: quem alcançar a porta pode baixar
transcrições, ver a assinatura cadastrada e disparar e-mails com a sua conta —
use apenas em rede confiável ou atrás de um login/proxy).

---

## Os dois motores de transcrição

O app tem uma camada de engine plugável (`app/engines/`). Você escolhe por
upload, na própria interface.

### 1. AssemblyAI (padrão) — nuvem, sem GPU
- Diarização nativa e madura, português suportado, arquivos de até ~10 h.
- Custo na casa de **US$ 0,17/hora de áudio** (modelo Universal + speaker
  labels) — uma reunião de 2 h sai por ~US$ 0,35. Contas novas ganham
  **US$ 50 em créditos**, o que dá centenas de horas para validar.
  Valores mudam; confira `assemblyai.com/pricing`.
- É o caminho certo enquanto sua máquina não tem GPU dedicada: o i5 4ª geração
  só faz a conversão ffmpeg (rápida) e o resto acontece na nuvem.

### 2. WhisperX — 100% local, privado
Whisper `large-v3` + alinhamento + diarização pyannote. **Nada sai da sua
máquina** — indicado para gravações sensíveis (clientes corporativos, órgãos
públicos) que não podem ir para uma API nos EUA.

```bash
pip install -r requirements.txt -r requirements-local.txt   # instala PyTorch
```

Configuração extra no `.env`:
1. Crie um token em `huggingface.co/settings/tokens` → `HF_TOKEN=`
2. Aceite os termos (gratuito, 1 clique cada) nas páginas dos modelos
   **`pyannote/speaker-diarization-3.1`** e **`pyannote/segmentation-3.0`**
   no Hugging Face. Sem isso a diarização falha com erro 401 — é a pegadinha
   clássica do setup.

Desempenho esperado para 2 h de áudio com `large-v3`:

| Hardware                  | Tempo aproximado |
|---------------------------|------------------|
| RTX 3060 / 3060 Ti (8 GB+)| 10–25 min        |
| CPU (i5 4ª geração)       | inviável (10 h+) — use `WHISPER_MODEL=small` para testes curtos |

Ou seja: no seu hardware atual, use AssemblyAI no dia a dia e deixe o WhisperX
pronto para quando a GPU do upgrade chegar — nada no app muda, só o `.env`.

---

## Ata automática (opcional)

Marque **"Gerar ata da reunião (IA)"** no upload. O transcript vai para a API
da Anthropic e volta um `ata.md` com resumo executivo, tópicos, decisões,
tabela de ações (responsável/prazo) e pontos em aberto.

Requer `ANTHROPIC_API_KEY` no `.env`. Uma reunião de 2 h custa centavos de
dólar com o modelo padrão (`claude-sonnet-4-6`).

---

## Nomes, assinatura e e-mail

### 1. Nomes dos participantes e grafia correta

Depois que a transcrição termina, o botão **Revisar nomes** abre a tela de revisão:

1. **Participantes** — digite o nome de cada um, com a grafia certa, um por linha
   (também dá para informar já no upload: o AssemblyAI recebe os nomes como
   `word_boost` e tende a acertar a grafia; se a API recusar o parâmetro, o
   Escriba refaz sem ele).
2. **Quem é quem** — o motor separa as vozes como “Falante 1, 2…”. Diga quem é cada
   um (com ajuda de trechos de amostra); o nome substitui o rótulo no TXT, SRT,
   VTT, JSON, na timeline, na ata e nos PDFs.
3. **Grafia no texto** — o Escriba procura palavras parecidas com os nomes
   informados (ex.: “Rosilene” × “Rosilaine”) e **sugere** a troca; nada é trocado
   sem você marcar. Também dá para adicionar correções manuais (“Fleming” →
   “Fernando”). A troca é de palavra inteira.
4. **Dados da reunião** — título, data e local, que saem no cabeçalho dos PDFs.

> No motor WhisperX (local) os nomes não alimentam o reconhecimento; valem a
> tela de revisão, as sugestões de grafia e a ata.

A transcrição original (`segmentos.json`) **nunca é alterada**: nomes e
correções são aplicados por cima toda vez que os arquivos são gerados, então dá
para revisar de novo quantas vezes quiser. Se a ata já existia quando você mudou
nomes ou grafia, o Escriba oferece **refazer a ata**.

### 2. Assinatura do cliente (botão “Perfil e assinatura”)

Cadastre uma vez: nome, cargo, organização, e-mail e a **assinatura** — desenhada
na tela (mouse, caneta ou dedo) ou enviada como foto/imagem (PNG, JPG, WEBP até
5 MB; o fundo do papel é removido sozinho). Isso fica em `data/profile/`.

A partir daí, os PDFs da **ata** e da **transcrição**:
- levam seu nome/organização no cabeçalho;
- terminam com um bloco **Assinaturas**: você já assinado como responsável pela
  elaboração, e uma **linha em branco por participante** (com o nome impresso)
  para cada um assinar por conta própria — impresso, ou com a assinatura digital
  que preferirem.

> ⚠️ É a **imagem** da assinatura aplicada ao documento, não uma assinatura
> digital certificada (ICP-Brasil / gov.br). Serve para documentos internos e
> para colher as demais assinaturas. Se a ata precisar de registro em cartório,
> de certificação ou de força probatória específica, use a assinatura digital
> apropriada — confirme com o jurídico.

Os PDFs são gerados na hora e ficam em cache (refeitos só quando algo muda:
nomes, perfil, assinatura, ata…).

### 3. Envio por e-mail

Preencha no `.env` (qualquer provedor SMTP):

```ini
SMTP_HOST=smtp.seuprovedor.com
SMTP_PORT=587
SMTP_USER=voce@empresa.com
SMTP_PASSWORD=senha-de-app
SMTP_FROM=voce@empresa.com
SMTP_SECURITY=starttls       # starttls (587) | ssl (465) | none (só servidor local)
```

Gmail, Outlook e Proton exigem **senha de app** (ou token SMTP), não a senha
normal da conta. Reinicie o Escriba; o botão **Enviar por e-mail** de cada
transcrição abre o formulário: Para / Cc, assunto, mensagem (já preenchida) e
anexos (por padrão, os dois PDFs assinados; também TXT, SRT, VTT, JSON e ata em
Markdown). As respostas vão para o e-mail do seu perfil. Limites: 20 destinatários
e 20 MB de anexos por envio (`ESCRIBA_EMAIL_MAX_MB`). O Escriba recusa enviar
senha por conexão sem criptografia para servidores remotos. Cada envio fica
registrado no card da transcrição.

---

## Testes

```bash
python test_integracao.py   # fluxo original (sem chaves de API)
python test_v2.py           # nomes, correções, assinatura, PDFs, e-mail (SMTP local)
python test_ui.py           # opcional: tela num Chromium (pip install playwright && playwright install chromium)
```

Usam um diretório temporário e engine/ata simulados; não gastam créditos nem
tocam no seu `data/`.

---

## Dicas para gravações do Zoom

- **Gravação local**: o Zoom gera `audio_only.m4a` e `zoom_0.mp4` — os dois
  funcionam; prefira o `.m4a` (upload menor, mesmo áudio).
- **Gravação em nuvem**: baixe o "Audio only (M4A)".
- Em *Configurações → Gravação*, ativar **"Otimizar para editor de vídeo de
  terceiros"** e desativar músicas/efeitos melhora o áudio para ASR.
- Informar o nº de participantes no upload melhora a separação de vozes.
- `.m4p` com DRM (compras antigas do iTunes) não pode ser decodificado pelo
  ffmpeg — o app mostra erro claro nesse caso.

## Estrutura

```
escriba/
├── run.py                     # sobe o servidor (localhost:8000)
├── app/
│   ├── main.py                # API FastAPI + UI estática
│   ├── worker.py              # fila (1 job por vez) em thread; também refaz atas
│   ├── review.py              # nomes dos participantes, correções, sugestões
│   ├── profile.py             # perfil do cliente + imagem da assinatura
│   ├── pdfdocs.py             # PDF da ata e da transcrição (WeasyPrint)
│   ├── mailer.py              # envio por SMTP
│   ├── media.py               # ffmpeg: validação, conversão, duração
│   ├── jobs.py                # estado persistido em data/jobs/{id}/meta.json
│   ├── exports.py             # TXT / SRT / VTT / JSON + timeline da UI
│   ├── summary.py             # ata via API Anthropic (opcional)
│   └── engines/
│       ├── base.py            # contrato Segment + merge de falas
│       ├── assemblyai_engine.py
│       └── whisperx_engine.py
├── data/jobs/<id>/            # original + saídas de cada transcrição
│                              #   segmentos.json = transcrição crua (intocada)
└── data/profile/              # perfil.json + assinatura.png do cliente
```

Jobs sobrevivem a reinício do servidor: o que estava na fila é reenfileirado
no startup (uma ata interrompida no meio fica marcada com erro, para refazer). A API tem documentação interativa em `/api/docs`.
