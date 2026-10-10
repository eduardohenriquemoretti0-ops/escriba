# Escriba — transcrição de reuniões, offline

Programa de desktop no estilo Otter.ai: recebe áudio ou vídeo, transcreve com
identificação de falantes (diarização) e exporta **TXT, SRT, VTT, JSON e PDF**,
com **ata automática** opcional gerada por IA.

**Novidades da v3**
1. **Funciona sem internet.** O motor padrão (**Local**) transcreve e separa os
   falantes no próprio computador. Se a conexão do cliente cair no meio do
   trabalho, nada muda.
2. **É um programa, não um site.** Abre numa janela própria (`Escriba.exe` no
   Windows, `run.sh`/`run.bat` pelo código-fonte). Não abre navegador e não
   fica exposto na rede.
3. **Não precisa mais instalar o ffmpeg.** A conversão de mídia usa o FFmpeg
   que já vem embutido no pacote Python (PyAV).

Da v2 continuam: nomes dos participantes, envio por e-mail e assinatura
(ver [Nomes, assinatura e e-mail](#nomes-assinatura-e-e-mail)).

**Formatos aceitos**
Áudio: `.m4a` `.mp3` `.wav` `.wma` `.m4p` `.aac` `.ogg` `.flac`
Vídeo: `.mp4` `.mov` `.avi` `.mpeg`/`.mpg` `.wmv` `.mkv` `.3gp` `.webm`

---

## Como usar

### Programa pronto (para o cliente)
Descompacte o `Escriba-windows-amd64.zip` e dê dois cliques em `Escriba.exe`.
Os modelos já vêm na pasta `models/`, então não é preciso instalar nada nem
ter internet. As transcrições ficam em `C:\Users\<usuário>\Escriba`.

### Pelo código-fonte (desenvolvimento)
```bash
./run.sh        # Linux
run.bat         # Windows
```
Na primeira vez, o script cria o `.venv`, instala as dependências e baixa os
modelos (~1,7 GB, precisa de internet só nessa hora). Depois disso, abre direto.

### Gerar o executável
O PyInstaller só gera executável para o sistema onde roda:
- **Windows:** numa máquina Windows 10/11 com Python 3.11+, rode `build_windows.bat`.
  Sai `dist\Escriba\Escriba.exe` e `dist\Escriba-windows-amd64.zip`.
- **Linux:** `python build.py` gera `dist/Escriba/Escriba`.

O modo servidor antigo (navegador em `localhost:8000`) continua disponível em
`run_servidor.py`, para quem quiser usar o Escriba pela rede.

---

## Os motores de transcrição

### 1. Local (padrão): offline, sem GPU
O pipeline imita o do AssemblyAI. Cada recurso tem um equivalente local:

| AssemblyAI | Escriba Local |
|---|---|
| modelo Universal | Whisper **large-v3-turbo** (int8, CPU) |
| `punctuate` / `format_text` | pontuação e maiúsculas nativas do Whisper |
| `keyterms_prompt` / `word_boost` | `hotwords` do Whisper com os participantes + `ESCRIBA_TERMOS` |
| `speaker_labels` | embedding de voz por frase (NeMo TitaNet, ONNX) + agrupamento espectral |
| `speakers_expected` / `speaker_options` | mín./máx. de falantes no envio (padrão 1–10) |
| `utterances` | frases do mesmo falante unidas em blocos |
| filtro de ruído/silêncio | VAD Silero e limiares anti-alucinação do Whisper |

Como funciona a separação de falantes (`app/engines/diarize.py`):
1. O Whisper marca o tempo de cada palavra.
2. As palavras viram frases, cortadas na pontuação ou numa pausa. A troca de
   falante quase sempre cai numa dessas fronteiras.
3. Cada frase de 1 s ou mais vira um vetor de timbre de voz.
4. Um agrupamento espectral estima quantas vozes existem pelo maior salto entre
   autovalores (eigengap), sempre dentro do mínimo/máximo informado.
5. Frases curtas ("Sim.", "Tá.") herdam a voz mais parecida. Em caso de
   empate, ficam com o falante anterior.

Não usa PyTorch nem token do Hugging Face. O executável fica leve e roda em
qualquer PC. Na CPU (i5 de 4 núcleos), leva aproximadamente o tempo do próprio
áudio. Para máquinas fracas, use `WHISPER_MODEL=small` no `.env` (bem mais
rápido, porém erra mais).

### 2. AssemblyAI: nuvem
Precisa de internet e de `ASSEMBLYAI_API_KEY`. Custa cerca de US$ 0,15–0,21
por hora de áudio, conforme o modelo (confira em `assemblyai.com/pricing`).
Use quando quiser comparar resultados ou quando o computador for muito lento.

### 3. WhisperX: local com GPU NVIDIA (avançado)
Whisper `large-v3` + pyannote via PyTorch. Exige GPU com 8 GB ou mais,
`pip install -r requirements-local.txt`, `HF_TOKEN` e o aceite dos termos dos
modelos `pyannote/speaker-diarization-3.1` e `pyannote/segmentation-3.0` no
Hugging Face. Não entra no executável.

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
python -m pytest test_diarize.py   # diarização local (sem modelos)
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
├── escriba_app.py             # programa de desktop (janela própria)
├── run.sh / run.bat           # abre o programa pelo código-fonte
├── build.py / escriba.spec    # gera o executável (PyInstaller)
├── baixar_modelos.py          # baixa os modelos para models/
├── run_servidor.py            # modo servidor antigo (localhost:8000)
├── app/
│   ├── main.py                # API FastAPI + UI estática
│   ├── worker.py              # fila (1 job por vez) em thread; também refaz atas
│   ├── review.py              # nomes dos participantes, correções, sugestões
│   ├── profile.py             # perfil do cliente + imagem da assinatura
│   ├── pdfdocs.py             # PDF da ata e da transcrição (WeasyPrint)
│   ├── mailer.py              # envio por SMTP
│   ├── media.py               # PyAV: validação, conversão, duração
│   ├── jobs.py                # estado persistido em data/jobs/{id}/meta.json
│   ├── exports.py             # TXT / SRT / VTT / JSON + timeline da UI
│   ├── summary.py             # ata via API Anthropic (opcional)
│   └── engines/
│       ├── base.py            # contrato Segment + merge de falas
│       ├── local_engine.py    # motor offline (padrão)
│       ├── diarize.py         # separação de falantes local
│       ├── assemblyai_engine.py
│       └── whisperx_engine.py
├── data/jobs/<id>/            # original + saídas de cada transcrição
│                              #   segmentos.json = transcrição crua (intocada)
└── data/profile/              # perfil.json + assinatura.png do cliente
```

Jobs sobrevivem a reinício do servidor: o que estava na fila é reenfileirado
no startup (uma ata interrompida no meio fica marcada com erro, para refazer). A API tem documentação interativa em `/api/docs`.
