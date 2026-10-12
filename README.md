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
- **Windows:** siga a seção [Windows](#windows) (Python 3.11 ou 3.12 e `build_windows.bat`).
- **Linux:** `python build.py` gera `dist/Escriba/Escriba`.
  O build recusa rodar com o Escriba aberto.

---

## Windows

**Requisitos**
- **Microsoft Edge WebView2 Runtime.** A janela do Escriba usa esse componente, que já
  vem na maioria dos Windows 10 e 11 atualizados. Sem ele, o pywebview cairia no IE11 e a
  tela quebraria sem aviso; por isso o Escriba confere antes de abrir a janela. Se faltar,
  mostra uma caixa com o link oficial do instalador
  (Evergreen Bootstrapper: `https://go.microsoft.com/fwlink/p/?LinkId=2124703`) e oferece
  abrir no navegador padrão, que continua offline.
- **.NET Framework 4.6.2 ou mais** (já vem no Windows 10 e 11).
- **Python 3.11 ou 3.12** para gerar o executável. O 3.13 ainda não foi verificado.

**Gerar o `Escriba.exe`** (num PC Windows, com internet; o PyInstaller não faz build cruzado)
1. Descompacte o `Escriba-windows-fonte.zip` numa pasta de trabalho. Ele tem o código,
   `requirements.txt`, `escriba.spec`, `escriba.ico`, `build_windows.bat`, `run.bat`, este
   README e a pasta `models/` (Whisper large-v3-turbo e o modelo de vozes).
2. Confira que existem `models\whisper\large-v3-turbo\model.bin` e
   `models\diarizacao\nemo_en_titanet_small.onnx`.
3. Feche o Escriba, se estiver aberto.
4. Dê dois cliques em `build_windows.bat` (ou rode-o no terminal). Ele:
   - cria `.venv-build` com Python 3.12 (ou 3.11, se for o que houver);
   - instala `requirements.txt` e o PyInstaller 6.22.3 (versões fixadas);
   - roda `build.py`, que copia os modelos para junto do programa, gera com o PyInstaller
     e monta o zip.
   Cada etapa confere o código de saída: se uma falhar, o script para com a mensagem.
   O build também recusa se o `Escriba.exe` estiver em execução.
5. Resultado: `dist\Escriba\Escriba.exe` e `dist\escriba-windows-amd64.zip`. Entregue a
   pasta `dist\Escriba` inteira, ou o zip.

Para validar o executável antes de entregar, rode uma transcrição completa pelo
`Escriba.exe` (não pelo código-fonte) e confira a transcrição, a ata e os dois PDFs.

**Rodar pelo código-fonte:** `run.bat`. Na primeira vez cria o `.venv`, mostra na tela os
erros do pip (sem `-q`) e baixa os modelos. O `pythonw` abre só a janela do programa.

**Onde ficam os dados**
- Programa empacotado (`Escriba.exe`): `%USERPROFILE%\Escriba` (ex.: `C:\Users\<você>\Escriba`).
  - `jobs\`: transcrições, atas e PDFs de cada reunião.
  - `profile\`: perfil e assinatura do cliente.
  - `nomes\nomes.json`: a lista de irmãos (nomes, observação para homônimos e grafias erradas).
  - `vozes\vozes.json`: vozes conhecidas (dado biométrico; apague pela tela "Vozes conhecidas").
  - `escriba.log`: registro de erros para o suporte.
- Pelo código-fonte (`run.bat`): a pasta `data\` do projeto, ou `ESCRIBA_DATA_DIR` no `.env`.

**Ata sem chave de API**
Sem `ANTHROPIC_API_KEY` no `.env`, a ata é gerada localmente (`app/ata_local.py`), sem
internet e sem IA. Ela traz data e duração, presidente, orador e tema, dirigente e leitor,
cânticos com número, comentaristas com a contagem e os participantes informados. Não
resume o conteúdo dos comentários. Os nomes são sugestões do motor e a revisão é que
confirma. Com a chave, a ata é feita pela IA.

**PDF no Windows (fpdf2)**
No Linux, os PDFs da ata e da transcrição saem pelo WeasyPrint. No Windows ele precisa de
Pango/GTK, que o programa não traz. Quando o WeasyPrint não carrega, `app/pdfdocs.py` gera o
mesmo conteúdo com o **fpdf2** (Python puro), fixado em `requirements.txt`, com a fonte
DejaVu Sans embutida (`app/fonts/`, licença Bitstream Vera, ver `LICENCA-DejaVu.txt`). O
layout é mais simples: cabeçalho com título, data e duração; participantes; falas com o
nome em negrito; ata em tópicos e tabelas; bloco de assinaturas. A hora de cada fala só
aparece quando o PDF é pedido com `show_times=True`, como no PDF do Linux.
Se o PDF estiver aberto no leitor, o Escriba avisa e pede para fechá-lo (HTTP 409), em vez
de dar erro interno.

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
| `speakers_expected` / `speaker_options` | mín./máx. de falantes no envio (estimativa 1–12) |
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

**Trechos pulados ("buracos").** Quando o áudio fica baixo (comentário longe do
microfone), o Whisper às vezes não devolve nenhuma palavra num trecho em que há
fala. O Escriba procura intervalos sem palavra (fora do cântico) em que o VAD,
com limiar mais baixo, acha pelo menos 1 s de fala; aplica ganho (até 18 dB, sem
estourar) e transcreve só esse trecho com os mesmos parâmetros, descartando
frases inventadas e trechos de baixa confiança (`app/buracos.py`). Isso custa
CPU (medido: +10,6% no salão, +28,5% numa conversa de 2 pessoas). Os buracos com
mais fala vão primeiro, e há um teto: `ESCRIBA_BURACOS_TETO_CPU` (segundos de
CPU por segundo de áudio, padrão 0,75). `ESCRIBA_BURACOS=0` desliga. O
progresso aparece na tela ("Recuperando trechos pulados — 3/12").

Não usa PyTorch nem token do Hugging Face. O executável fica leve e roda em
qualquer PC. Na CPU (i5 de 4 núcleos), leva aproximadamente o tempo do próprio
áudio. Para máquinas fracas, use `WHISPER_MODEL=small` no `.env` (bem mais
rápido, porém erra mais).

### Reconhecimento de vozes e nomes (motor Local)
A transcrição acontece em três fases:

1. **Reconhecer vozes**, antes de transcrever e só pelo som (~20 s por
   reunião). Conta quantas pessoas falam (1, 2…) e compara cada voz com as
   **vozes conhecidas**.
2. **Transcrever**, já separando as falas no número de vozes da fase 1.
3. **Dar nome às vozes**, nesta ordem:
   - voz reconhecida pela memória ("Fernando, 91% parecida");
   - apresentação na conversa ("aqui é o Fernando", "Eduardo falando");
   - quem é chamado pelo nome ("Então fechou, Fernando." indica que o *outro*
     é o Fernando). Fala citada ("ele disse: …, Fernando") não conta.
   Grafia parecida com um participante informado vale por ele: "Rosilene"
   na fala vira "Rosilaine Andrade".

**Reunião com estudo (A Sentinela).** Quando a reunião tem a estrutura de um
estudo (um dirigente com pelo menos 8 chamadas e 2 anúncios de parágrafo ou
leitura), a fala depois de cada chamada ("Irmão Paulo, por favor") é de quem
foi chamado, até o dirigente voltar, mesmo que a voz se pareça com a de outra
pessoa no mesmo microfone (`app/estudo.py`). Nome só sai com evidência: trecho
sem chamada nem agradecimento sai **"[?]"**; nome fora da lista de irmãos e dos
participantes precisa de chamada **e** agradecimento, senão sai "[?] Nome?";
homônimo da lista sai "[?] Nome (homônimo)". Uma conversa comum não tem essa
estrutura e fica como está. "[?]" nunca vira nome nem voz aprendida: ouça o
trecho e digite o nome na revisão. Grafia que o Whisper erra com frequência
("Marilês" para Marines) deve ser cadastrada na lista de irmãos.

Os nomes saem **sugeridos** na tela "Revisar nomes". Ao salvar a revisão,
o Escriba **aprende a voz** de cada pessoa confirmada (`app/vozes.py`). Na
próxima reunião com essa pessoa, ela é reconhecida já na fase 1. Cada reunião
confirmada torna o reconhecimento mais estável. Só aprende o que foi
confirmado; um palpite errado nunca vira memória.

A tela **Vozes conhecidas** lista quem o Escriba reconhece e permite apagar.
As assinaturas ficam em `<dados>/vozes/vozes.json`, só neste computador.
> ⚠️ Assinatura de voz é dado biométrico (dado pessoal sensível na LGPD).
> Grave e guarde vozes apenas com o consentimento dos participantes.

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
python -m pytest test_diarize.py test_vozes_nomes.py   # diarização, vozes e nomes
python -m pytest tests/     # ata local, irmãos, PDF de reserva (fpdf2), salvar, janela
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
│   ├── vozes.py               # memória de vozes (aprende a cada revisão)
│   ├── nomes.py               # descobre nomes pela conversa
│   ├── profile.py             # perfil do cliente + imagem da assinatura
│   ├── pdfdocs.py             # PDF da ata e da transcrição (WeasyPrint; fpdf2 no Windows)
│   ├── arquivo_seguro.py      # gravação com nova tentativa (arquivo aberto no Windows)
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
