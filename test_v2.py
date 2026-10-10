"""Teste dos recursos da v2: nomes dos participantes, correção de grafia,
assinatura do cliente, PDFs, e-mail (servidor SMTP de verdade, local).

Sem chaves de API: o engine e a geração da ata são substituídos por fakes.
Os PDFs de verdade são gerados e inspecionados (texto, imagem embutida).
"""
import email
import io
import json
import os
import re
import socket
import subprocess
import tempfile
import time
from email import policy
from pathlib import Path
from unittest.mock import patch

_TMP = tempfile.mkdtemp(prefix="escriba_test2_")
os.environ["ESCRIBA_DATA_DIR"] = _TMP


def _porta_livre() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


SMTP_PORT = _porta_livre()
os.environ.update({
    "SMTP_HOST": "127.0.0.1", "SMTP_PORT": str(SMTP_PORT),
    "SMTP_FROM": "escriba@teste.local", "SMTP_SECURITY": "none",
    "SMTP_USER": "", "SMTP_PASSWORD": "",
    "ANTHROPIC_API_KEY": "chave-falsa-de-teste",
})

from aiosmtpd.controller import Controller  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from app import config, jobs, mailer, profile, review  # noqa: E402
from app.engines.base import Segment  # noqa: E402
from app.main import app  # noqa: E402

BASE = Path(__file__).parent
OUT = Path(os.environ.get("ESCRIBA_TEST_OUT", _TMP))
OUT.mkdir(parents=True, exist_ok=True)

FAKE_SEGMENTS = [
    Segment(0.0, 6.0, "A", "Bom dia, aqui é a Rosilene. Vamos começar a reunião de hoje."),
    Segment(6.5, 12.0, "B", "Bom dia, Rosilene. Eduardo falando, tenho o orçamento da Acme & Cia pronto."),
    Segment(12.5, 20.0, "A", "Ótimo. O Eduardo fica responsável pelo orçamento até sexta-feira."),
    Segment(20.5, 25.0, "B", "Combinado. Uso a tag <b>negrito</b> só para testar o escape."),
]
VOCAB_VISTO: list = []

ATA_FAKE = """# Ata da reunião

## Resumo executivo
Rosilaine Andrade abriu a reunião e Eduardo Moretti apresentou o orçamento da Acme & Cia.

## Principais tópicos
- Orçamento da Acme & Cia
- Prazos

## Decisões tomadas
Nenhuma decisão explícita registrada

## Ações e pendências
| Ação | Responsável | Prazo |
|------|-------------|-------|
| Entregar o orçamento | Eduardo Moretti | sexta-feira |

## Pontos em aberto
Nenhum.
"""
PROMPT_VISTO: list = []


def fake_transcribe(self, audio_path, *, language, min_speakers,
                    max_speakers, progress, vocabulary=None):
    VOCAB_VISTO.append(vocabulary)
    progress("Transcrevendo (fake)")
    return list(FAKE_SEGMENTS)


def fake_ata(transcript_txt, participants=None):
    PROMPT_VISTO.append((transcript_txt, participants))
    return ATA_FAKE


class Caixa:
    def __init__(self):
        self.msgs = []

    async def handle_DATA(self, server, session, envelope):
        self.msgs.append((envelope.mail_from, list(envelope.rcpt_tos),
                          email.message_from_bytes(envelope.content, policy=policy.default)))
        return "250 OK"


def esperar(client, job_id, campo, valores, tentativas=80):
    job = {}
    for _ in range(tentativas):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job[campo] in valores:
            return job
        time.sleep(0.25)
    raise AssertionError(f"timeout esperando {campo} em {valores}: {job}")


def png_assinatura_desenhada() -> bytes:
    """Traços sobre fundo transparente (o que o canvas da tela envia)."""
    img = Image.new("RGBA", (600, 200), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.line([(20, 150), (90, 40), (130, 160), (190, 50), (260, 120), (330, 60)],
           fill=(16, 24, 40, 255), width=6)
    d.line([(20, 175), (330, 175)], fill=(16, 24, 40, 255), width=3)
    buf = io.BytesIO(); img.save(buf, "PNG")
    return buf.getvalue()


def foto_assinatura() -> bytes:
    """'Foto' de uma assinatura em papel: fundo claro com ruído, traço azul."""
    img = Image.new("RGB", (900, 400), (236, 233, 224))
    d = ImageDraw.Draw(img)
    for x in range(0, 900, 7):                       # sombra/ruído
        d.line([(x, 0), (x, 400)], fill=(230 + (x % 5), 227, 218), width=2)
    d.line([(100, 300), (200, 120), (260, 310), (360, 110), (450, 260), (600, 140)],
           fill=(25, 40, 120), width=9)
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def pdf_texto(path: Path) -> str:
    return subprocess.run(["pdftotext", "-layout", str(path), "-"],
                          check=True, capture_output=True, text=True).stdout


def norm(texto: str) -> str:
    """Sem espaços e em minúsculas: títulos em caixa alta/espaçados do PDF
    ('P A R T I C I P A N T E S') continuam comparáveis."""
    return re.sub(r"\s+", "", texto).casefold()


def pdf_imagens(path: Path) -> int:
    out = subprocess.run(["pdfimages", "-list", str(path)],
                         check=True, capture_output=True, text=True).stdout
    # a máscara de transparência aparece como 'smask'; conta só as imagens
    return sum(1 for ln in out.splitlines()[2:] if ln.split()[2:3] == ["image"])


def testar_engine_assemblyai() -> None:
    """word_boost: enviado quando há nomes; refeito sem ele se a API recusar."""
    import sys
    import types

    chamadas: list[dict] = []
    cenario = {"modo": "ok"}

    class Status:
        error, completed = "error", "completed"

    class Utt:
        start, end, speaker, text = 0, 1500, "A", "Olá, Rosilaine."

    class Transcript:
        def __init__(self, ok):
            self.status = Status.completed if ok else Status.error
            self.error = None if ok else "word_boost nao suportado"
            self.utterances = [Utt()] if ok else []
            self.text = "Olá, Rosilaine."

    class Cfg:
        def __init__(self, **kw):
            if cenario["modo"] == "sdk_antigo" and "word_boost" in kw:
                raise TypeError("unexpected keyword 'word_boost'")
            self.kw = kw

    class Transcriber:
        def __init__(self, config):
            self.cfg = config

        def transcribe(self, path):
            chamadas.append(self.cfg.kw)
            recusa = cenario["modo"] == "api_recusa" and "word_boost" in self.cfg.kw
            return Transcript(ok=not recusa)

    fake = types.SimpleNamespace(
        settings=types.SimpleNamespace(api_key=""), TranscriptStatus=Status,
        TranscriptionConfig=Cfg, Transcriber=Transcriber)
    from app.engines.assemblyai_engine import AssemblyAIEngine, _boost_terms
    assert _boost_terms(["Rosilaine Andrade", "Ed", "Eduardo Moretti", "rosilaine"]) == \
        ["Rosilaine Andrade", "Rosilaine", "Andrade", "Eduardo Moretti", "Eduardo", "Moretti", "rosilaine"]
    avisos: list[str] = []
    with patch.dict(sys.modules, {"assemblyai": fake}), patch("app.config.ASSEMBLYAI_API_KEY", "k"):
        for modo, esperado in (("ok", [True]), ("api_recusa", [True, False]), ("sdk_antigo", [False])):
            chamadas.clear(); cenario["modo"] = modo
            segs = AssemblyAIEngine().transcribe(Path("x.mp3"), language="pt", min_speakers=None,
                                                 max_speakers=None, progress=avisos.append,
                                                 vocabulary=["Rosilaine Andrade"])
            assert [("word_boost" in c) for c in chamadas] == esperado, (modo, chamadas)
            assert segs and segs[0].text == "Olá, Rosilaine.", modo
            if esperado[0]:
                assert chamadas[0]["boost_param"] == "high" and "Rosilaine" in chamadas[0]["word_boost"]
        chamadas.clear(); cenario["modo"] = "ok"          # sem nomes: nada de boost
        AssemblyAIEngine().transcribe(Path("x.mp3"), language="pt", min_speakers=None,
                                      max_speakers=None, progress=avisos.append)
        assert "word_boost" not in chamadas[0]


def main() -> None:
    testar_engine_assemblyai()
    wav = Path(_TMP) / "_teste.wav"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=25",
                    "-ac", "1", "-ar", "16000", str(wav)], check=True, capture_output=True)

    caixa = Caixa()
    smtp = Controller(caixa, hostname="127.0.0.1", port=SMTP_PORT)
    smtp.start()
    try:
        with patch("app.engines.assemblyai_engine.AssemblyAIEngine.transcribe", fake_transcribe), \
             patch("app.summary.generate_ata", fake_ata), \
             TestClient(app) as client:

            # ---- config / perfil vazio --------------------------------------
            cfg = client.get("/api/config").json()
            assert cfg == {"ata_disponivel": True, "email_configurado": True}, cfg
            p0 = client.get("/api/profile").json()
            assert p0["tem_assinatura"] is False and p0["nome"] == ""
            assert client.get("/api/profile/signature.png").status_code == 404

            # ---- upload com nomes + ata -------------------------------------
            with wav.open("rb") as f:
                r = client.post("/api/jobs", files={"file": ("reuniao_acme.wav", f, "audio/wav")},
                                data={"engine": "assemblyai", "language": "pt",
                                      "want_summary": "true",
                                      "participants": "Rosilaine Andrade\nEduardo Moretti, rosilaine andrade"})
            assert r.status_code == 200, r.text
            jid = r.json()["id"]
            job = esperar(client, jid, "status", ("concluido", "erro"))
            assert job["status"] == "concluido", job
            assert VOCAB_VISTO[-1] == ["Rosilaine Andrade", "Eduardo Moretti"], VOCAB_VISTO
            assert job["participants"] == ["Rosilaine Andrade", "Eduardo Moretti"]
            assert job["ata_status"] == "pronta", job
            assert set(job["outputs"]) >= {"txt", "srt", "vtt", "json", "ata",
                                           "ata_pdf", "transcricao_pdf"}, job["outputs"]
            assert job["meeting"]["title"] == "reuniao_acme"
            assert PROMPT_VISTO[-1][1] == ["Rosilaine Andrade", "Eduardo Moretti"]

            # ---- tela de revisão --------------------------------------------
            rv = client.get(f"/api/jobs/{jid}/review").json()
            assert [s["label"] for s in rv["speakers"]] == ["Falante 1", "Falante 2"]
            sug = {(s["de"], s["para"]) for s in rv["suggestions"]}
            assert ("Rosilene", "Rosilaine") in sug, rv["suggestions"]
            assert not any(s["de"] == "Eduardo" for s in rv["suggestions"])

            # nomes inválidos/injetados são ignorados ou limpos
            rv = client.put(f"/api/jobs/{jid}/review", json={
                "speaker_names": {"Falante 1": "  Rosilaine\x00 Andrade ", "Falante 9": "Fantasma", "Falante 2": ""},
                "corrections": [{"de": "Rosilene", "para": "Rosilaine"}, {"de": "", "para": "x"}, {"de": "a", "para": "a"}],
                "meeting": {"title": "Reunião Acme", "date": "08/10/2026", "place": "Jundiaí"},
            }).json()
            assert rv["speaker_names"] == {"Falante 1": "Rosilaine Andrade"}, rv["speaker_names"]
            assert rv["corrections"] == [{"de": "Rosilene", "para": "Rosilaine"}]
            assert rv["ata_desatualizada"] is True
            assert not any(s["de"] == "Rosilene" for s in rv["suggestions"])

            txt = client.get(f"/api/jobs/{jid}/file/txt").text
            assert "Rosilaine Andrade · 00:00:00" in txt and "Falante 2 · 00:00:06" in txt
            assert "Rosilene" not in txt and "Bom dia, Rosilaine." in txt
            raw = json.loads((config.JOBS_DIR / jid / "segmentos.json").read_text())
            assert "Rosilene" in raw[0]["text"], "o cru nunca pode ser alterado"
            vtt = client.get(f"/api/jobs/{jid}/file/vtt").text
            assert "<v Rosilaine Andrade>" in vtt

            # idempotência: salvar de novo não aplica a correção duas vezes
            client.put(f"/api/jobs/{jid}/review", json={"corrections": [{"de": "Eduardo", "para": "Eduardo Moretti"}]})
            client.put(f"/api/jobs/{jid}/review", json={"corrections": [{"de": "Eduardo", "para": "Eduardo Moretti"}]})
            txt = client.get(f"/api/jobs/{jid}/file/txt").text
            assert "Eduardo Moretti Moretti" not in txt and "Eduardo Moretti falando" in txt
            client.put(f"/api/jobs/{jid}/review", json={
                "corrections": [{"de": "Rosilene", "para": "Rosilaine"}],
                "speaker_names": {"Falante 1": "Rosilaine Andrade", "Falante 2": "Eduardo Moretti"}})
            tl = client.get(f"/api/jobs/{jid}").json()
            assert tl["speakers"] == ["Rosilaine Andrade", "Eduardo Moretti"]

            # ---- PDFs sem perfil: saem, sem assinatura ------------------------
            r = client.get(f"/api/jobs/{jid}/file/ata_pdf")
            assert r.status_code == 200 and r.content[:4] == b"%PDF", r.text[:200]
            ata_pdf_path = config.JOBS_DIR / jid / "ata.pdf"
            assert pdf_imagens(ata_pdf_path) == 0

            # ---- perfil + assinatura ------------------------------------------
            bad = client.put("/api/profile", json={"nome": "X", "email": "isso nao e email"})
            assert bad.status_code == 422
            p = client.put("/api/profile", json={
                "nome": "Eduardo Moretti", "cargo": "Diretor de Tecnologia",
                "organizacao": "Moretti Informática", "email": "responsavel@exemplo.com"}).json()
            assert p["nome"] == "Eduardo Moretti" and p["tem_assinatura"] is False

            assert client.post("/api/profile/signature", files={"file": ("a.txt", b"nao sou imagem", "text/plain")}).status_code == 422
            assert client.post("/api/profile/signature", files={"file": ("a.png", b"", "image/png")}).status_code == 422
            vazia = io.BytesIO(); Image.new("RGBA", (100, 50), (0, 0, 0, 0)).save(vazia, "PNG")
            assert client.post("/api/profile/signature", files={"file": ("v.png", vazia.getvalue(), "image/png")}).status_code == 422
            grande = b"\x89PNG" + b"0" * (config.MAX_SIGNATURE_BYTES + 10)
            assert client.post("/api/profile/signature", files={"file": ("g.png", grande, "image/png")}).status_code == 422

            # 1) foto em papel -> fundo transparente
            p = client.post("/api/profile/signature", files={"file": ("foto.jpg", foto_assinatura(), "image/jpeg")}).json()
            assert p["tem_assinatura"] is True
            im = Image.open(io.BytesIO(client.get("/api/profile/signature.png").content))
            assert im.mode == "RGBA" and max(im.size) <= 700
            alpha = im.getchannel("A")
            assert alpha.getextrema()[0] == 0, "fundo da foto deveria ficar transparente"
            cobertura = sum(1 for v in alpha.tobytes() if v > 128) / (im.width * im.height)
            assert 0.01 < cobertura < 0.5, f"traço fora do esperado: {cobertura:.3f}"
            im.convert("RGBA").save(OUT / "assinatura_da_foto.png")

            # 2) desenho do canvas (já transparente)
            client.post("/api/profile/signature", files={"file": ("sig.png", png_assinatura_desenhada(), "image/png")})
            im2 = Image.open(io.BytesIO(client.get("/api/profile/signature.png").content))
            assert im2.mode == "RGBA" and im2.width < 600, "deveria recortar no traço"

            # ---- PDFs com assinatura (+ cache) ----------------------------------
            t0 = time.time()
            r = client.get(f"/api/jobs/{jid}/file/ata_pdf")
            assert r.status_code == 200, r.text
            assert ata_pdf_path.exists() and ata_pdf_path.stat().st_mtime >= t0 - 1, "perfil novo deve invalidar o cache"
            ata_txt = pdf_texto(ata_pdf_path)
            for trecho in ("Ata da reunião", "Reunião Acme", "08/10/2026", "Jundiaí", "Moretti Informática",
                           "Rosilaine Andrade", "Eduardo Moretti", "Diretor de Tecnologia",
                           "Responsável pela elaboração", "Acme & Cia", "sexta-feira", "Assinaturas", "Participantes"):
                assert norm(trecho) in norm(ata_txt), f"faltou '{trecho}' na ata:\n{ata_txt}"
            assert pdf_imagens(ata_pdf_path) == 1, "a assinatura do cliente deve estar embutida"

            mt = ata_pdf_path.stat().st_mtime
            client.get(f"/api/jobs/{jid}/file/ata_pdf")
            assert ata_pdf_path.stat().st_mtime == mt, "segunda chamada deveria vir do cache"

            r = client.get(f"/api/jobs/{jid}/file/transcricao_pdf")
            assert r.status_code == 200 and r.content[:4] == b"%PDF"
            tr_path = config.JOBS_DIR / jid / "transcricao.pdf"
            tr_txt = pdf_texto(tr_path)
            for trecho in ("Transcrição da reunião", "Rosilaine Andrade", "Bom dia, Rosilaine.",
                           "Acme & Cia", "<b>negrito</b>", "Responsável pela elaboração", "Participantes"):
                assert norm(trecho) in norm(tr_txt), f"faltou '{trecho}' na transcrição:\n{tr_txt}"
            assert "Rosilene" not in tr_txt
            assert pdf_imagens(tr_path) == 1

            # troca de nome muda o PDF (cache invalidado)
            mt = tr_path.stat().st_mtime
            time.sleep(0.05)
            client.put(f"/api/jobs/{jid}/review", json={"meeting": {"title": "Reunião Acme", "date": "09/10/2026", "place": "Jundiaí"}})
            client.get(f"/api/jobs/{jid}/file/transcricao_pdf")
            assert tr_path.stat().st_mtime > mt and "09/10/2026" in pdf_texto(tr_path)

            # PDFs de amostra para inspeção visual
            for nome in ("ata.pdf", "transcricao.pdf"):
                (OUT / f"amostra_{nome}").write_bytes((config.JOBS_DIR / jid / nome).read_bytes())

            # ---- sem participantes cadastrados: linhas em branco ---------------
            with wav.open("rb") as f:
                r = client.post("/api/jobs", files={"file": ("sem_nomes.wav", f, "audio/wav")}, data={"engine": "assemblyai", "language": "pt"})
            jid2 = r.json()["id"]
            esperar(client, jid2, "status", ("concluido", "erro"))
            assert client.get(f"/api/jobs/{jid2}/file/ata_pdf").status_code == 404, "sem ata -> 404"
            assert client.get(f"/api/jobs/{jid2}/file/transcricao_pdf").status_code == 200
            sem = pdf_texto(config.JOBS_DIR / jid2 / "transcricao.pdf")
            assert "Falante 1" in sem and sem.count("Nome") >= 6, "6 linhas em branco quando não há nomes"
            # gerar ata depois (botão "Gerar ata")
            assert client.post(f"/api/jobs/{jid2}/ata").status_code == 200
            j2 = esperar(client, jid2, "ata_status", ("pronta", "erro"))
            assert j2["ata_status"] == "pronta" and "ata_pdf" in j2["outputs"], j2
            with patch("app.config.ANTHROPIC_API_KEY", ""):
                assert client.post(f"/api/jobs/{jid2}/ata").status_code == 400

            # ---- e-mail ----------------------------------------------------------
            st = client.get("/api/email/status").json()
            assert st == {"configurado": True, "de": "escriba@teste.local"}

            def enviar(**kw):
                body = {"para": "cliente@exemplo.com", "anexos": ["ata_pdf", "transcricao_pdf"], **kw}
                return client.post(f"/api/jobs/{jid}/email", json=body)

            assert enviar(para="").status_code == 422
            assert enviar(para="nao-e-email").status_code == 422
            assert enviar(cc="ok@x.com, lixo").status_code == 422
            assert enviar(anexos=[]).status_code == 422
            assert enviar(anexos=["../../etc/passwd"]).status_code == 422
            assert enviar(para=",".join(f"p{i}@x.com" for i in range(25))).status_code == 400
            assert not caixa.msgs

            r = enviar(para="cliente@exemplo.com; outro@exemplo.com", cc="copia@exemplo.com",
                       assunto="Ata da reunião\r\nBcc: invasor@mal.com", mensagem="")
            assert r.status_code == 200, r.text
            assert len(caixa.msgs) == 1
            de, rcpts, msg = caixa.msgs[0]
            assert de == "escriba@teste.local"
            assert sorted(rcpts) == ["cliente@exemplo.com", "copia@exemplo.com", "outro@exemplo.com"]
            assert "invasor@mal.com" not in rcpts and msg["Bcc"] is None, "injeção de cabeçalho"
            assert "Ata da reunião" in msg["Subject"] and "\n" not in msg["Subject"]
            assert msg["Reply-To"] == "responsavel@exemplo.com"
            assert "Eduardo Moretti" in msg["From"]
            nomes = sorted(a.get_filename() for a in msg.iter_attachments())
            assert nomes == ["reuniao_acme_ata.pdf", "reuniao_acme_transcricao.pdf"], nomes
            for a in msg.iter_attachments():
                assert a.get_content_type() == "application/pdf" and a.get_content()[:4] == b"%PDF"
            corpo = msg.get_body(preferencelist=("plain",)).get_content()
            assert "Reunião Acme" in corpo and "Eduardo Moretti" in corpo, corpo
            log = client.get(f"/api/jobs/{jid}").json()["email_log"]
            assert len(log) == 1 and log[0]["para"] == ["cliente@exemplo.com", "outro@exemplo.com"]

            # outros formatos de anexo + limite de tamanho
            assert enviar(anexos=["txt", "srt"]).status_code == 200
            assert sorted(a.get_filename() for a in caixa.msgs[-1][2].iter_attachments()) == ["reuniao_acme.srt", "reuniao_acme.txt"]
            with patch("app.config.EMAIL_MAX_ATTACH_MB", 0):
                r = enviar()
                assert r.status_code == 400 and "limite" in r.json()["detail"], r.text

            # servidor SMTP fora do ar -> erro legível, sem estourar 500
            with patch("app.config.SMTP_PORT", _porta_livre()):
                r = enviar()
                assert r.status_code == 400 and "enviar" in r.json()["detail"].lower(), r.text
            # e-mail não configurado
            with patch("app.config.SMTP_HOST", ""):
                assert client.get("/api/email/status").json()["configurado"] is False
                r = enviar()
                assert r.status_code == 400 and ".env" in r.json()["detail"]
            # senha em servidor remoto sem criptografia é recusada
            with patch("app.config.SMTP_HOST", "smtp.exemplo.com"), patch("app.config.SMTP_PASSWORD", "segredo"):
                try:
                    mailer.send(to=["a@b.com"], cc=[], subject="x", body="y", attachments=[])
                    raise AssertionError("deveria recusar senha sem TLS")
                except mailer.MailError as exc:
                    assert "sem criptografia" in str(exc)

            # ---- robustez ----------------------------------------------------------
            assert client.get("/api/jobs/..").status_code in (404, 405)
            assert client.get("/api/jobs/zzzzzzzzzzzz/review").status_code == 404
            fantasma = jobs.new_job(filename="x.wav", engine="assemblyai", language="pt",
                                    min_speakers=None, max_speakers=None, want_summary=False)
            assert client.get(f"/api/jobs/{fantasma.id}/review").status_code == 409, "revisão só depois de concluir"
            assert client.get(f"/api/jobs/{jid}/file/ata_pdf;rm").status_code == 404
            assert client.get(f"/api/jobs/{jid}/file/xyz").status_code == 404

            # remoção da assinatura -> PDF sem imagem
            assert client.delete("/api/profile/signature").json()["tem_assinatura"] is False
            client.get(f"/api/jobs/{jid}/file/ata_pdf")
            assert pdf_imagens(config.JOBS_DIR / jid / "ata.pdf") == 0

            # exclusão
            for j in (jid, jid2, fantasma.id):
                assert client.delete(f"/api/jobs/{j}").json()["ok"] is True
    finally:
        smtp.stop()

    print("TESTES DA V2 PASSARAM ✔")
    print("amostras em:", OUT)


if __name__ == "__main__":
    main()
