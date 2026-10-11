"""PDF de reserva (fpdf2): transcrição e ata quando o WeasyPrint não carrega.

No Windows o WeasyPrint precisa de Pango/GTK, que o programa não traz. Aqui o
import do WeasyPrint é forçado a falhar (sys.modules["weasyprint"] = None), e o
PDF tem que sair pelo fpdf2, com acentos e nomes preservados.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import config, jobs, pdfdocs, profile, review  # noqa: E402
from app.engines.base import Segment  # noqa: E402

FALA_COM_ACENTOS = "Bom dia, pessoal. Ação, coração e informação são importantes."


@pytest.fixture
def dados(tmp_path, monkeypatch):
    """Dados do Escriba num diretório temporário (jobs, perfil e assinatura)."""
    perfil = tmp_path / "profile"
    (tmp_path / "jobs").mkdir()
    perfil.mkdir()
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(config, "PROFILE_DIR", perfil)
    monkeypatch.setattr(profile, "PROFILE_JSON", perfil / "perfil.json")
    monkeypatch.setattr(profile, "SIGNATURE_PNG", perfil / "assinatura.png")
    return tmp_path


@pytest.fixture
def sem_weasyprint(monkeypatch):
    """Simula o Windows sem Pango/GTK: `from weasyprint import ...` falha."""
    monkeypatch.setitem(sys.modules, "weasyprint", None)


def _assinatura_png() -> bytes:
    """Rubrica simples com fundo transparente (PNG RGBA, como a da tela de perfil)."""
    img = Image.new("RGBA", (300, 90), (0, 0, 0, 0))
    ImageDraw.Draw(img).line([(10, 60), (90, 15), (150, 70), (280, 20)],
                             fill=(16, 24, 40, 255), width=6)
    saida = io.BytesIO()
    img.save(saida, format="PNG")
    return saida.getvalue()


def _job() -> jobs.Job:
    job = jobs.new_job(filename="reuniao_jose.m4a", engine="local", language="pt",
                       min_speakers=None, max_speakers=None, want_summary=False,
                       participants=["Edna", "José Eduardo"])
    job.duration = 1359.0
    job.meeting = {"title": "Reunião de sábado", "date": "10/10/2026",
                   "place": "Salão do Reino"}
    job.speaker_names = {"Falante 1": "Edna", "Falante 2": "José Eduardo"}
    job.status = jobs.STATUS_DONE
    job.save()
    review.save_raw(job, [
        Segment(0.0, 4.0, "Falante 1", FALA_COM_ACENTOS),
        Segment(4.5, 9.0, "Falante 2", "Perfeito, José. Obrigado, é isso mesmo."),
    ])
    profile.save({"nome": "José Eduardo", "cargo": "Coordenador",
                  "organizacao": "Congregação Edna"})
    profile.set_signature(_assinatura_png())
    return job


def _texto_do_pdf(caminho: Path) -> str:
    pypdf = pytest.importorskip("pypdf")
    leitor = pypdf.PdfReader(str(caminho))
    return "\n".join(pagina.extract_text() for pagina in leitor.pages)


def test_transcricao_sai_pelo_fpdf_com_acentos(dados, sem_weasyprint, caplog):
    job = _job()

    caminho = pdfdocs.transcript_pdf(job, show_times=True)

    assert caminho.read_bytes()[:5] == b"%PDF-"
    assert "gerando transcricao.pdf com fpdf2" in caplog.text   # prova o caminho usado
    texto = _texto_do_pdf(caminho)
    for trecho in ("Transcrição da reunião", "Edna", "José Eduardo",
                   "Salão do Reino", "10/10/2026", "Ação", "coração", "informação",
                   "são importantes", "é isso mesmo"):
        assert trecho in texto, f"faltou no PDF: {trecho!r}"
    assert "00:00:00" in texto                     # hora do início da fala


def test_transcricao_sem_horas_por_padrao(dados, sem_weasyprint):
    caminho = pdfdocs.transcript_pdf(_job())
    texto = _texto_do_pdf(caminho)
    assert "José Eduardo" in texto
    assert "00:00:00" not in texto


def test_falante_em_negrito_no_fpdf(dados, sem_weasyprint):
    pypdf = pytest.importorskip("pypdf")
    caminho = pdfdocs.transcript_pdf(_job())
    leitor = pypdf.PdfReader(str(caminho))
    fontes = leitor.pages[0]["/Resources"]["/Font"]
    nomes = {str(fontes[k].get_object()["/BaseFont"]) for k in fontes}
    assert any("Bold" in n for n in nomes), nomes


def test_ata_sai_pelo_fpdf_com_markdown_e_acentos(dados, sem_weasyprint):
    job = _job()
    job.path("ata.md").write_text(
        "# Ata da reunião\n\n"
        "## Decisões\n\n"
        "- **Ação** aprovada por José Eduardo.\n"
        "- Próxima reunião com a Edna.\n\n"
        "| Responsável | Prazo |\n"
        "|---|---|\n"
        "| Edna | 10/11/2026 |\n",
        encoding="utf-8")

    caminho = pdfdocs.ata_pdf(job)

    assert caminho.read_bytes()[:5] == b"%PDF-"
    texto = _texto_do_pdf(caminho)
    for trecho in ("Decisões", "Ação", "José Eduardo", "Responsável", "Prazo",
                   "10/11/2026", "ASSINATURAS"):
        assert trecho in texto, f"faltou na ata: {trecho!r}"
    assert "**" not in texto                       # marcação do Markdown não sai no PDF


def test_caracteres_sem_glifo_viram_interrogacao():
    assert pdfdocs._texto_pdf("José ç ã é") == "José ç ã é"
    assert pdfdocs._texto_pdf("a一b") == "a?b"          # ideograma: DejaVu Sans não tem


def test_fallback_grava_pdf_completo_sem_temporario(dados, sem_weasyprint):
    job = _job()
    caminho = pdfdocs.transcript_pdf(job)
    assert caminho.exists()
    assert not caminho.with_suffix(".pdf.tmp").exists()
    assert job.path("transcricao.pdf.key").read_text(encoding="utf-8")
