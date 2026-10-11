"""PDFs da ata e da transcrição, já com a assinatura do cliente.

Motor principal: WeasyPrint (HTML/CSS, no Linux). Quando o WeasyPrint não carrega
(no Windows ele precisa do Pango/GTK, que o programa não traz), o mesmo conteúdo
sai por fpdf2 (Python puro), com layout mais simples e a fonte DejaVu embutida.

Cada PDF fecha com um bloco de assinaturas:
  - o responsável pela elaboração (o cliente que adquiriu o software) já sai
    assinado, com a imagem cadastrada em "Perfil e assinatura";
  - os participantes têm uma linha em branco, com o nome impresso, para
    assinarem por conta própria (impressão ou assinatura digital externa).

Os PDFs são gerados sob demanda e guardados em cache: só são refeitos quando
algo muda (nomes, correções, perfil, assinatura, ata...).
"""
from __future__ import annotations

import base64
import functools
import hashlib
import html
import io
import json
import logging
import re
import threading
import unicodedata
from pathlib import Path
from typing import Callable, Optional

import markdown as md

from . import arquivo_seguro, jobs, profile, review
from .exports import hms

LAYOUT_VERSION = "1"          # mude para invalidar todos os caches de PDF
ACCENT = "#1B5E43"            # verde institucional

_render_lock = threading.Lock()     # WeasyPrint/fontconfig: um PDF por vez
log = logging.getLogger("escriba.pdf")

# Fonte embutida do PDF de reserva (fpdf2). DejaVu: licença Bitstream Vera,
# permite redistribuir; ver app/fonts/LICENCA-DejaVu.txt.
FONTES_DIR = Path(__file__).resolve().parent / "fonts"
FONTE_NORMAL = FONTES_DIR / "DejaVuSans.ttf"
FONTE_NEGRITO = FONTES_DIR / "DejaVuSans-Bold.ttf"
_FAMILIA = "Escriba"
_COR_ACENTO = (27, 94, 67)
_COR_TINTA = (28, 35, 33)
_COR_CINZA = (95, 107, 102)
_COR_SUAVE = (123, 139, 132)
_COR_NOME = (20, 61, 46)
_COR_HORA = (138, 160, 150)
_COR_FILETE = (207, 220, 213)


class PdfError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _e(value: object) -> str:
    return html.escape(str(value or ""), quote=True)


def _css_string(value: str, limit: int = 90) -> str:
    """Texto seguro para usar dentro de content: "..." no CSS."""
    value = re.sub(r"[\r\n\\\"]+", " ", value)[:limit]
    return value


def _make_fetcher():
    """Só aceita imagens embutidas (data:). Nada de buscar URL externa a
    partir de texto vindo da transcrição/ata.

    WeasyPrint >= 68 usa a classe URLFetcher; versões anteriores, uma função.
    """
    try:
        from weasyprint.urls import URLFetcher
    except ImportError:                       # WeasyPrint antigo
        from weasyprint import default_url_fetcher

        def fetcher(url, *args, **kwargs):
            if url.startswith("data:"):
                return default_url_fetcher(url)
            raise ValueError(f"Recurso externo bloqueado: {url[:60]}")
        return fetcher
    return URLFetcher(allowed_protocols=["data"])


def _write_pdf(html_doc: str, dest: Path) -> None:
    from weasyprint import HTML
    tmp = dest.with_suffix(".pdf.tmp")
    with _render_lock:
        HTML(string=html_doc, url_fetcher=_make_fetcher()).write_pdf(str(tmp))
    arquivo_seguro.trocar(tmp, dest)     # Windows: PDF aberto no leitor -> 409 amigável


def _publicar(dest: Path, html_doc: Callable[[], str], reserva: Callable[[], bytes]) -> None:
    """Grava o PDF com WeasyPrint; se ele não carregar, com fpdf2.

    `html_doc` e `reserva` são funções para só montar o conteúdo do motor que for usado.
    """
    try:
        from weasyprint import HTML  # noqa: F401 — teste de carga (Windows sem GTK falha aqui)
    except (ImportError, OSError) as exc:
        log.warning("WeasyPrint indisponível (%s); gerando %s com fpdf2.", exc, dest.name)
        _gravar_pdf(dest, reserva())
        return
    _write_pdf(html_doc(), dest)


def _gravar_pdf(dest: Path, dados: bytes) -> None:
    tmp = dest.with_suffix(".pdf.tmp")
    with _render_lock:
        tmp.write_bytes(dados)
    arquivo_seguro.trocar(tmp, dest)


# --------------------------------------------------------------------------
# Estilo
# --------------------------------------------------------------------------
def _css(footer_left: str) -> str:
    return """
@page {
  size: A4;
  margin: 2.1cm 1.9cm 2.3cm 1.9cm;
  @bottom-left { content: "%(footer)s";
    font-family: "Liberation Sans", sans-serif; font-size: 7.5pt;
    color: #8AA096; letter-spacing: .04em; }
  @bottom-right { content: "Página " counter(page) " de " counter(pages);
    font-family: "Liberation Sans", sans-serif; font-size: 7.5pt;
    color: #8AA096; }
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: "Liberation Serif", serif; font-size: 10.5pt;
  color: #1C2321; line-height: 1.5; }

.topo { border-bottom: 2.5pt solid %(accent)s; padding-bottom: 12pt;
  margin-bottom: 16pt; }
.orgao { font-family: "Liberation Sans", sans-serif; font-size: 8.5pt;
  letter-spacing: .18em; text-transform: uppercase; color: %(accent)s;
  font-weight: bold; }
h1 { font-family: "Liberation Sans", sans-serif; font-size: 20pt;
  font-weight: bold; margin-top: 6pt; letter-spacing: .01em; }
.subtitulo { font-family: "Liberation Sans", sans-serif; font-size: 10pt;
  color: #5F6B66; margin-top: 3pt; }

table.meta { width: 100%%; border-collapse: collapse; margin: 4pt 0 14pt; }
table.meta td { padding: 4pt 10pt 4pt 0; vertical-align: top;
  border-bottom: .5pt solid #E3EAE6; }
table.meta .rotulo { font-family: "Liberation Sans", sans-serif; font-size: 7.5pt;
  letter-spacing: .12em; text-transform: uppercase; color: #7B8B84;
  width: 25%%; white-space: nowrap; }

h2 { font-family: "Liberation Sans", sans-serif; font-size: 11pt;
  letter-spacing: .1em; text-transform: uppercase; color: %(accent)s;
  margin: 18pt 0 7pt; padding-bottom: 3pt; border-bottom: .75pt solid #CFDCD5;
  page-break-after: avoid; }
h3 { font-family: "Liberation Sans", sans-serif; font-size: 10.5pt;
  margin: 12pt 0 4pt; page-break-after: avoid; }
p { margin-bottom: 7pt; text-align: justify; }
ul, ol { margin: 0 0 8pt 16pt; }
li { margin-bottom: 5pt; text-align: justify; }
strong { color: #143D2E; }
blockquote { border-left: 2pt solid #CFDCD5; padding-left: 9pt;
  color: #5F6B66; margin: 6pt 0 10pt; }

table.acoes { width: 100%%; border-collapse: collapse; font-size: 9pt;
  font-family: "Liberation Sans", sans-serif; margin: 6pt 0 10pt; }
table.acoes th { background: %(accent)s; color: #FFF; text-align: left;
  padding: 5pt 7pt; font-size: 8pt; letter-spacing: .08em;
  text-transform: uppercase; }
table.acoes td { padding: 5pt 7pt; border-bottom: .5pt solid #E3EAE6;
  vertical-align: top; }
table.acoes tr { page-break-inside: avoid; }
table.acoes tr:nth-child(even) td { background: #F6F9F7; }

.fala { margin-bottom: 8pt; page-break-inside: avoid; }
.quem { font-family: "Liberation Sans", sans-serif; font-weight: bold;
  font-size: 9.5pt; }
.ts { font-family: "DejaVu Sans Mono", monospace; font-size: 8pt;
  color: #8AA096; }
.texto { margin-top: 1pt; text-align: justify; font-size: 10pt; }
.nota { font-style: italic; color: #5F6B66; font-size: 9.5pt;
  border-left: 2pt solid #CFDCD5; padding-left: 9pt; margin: 10pt 0 14pt; }

/* ---------- assinaturas ---------- */
.assinaturas { margin-top: 30pt; }
.assinaturas.compacto { page-break-inside: avoid; }
.assinaturas h2 { margin-top: 0; }
.local-data { font-family: "Liberation Sans", sans-serif; font-size: 9pt;
  color: #5F6B66; margin-bottom: 10pt; }
.resp { width: 55%%; margin: 4pt 0 18pt; page-break-inside: avoid; }
.resp .espaco { height: 52pt; text-align: center; }
.resp img.sig { max-height: 50pt; max-width: 190pt; }
.linha-ass { border-top: .75pt solid #1C2321; padding-top: 4pt;
  font-family: "Liberation Sans", sans-serif; font-size: 9pt;
  text-align: center; }
.cargo { font-size: 8pt; color: #5F6B66; }
.papel { font-family: "Liberation Sans", sans-serif; font-size: 7.5pt;
  letter-spacing: .1em; text-transform: uppercase; color: #7B8B84;
  text-align: center; margin-top: 2pt; }
.sub-ass { font-family: "Liberation Sans", sans-serif; font-size: 8pt;
  letter-spacing: .12em; text-transform: uppercase; color: #7B8B84;
  margin-bottom: 4pt; }
table.part { width: 100%%; border-collapse: collapse; }
table.part tr { page-break-inside: avoid; }
table.part td { width: 50%%; padding: 30pt 10pt 0; vertical-align: bottom; }
""" % {"footer": _css_string(footer_left), "accent": ACCENT}


# --------------------------------------------------------------------------
# Blocos de HTML
# --------------------------------------------------------------------------
def _topo(org: str, titulo: str, sub: str) -> str:
    return (f'<div class="topo"><div class="orgao">{_e(org)}</div>'
            f'<h1>{_e(titulo)}</h1><div class="subtitulo">{_e(sub)}</div></div>')


def _participant_names(job: jobs.Job) -> list[str]:
    """Quem assina: os participantes informados; senão os falantes que já
    receberam nome (os 'Falante N' genéricos ficam de fora)."""
    if job.participants:
        return list(job.participants)
    named = [n for n in job.speakers if not re.match(r"^Falante \d+$", n)]
    return named


def _meta_pares(job: jobs.Job, who: list[str]) -> list[tuple[str, str]]:
    m = job.meeting
    return [("Reunião", review.default_title(job)),
            ("Data", m.get("date", "")),
            ("Local", m.get("place", "")),
            ("Duração da gravação", hms(job.duration) if job.duration else ""),
            ("Participantes", ", ".join(who))]


def _meta_rows(job: jobs.Job, who: list[str]) -> str:
    body = "".join(
        f'<tr><td class="rotulo">{_e(k)}</td><td>{_e(v)}</td></tr>'
        for k, v in _meta_pares(job, who) if v)
    return f'<table class="meta">{body}</table>' if body else ""


def _signature_block(job: jobs.Job, prof: dict, who: list[str]) -> str:
    sig = profile.signature_bytes()
    partes = ['<div class="assinaturas%s">' % (
        " compacto" if len(who) <= 8 else "")]
    partes.append("<h2>Assinaturas</h2>")

    m = job.meeting
    local = ", ".join(x for x in (m.get("place", ""), m.get("date", "")) if x)
    if local:
        partes.append(f'<div class="local-data">{_e(local)}</div>')

    if prof.get("nome") or sig:
        img = ""
        if sig:
            b64 = base64.b64encode(sig).decode("ascii")
            img = f'<img class="sig" src="data:image/png;base64,{b64}" alt="">'
        cargo = " — ".join(x for x in (prof.get("cargo"), prof.get("organizacao")) if x)
        partes.append(
            '<div class="resp"><div class="espaco">' + img + "</div>"
            f'<div class="linha-ass"><strong>{_e(prof.get("nome") or "Responsável")}'
            f'</strong>' + (f'<br><span class="cargo">{_e(cargo)}</span>' if cargo else "")
            + '</div><div class="papel">Responsável pela elaboração</div></div>')

    partes.append('<div class="sub-ass">Participantes</div>')
    nomes = who or [""] * 6          # sem nomes: 6 linhas em branco
    cel = lambda n: (  # noqa: E731
        f'<td><div class="linha-ass">{_e(n) if n else "Nome"}</div></td>')
    linhas = []
    for i in range(0, len(nomes), 2):
        par = nomes[i:i + 2]
        if len(par) == 1:
            par.append(None)                       # sem par: célula vazia
        linhas.append("<tr>" + "".join(
            cel(n) if n is not None else "<td></td>" for n in par) + "</tr>")
    partes.append('<table class="part">' + "".join(linhas) + "</table></div>")
    return "".join(partes)


def _subtitle(job: jobs.Job, extra: str = "") -> str:
    m = job.meeting
    pieces = [review.default_title(job), m.get("date", ""), m.get("place", ""), extra]
    return " · ".join(p for p in pieces if p)


# --------------------------------------------------------------------------
# Ata
# --------------------------------------------------------------------------
def _ata_html_body(markdown_text: str) -> str:
    # escapa HTML cru vindo do texto (a ata é gerada por IA) antes do Markdown
    safe = markdown_text.replace("&", "&amp;").replace("<", "&lt;")
    body = md.markdown(safe, extensions=["tables", "sane_lists"])
    body = re.sub(r"^\s*<h1>.*?</h1>\s*", "", body, count=1, flags=re.S)  # o topo já tem título
    return body.replace("<table>", '<table class="acoes">')


def _cache_key(job: jobs.Job, kind: str, source: Path) -> str:
    sig = profile.signature_bytes() or b""
    payload = json.dumps({
        "v": LAYOUT_VERSION, "kind": kind, "meeting": job.meeting,
        "participants": job.participants, "names": job.speaker_names,
        "corrections": job.corrections, "duration": job.duration,
        "filename": job.filename, "profile": profile.load(),
        "sig": hashlib.sha256(sig).hexdigest(),
        "src": hashlib.sha256(source.read_bytes()).hexdigest(),
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cached(job: jobs.Job, name: str, key: str) -> Optional[Path]:
    pdf, keyfile = job.path(name), job.path(name + ".key")
    if pdf.exists() and keyfile.exists() and keyfile.read_text(encoding="utf-8") == key:
        return pdf
    return None


def ata_pdf(job: jobs.Job) -> Path:
    ata_path = job.path("ata.md")
    if not ata_path.exists():
        raise PdfError("A ata ainda não foi gerada para esta reunião.")
    key = _cache_key(job, "ata", ata_path)
    hit = _cached(job, "ata.pdf", key)
    if hit:
        return hit

    prof = profile.load()
    org = prof.get("organizacao") or "Escriba"
    who = _participant_names(job)
    # A ata já nasce da transcrição com nomes/grafia corrigidos; reaplicar as
    # correções aqui poderia duplicar trechos (ex.: "Ana" -> "Ana Maria Maria").
    # Se o cliente mudar a revisão depois, a tela oferece "Refazer a ata".
    texto = ata_path.read_text(encoding="utf-8")

    doc = (
        "<html><head><meta charset='utf-8'><style>"
        + _css(f"{org} · Ata · {review.default_title(job)}") + "</style></head><body>"
        + _topo(org, "Ata da reunião", _subtitle(job))
        + _meta_rows(job, who)
        + _ata_html_body(texto)
        + _signature_block(job, prof, who)
        + "</body></html>"
    )
    _publicar(job.path("ata.pdf"), lambda: doc,
              lambda: _pdf_reserva_ata(job, org, prof, who, texto))
    job.path("ata.pdf.key").write_text(key, encoding="utf-8")
    return job.path("ata.pdf")


# --------------------------------------------------------------------------
# Transcrição
# --------------------------------------------------------------------------
def transcript_pdf(job: jobs.Job, show_times: bool = False) -> Path:
    seg_path = job.path(review.SEG_FILE)
    if not seg_path.exists():
        raise PdfError("A transcrição ainda não está pronta.")
    key = _cache_key(job, f"transcricao:{show_times}", seg_path)
    hit = _cached(job, "transcricao.pdf", key)
    if hit:
        return hit

    prof = profile.load()
    org = prof.get("organizacao") or "Escriba"
    segs = review.segments_for(job)
    names = review.names_for(job, segs)

    # junta falas seguidas do mesmo nome (acontece quando dois "Falante N"
    # foram mapeados para a mesma pessoa)
    falas: list[dict] = []
    for s in segs:
        nome = names[s.speaker]
        if falas and falas[-1]["quem"] == nome:
            falas[-1]["txt"] += " " + s.text
        else:
            falas.append({"quem": nome, "ts": s.start, "txt": s.text})

    blocos = []
    for f in falas:
        ts = f' <span class="ts">{hms(f["ts"])}</span>' if show_times else ""
        blocos.append(
            f'<div class="fala"><span class="quem">{_e(f["quem"])}</span>{ts}'
            f'<div class="texto">{_e(f["txt"])}</div></div>')

    who = _participant_names(job)
    dur = f"duração da gravação: {hms(job.duration)}" if job.duration else ""
    doc = (
        "<html><head><meta charset='utf-8'><style>"
        + _css(f"{org} · Transcrição · {review.default_title(job)}") + "</style></head><body>"
        + _topo(org, "Transcrição da reunião", _subtitle(job, dur))
        + _meta_rows(job, who)
        + "".join(blocos)
        + _signature_block(job, prof, who)
        + "</body></html>"
    )
    _publicar(job.path("transcricao.pdf"), lambda: doc,
              lambda: _pdf_reserva_transcricao(job, org, prof, who, falas, show_times))
    job.path("transcricao.pdf.key").write_text(key, encoding="utf-8")
    return job.path("transcricao.pdf")


# --------------------------------------------------------------------------
# PDF de reserva (fpdf2, Python puro): usado quando o WeasyPrint não carrega
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _cmap_fonte() -> frozenset[int]:
    """Caracteres que a fonte embutida desenha (evita glifo ausente no PDF)."""
    from fontTools.ttLib import TTFont

    fonte = TTFont(str(FONTE_NORMAL), lazy=True)
    try:
        return frozenset((fonte.getBestCmap() or {}).keys())
    finally:
        fonte.close()


def _texto_pdf(texto: object) -> str:
    """Texto que a fonte consegue desenhar; o que faltar vira '?'."""
    bruto = unicodedata.normalize("NFC", str(texto or "")).replace("\r", "")
    bruto = bruto.replace("\t", "    ")
    cmap = _cmap_fonte()
    return "".join(c if (c == "\n" or ord(c) in cmap) else "?" for c in bruto)


def _novo_pdf(titulo: str, autor: str, rodape: str):
    from fpdf import FPDF

    class _Pdf(FPDF):
        def footer(self) -> None:
            self.set_y(-13)
            self.set_font(_FAMILIA, "", 7.5)
            self.set_text_color(*_COR_HORA)
            self.cell(self.epw - 45, 5, _texto_pdf(rodape)[:110], align="L")
            self.cell(45, 5, f"Página {self.page_no()} de {{nb}}", align="R",
                      new_x="LMARGIN", new_y="TOP")

    pdf = _Pdf(orientation="P", unit="mm", format="A4")
    pdf.set_margins(20, 20, 20)
    pdf.set_auto_page_break(True, margin=22)
    pdf.set_title(_texto_pdf(titulo))
    pdf.set_author(_texto_pdf(autor))
    pdf.set_creator("Escriba")
    pdf.add_font(_FAMILIA, "", str(FONTE_NORMAL))
    pdf.add_font(_FAMILIA, "B", str(FONTE_NEGRITO))
    pdf.alias_nb_pages()
    pdf.add_page()
    return pdf


def _fpdf_topo(pdf, org: str, titulo: str, sub: str, meta: list[tuple[str, str]]) -> None:
    pdf.set_font(_FAMILIA, "B", 8.5)
    pdf.set_text_color(*_COR_ACENTO)
    pdf.cell(0, 5, _texto_pdf(org).upper(), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(_FAMILIA, "B", 17)
    pdf.set_text_color(*_COR_TINTA)
    pdf.multi_cell(0, 8, _texto_pdf(titulo), new_x="LMARGIN", new_y="NEXT")
    if sub:
        pdf.set_font(_FAMILIA, "", 9.5)
        pdf.set_text_color(*_COR_CINZA)
        pdf.multi_cell(0, 5, _texto_pdf(sub), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_draw_color(*_COR_ACENTO)
    pdf.set_line_width(0.8)
    y = pdf.get_y()
    pdf.line(20, y, 190, y)
    pdf.ln(4)
    for rotulo, valor in meta:
        if not valor:
            continue
        pdf.set_font(_FAMILIA, "B", 7.5)
        pdf.set_text_color(*_COR_SUAVE)
        pdf.cell(42, 5.5, _texto_pdf(rotulo).upper())
        pdf.set_font(_FAMILIA, "", 9.5)
        pdf.set_text_color(*_COR_TINTA)
        pdf.multi_cell(0, 5.5, _texto_pdf(valor), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)


def _fpdf_secao(pdf, titulo: str) -> None:
    if pdf.get_y() > 250:
        pdf.add_page()
    pdf.ln(3)
    pdf.set_font(_FAMILIA, "B", 9.5)
    pdf.set_text_color(*_COR_ACENTO)
    pdf.cell(0, 6, _texto_pdf(titulo).upper(), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*_COR_FILETE)
    pdf.set_line_width(0.3)
    y = pdf.get_y()
    pdf.line(20, y, 190, y)
    pdf.ln(2)


def _fpdf_fala(pdf, quem: str, hora: str, texto: str) -> None:
    if pdf.get_y() > 255:            # não deixa o nome da fala sozinho no fim da página
        pdf.add_page()
    pdf.set_font(_FAMILIA, "B", 9.5)
    pdf.set_text_color(*_COR_NOME)
    pdf.write(5.5, _texto_pdf(quem))
    if hora:
        pdf.set_font(_FAMILIA, "", 8)
        pdf.set_text_color(*_COR_HORA)
        pdf.write(5.5, "   " + hora)
    pdf.ln(5.5)
    pdf.set_font(_FAMILIA, "", 10)
    pdf.set_text_color(*_COR_TINTA)
    pdf.multi_cell(0, 5.2, _texto_pdf(texto), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2.5)


_MD_MARCAS = re.compile(r"\*\*|__|\*|`")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_SEPARADOR = re.compile(r"\s*\|?[\s:|-]+\|?\s*")


def _md_plano(texto: str) -> str:
    """Tira a marcação de Markdown que o PDF de reserva não desenha."""
    return _MD_MARCAS.sub("", _MD_LINK.sub(r"\1", texto)).strip()


def _fpdf_markdown(pdf, texto: str) -> None:
    """Ata em Markdown: títulos, listas, tabelas e parágrafos, em layout simples."""
    linhas = texto.splitlines()
    # o título (# ...) já está no topo do documento
    while linhas and not linhas[0].strip():
        linhas.pop(0)
    if linhas and re.match(r"^#\s", linhas[0]):
        linhas.pop(0)
    for bruta in linhas:
        linha = bruta.rstrip()
        if not linha.strip():
            pdf.ln(2)
            continue
        if "-" in linha and _MD_SEPARADOR.fullmatch(linha):
            continue                                  # linha |---|---| de tabela
        pdf.set_font(_FAMILIA, "", 10)
        pdf.set_text_color(*_COR_TINTA)
        if linha.lstrip().startswith("|"):
            celulas = [_md_plano(c) for c in linha.strip().strip("|").split("|")]
            pdf.set_font(_FAMILIA, "", 9)
            pdf.multi_cell(0, 5, _texto_pdf("  |  ".join(c for c in celulas if c)),
                           new_x="LMARGIN", new_y="NEXT")
            continue
        cabeca = re.match(r"^(#{1,6})\s+(.*)$", linha)
        if cabeca:
            pdf.ln(2)
            pdf.set_font(_FAMILIA, "B", 11 if len(cabeca.group(1)) <= 2 else 10)
            pdf.set_text_color(*_COR_ACENTO)
            pdf.multi_cell(0, 6, _texto_pdf(_md_plano(cabeca.group(2))),
                           new_x="LMARGIN", new_y="NEXT")
            continue
        item = re.match(r"^\s*([-*+]|\d+[.)])\s+(.*)$", linha)
        if item:
            marca = item.group(1) if item.group(1)[0].isdigit() else "•"
            pdf.set_x(26)
            pdf.multi_cell(164, 5.2, _texto_pdf(f"{marca}  {_md_plano(item.group(2))}"),
                           new_x="LMARGIN", new_y="NEXT")
            continue
        pdf.multi_cell(0, 5.2, _texto_pdf(_md_plano(linha)), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def _fpdf_assinaturas(pdf, job: jobs.Job, prof: dict, who: list[str]) -> None:
    _fpdf_secao(pdf, "Assinaturas")
    m = job.meeting
    local = ", ".join(x for x in (m.get("place", ""), m.get("date", "")) if x)
    if local:
        pdf.set_font(_FAMILIA, "", 9)
        pdf.set_text_color(*_COR_CINZA)
        pdf.cell(0, 5, _texto_pdf(local), new_x="LMARGIN", new_y="NEXT")

    sig = profile.signature_bytes()
    if prof.get("nome") or sig:
        y0 = pdf.get_y() + 1
        if sig:
            pdf.image(io.BytesIO(sig), x=24, y=y0, h=16)
        pdf.set_y(y0 + 18)
        pdf.set_draw_color(*_COR_TINTA)
        pdf.set_line_width(0.3)
        y = pdf.get_y()
        pdf.line(20, y, 95, y)
        pdf.set_font(_FAMILIA, "B", 9)
        pdf.set_text_color(*_COR_TINTA)
        pdf.set_x(20)
        pdf.cell(75, 5, _texto_pdf(prof.get("nome") or "Responsável"), align="C",
                 new_x="LMARGIN", new_y="NEXT")
        cargo = " — ".join(x for x in (prof.get("cargo"), prof.get("organizacao")) if x)
        if cargo:
            pdf.set_font(_FAMILIA, "", 8)
            pdf.set_text_color(*_COR_CINZA)
            pdf.set_x(20)
            pdf.cell(75, 4.5, _texto_pdf(cargo), align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font(_FAMILIA, "", 7.5)
        pdf.set_text_color(*_COR_SUAVE)
        pdf.set_x(20)
        pdf.cell(75, 4.5, "RESPONSÁVEL PELA ELABORAÇÃO", align="C",
                 new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)
    pdf.set_font(_FAMILIA, "B", 7.5)
    pdf.set_text_color(*_COR_SUAVE)
    pdf.cell(0, 5, "PARTICIPANTES", new_x="LMARGIN", new_y="NEXT")
    nomes = who or [""] * 6                  # sem nomes: 6 linhas em branco
    for i in range(0, len(nomes), 2):
        if pdf.get_y() > 245:
            pdf.add_page()
        y = pdf.get_y() + 14
        for col, nome in enumerate(nomes[i:i + 2]):
            x = 20 + col * 90
            pdf.set_draw_color(*_COR_TINTA)
            pdf.line(x, y, x + 75, y)
            pdf.set_xy(x, y + 0.8)
            pdf.set_font(_FAMILIA, "", 9)
            pdf.set_text_color(*_COR_TINTA)
            pdf.cell(75, 5, _texto_pdf(nome) if nome else "Nome", align="C")
        pdf.set_y(y + 9)


def _pdf_reserva_ata(job: jobs.Job, org: str, prof: dict, who: list[str], texto: str) -> bytes:
    pdf = _novo_pdf("Ata da reunião", org, f"{org} · Ata · {review.default_title(job)}")
    _fpdf_topo(pdf, org, "Ata da reunião", _subtitle(job), _meta_pares(job, who))
    _fpdf_markdown(pdf, texto)
    _fpdf_assinaturas(pdf, job, prof, who)
    return bytes(pdf.output())


def _pdf_reserva_transcricao(job: jobs.Job, org: str, prof: dict, who: list[str],
                             falas: list[dict], mostrar_horas: bool) -> bytes:
    dur = f"duração da gravação: {hms(job.duration)}" if job.duration else ""
    pdf = _novo_pdf("Transcrição da reunião", org,
                    f"{org} · Transcrição · {review.default_title(job)}")
    _fpdf_topo(pdf, org, "Transcrição da reunião", _subtitle(job, dur),
               _meta_pares(job, who))
    for f in falas:
        hora = hms(f["ts"]) if mostrar_horas else ""
        _fpdf_fala(pdf, f["quem"], hora, f["txt"])
    _fpdf_assinaturas(pdf, job, prof, who)
    return bytes(pdf.output())
