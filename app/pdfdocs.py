"""PDFs da ata e da transcrição (WeasyPrint), já com a assinatura do cliente.

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
import hashlib
import html
import json
import re
import threading
from pathlib import Path
from typing import Optional

import markdown as md

from . import jobs, profile, review
from .exports import hms

LAYOUT_VERSION = "1"          # mude para invalidar todos os caches de PDF
ACCENT = "#1B5E43"            # verde institucional

_render_lock = threading.Lock()     # WeasyPrint/fontconfig: um PDF por vez


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
    tmp.replace(dest)


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


def _meta_rows(job: jobs.Job, who: list[str]) -> str:
    m = job.meeting
    rows = [("Reunião", review.default_title(job)),
            ("Data", m.get("date", "")),
            ("Local", m.get("place", "")),
            ("Duração da gravação", hms(job.duration) if job.duration else ""),
            ("Participantes", ", ".join(who))]
    body = "".join(
        f'<tr><td class="rotulo">{_e(k)}</td><td>{_e(v)}</td></tr>'
        for k, v in rows if v)
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
    if pdf.exists() and keyfile.exists() and keyfile.read_text() == key:
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
    _write_pdf(doc, job.path("ata.pdf"))
    job.path("ata.pdf.key").write_text(key)
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
    _write_pdf(doc, job.path("transcricao.pdf"))
    job.path("transcricao.pdf.key").write_text(key)
    return job.path("transcricao.pdf")
