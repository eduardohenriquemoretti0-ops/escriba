"""Envio da ata e da transcrição por e-mail (SMTP).

Configuração no .env (SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD,
SMTP_FROM, SMTP_SECURITY). Sem SMTP_HOST o app continua funcionando e a tela
de envio avisa que falta configurar.
"""
from __future__ import annotations

import re
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formataddr
from typing import Iterable

from . import config

_ADDR = re.compile(r"^[A-Za-z0-9._%+\-']+@[A-Za-z0-9\-]+(\.[A-Za-z0-9\-]+)+$")
_SPLIT = re.compile(r"[,\;\n]+")
_LOCAL = {"localhost", "127.0.0.1", "::1"}


class MailError(RuntimeError):
    """Erro com mensagem já pronta para mostrar ao usuário."""


def configured() -> bool:
    return bool(config.SMTP_HOST and config.SMTP_FROM)


def parse_addresses(raw: str | Iterable[str], field: str = "Para") -> list[str]:
    if isinstance(raw, str):
        parts = _SPLIT.split(raw)
    else:
        parts = [p for item in raw for p in _SPLIT.split(item or "")]
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if not _ADDR.match(p):
            raise MailError(f"E-mail inválido no campo {field}: {p[:60]}")
        if p.lower() not in (a.lower() for a in out):
            out.append(p)
    return out


def _clean_line(text: str, limit: int = 200) -> str:
    return re.sub(r"[\r\n\t]+", " ", text or "").strip()[:limit]


def send(*, to: list[str], cc: list[str], subject: str, body: str,
         attachments: list[tuple[str, bytes, str]],
         sender_name: str = "", reply_to: str = "") -> None:
    """attachments: [(nome_do_arquivo, bytes, 'tipo/subtipo'), ...]"""
    if not configured():
        raise MailError(
            "E-mail não configurado: preencha SMTP_HOST, SMTP_USER, "
            "SMTP_PASSWORD e SMTP_FROM no arquivo .env e reinicie o Escriba.")
    if not to:
        raise MailError("Informe pelo menos um destinatário.")
    if len(to) + len(cc) > config.EMAIL_MAX_RECIPIENTS:
        raise MailError(f"Máximo de {config.EMAIL_MAX_RECIPIENTS} destinatários por envio.")
    total = sum(len(data) for _, data, _ in attachments)
    if total > config.EMAIL_MAX_ATTACH_MB * 1024 * 1024:
        raise MailError(
            f"Anexos somam {total / 1_048_576:.1f} MB; o limite configurado é "
            f"{config.EMAIL_MAX_ATTACH_MB} MB. Envie menos arquivos por vez.")

    msg = EmailMessage()
    msg["From"] = formataddr((_clean_line(sender_name, 80), config.SMTP_FROM))
    msg["To"] = ", ".join(to)
    if cc:
        msg["Cc"] = ", ".join(cc)
    if reply_to and _ADDR.match(reply_to):
        msg["Reply-To"] = reply_to
    msg["Subject"] = _clean_line(subject) or "Ata e transcrição da reunião"
    msg.set_content((body or "").strip() + "\n")
    for name, data, mime in attachments:
        maintype, _, subtype = mime.partition("/")
        msg.add_attachment(data, maintype=maintype, subtype=subtype or "octet-stream",
                           filename=name)

    security = config.SMTP_SECURITY
    if security not in ("starttls", "ssl", "none"):
        raise MailError("SMTP_SECURITY deve ser starttls, ssl ou none.")
    if security == "none" and config.SMTP_PASSWORD and config.SMTP_HOST not in _LOCAL:
        raise MailError(
            "Recusei enviar a senha SMTP sem criptografia. Use "
            "SMTP_SECURITY=starttls (porta 587) ou ssl (porta 465).")

    ctx = ssl.create_default_context()
    try:
        if security == "ssl":
            server = smtplib.SMTP_SSL(config.SMTP_HOST, config.SMTP_PORT,
                                      timeout=60, context=ctx)
        else:
            server = smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=60)
        with server:
            server.ehlo()
            if security == "starttls":
                server.starttls(context=ctx)
                server.ehlo()
            if config.SMTP_USER and config.SMTP_PASSWORD:
                server.login(config.SMTP_USER, config.SMTP_PASSWORD)
            server.send_message(msg, to_addrs=to + cc)
    except smtplib.SMTPAuthenticationError:
        raise MailError("O servidor recusou o usuário/senha de e-mail "
                        "(confira SMTP_USER e SMTP_PASSWORD; em Gmail/Proton "
                        "use senha de app/token SMTP).") from None
    except smtplib.SMTPRecipientsRefused:
        raise MailError("O servidor recusou um ou mais destinatários.") from None
    except (smtplib.SMTPException, OSError, ssl.SSLError) as exc:
        raise MailError(f"Não consegui enviar o e-mail: {exc}") from None
