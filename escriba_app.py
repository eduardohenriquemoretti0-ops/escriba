"""Escriba como programa de desktop.

Abre o Escriba numa janela própria — sem navegador, sem precisar de internet.
É o ponto de entrada do executável (Escriba.exe / Escriba no Linux) e do
`run` (run.bat / run.sh).

Por dentro, a interface continua sendo a mesma página web, servida por um
servidor embutido que só escuta em 127.0.0.1, numa porta aleatória, e só
responde a quem tiver o token desta sessão (a própria janela). Nenhuma outra
máquina da rede — nem outra aba de navegador — consegue usá-lo.
"""
from __future__ import annotations

import logging
import os
import secrets
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _setup_logging() -> Path:
    # sem console no .exe: erros vão para um arquivo que o suporte pode pedir
    from app import config

    log_file = config.DATA_DIR / "escriba.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8")],
    )
    if sys.stderr is None:  # PyInstaller --windowed: sem stdout/stderr
        sys.stdout = sys.stderr = open(log_file, "a", encoding="utf-8")  # noqa: SIM115
    return log_file


def _wait_ready(url: str, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return
        except Exception:  # noqa: BLE001 — servidor ainda subindo
            time.sleep(0.2)
    raise RuntimeError("O servidor interno do Escriba não respondeu.")


def main() -> None:
    token = secrets.token_urlsafe(24)
    os.environ["ESCRIBA_SESSION_TOKEN"] = token   # lido pelo app.main
    log_file = _setup_logging()
    log = logging.getLogger("escriba")

    import uvicorn

    from app.main import app

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=port, log_config=None, log_level="info"))
    threading.Thread(target=server.run, daemon=True, name="escriba-server").start()

    base = f"http://127.0.0.1:{port}"
    _wait_ready(f"{base}/api/health")
    url = f"{base}/?t={token}"
    log.info("Escriba pronto em %s (log: %s)", base, log_file)

    try:
        import webview
    except ImportError:
        webview = None

    if webview is None or os.getenv("ESCRIBA_NAVEGADOR") == "1":
        # Sem componente de janela: usa o navegador padrão (continua offline)
        import webbrowser
        webbrowser.open(url)
        print(f"Escriba aberto no navegador. Feche esta janela para encerrar.\n{base}")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    else:
        webview.settings["ALLOW_DOWNLOADS"] = True       # baixar TXT/PDF/...
        webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True
        webview.create_window("Escriba — transcrição de reuniões", url,
                              width=1280, height=860, min_size=(900, 600))
        webview.start(private_mode=False)

    server.should_exit = True


if __name__ == "__main__":
    main()
