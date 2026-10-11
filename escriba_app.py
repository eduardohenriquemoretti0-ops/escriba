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

log = logging.getLogger("escriba")


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


class JsApi:
    """Métodos que a página chama por `window.pywebview.api` (só no programa).

    O pywebview roda cada chamada numa thread própria, então é seguro abrir o
    diálogo daqui. Métodos e atributos com `_` não ficam expostos à página.
    """

    def __init__(self) -> None:
        self._janela = None            # preenchido logo após create_window
        self._lock = threading.Lock()  # um diálogo de salvar por vez
        self._ultima_pasta: Path | None = None

    def salvar_arquivo(self, job_id: str, kind: str, nome_sugerido: str = "") -> dict:
        """Pede o caminho ao usuário e grava o arquivo pedido.

        Args:
            job_id: Id do job (validado em `app.salvar`).
            kind: Tipo do arquivo (lista branca em `app.salvar`).
            nome_sugerido: Nome preferido; é sanitizado no servidor.

        Returns:
            `{"status": "salvo", "caminho": ...}`, `{"status": "cancelado"}` ou
            `{"status": "erro", "mensagem": ...}`. O "salvo" só vem depois de o
            arquivo estar gravado.
        """
        from app import salvar

        if not self._lock.acquire(blocking=False):
            return {"status": "erro", "mensagem": "Já há uma janela de salvar aberta."}
        try:
            return self._salvar(salvar, job_id, kind, nome_sugerido)
        except salvar.SalvarError as exc:
            log.warning("Salvar %s/%s falhou: %s", job_id, kind, exc)
            return {"status": "erro", "mensagem": str(exc)}
        except Exception as exc:  # noqa: BLE001 — fronteira da API: registra e avisa
            log.exception("Erro inesperado ao salvar %s/%s", job_id, kind)
            return {"status": "erro", "mensagem": f"Não consegui salvar: {exc}"}
        finally:
            self._lock.release()

    def _salvar(self, salvar, job_id: str, kind: str, nome_sugerido: str) -> dict:
        arquivo = salvar.preparar(job_id, kind)   # valida antes de abrir diálogo
        if self._janela is None:
            raise salvar.SalvarError("A janela do Escriba não está pronta.")

        extensao = arquivo.extensao
        nome = salvar.nome_seguro(nome_sugerido, extensao, arquivo.nome)
        escolha = self._janela.create_file_dialog(
            _tipo_salvar(),
            directory=str(self._pasta_inicial()),
            save_filename=nome,
            file_types=(f"Arquivo {extensao.upper()} (*.{extensao})",),
        )
        if not escolha:
            return {"status": "cancelado"}

        destino = Path(escolha[0])
        if not destino.name.lower().endswith(f".{extensao}"):
            destino = destino.with_name(f"{destino.name}.{extensao}")
        gravado = salvar.gravar(arquivo.origem, destino)
        self._ultima_pasta = gravado.parent
        log.info("Arquivo salvo pela janela: %s", gravado)
        return {"status": "salvo", "caminho": str(gravado)}

    def _pasta_inicial(self) -> Path:
        if self._ultima_pasta and self._ultima_pasta.is_dir():
            return self._ultima_pasta
        downloads = Path.home() / "Downloads"
        return downloads if downloads.is_dir() else Path.home()


def _tipo_salvar():
    """`FileDialog.SAVE` (pywebview 5+/6); cai para `SAVE_DIALOG` em versões antigas."""
    import webview

    fd = getattr(webview, "FileDialog", None)
    return fd.SAVE if fd is not None else webview.SAVE_DIALOG


def main() -> None:
    token = secrets.token_urlsafe(24)
    os.environ["ESCRIBA_SESSION_TOKEN"] = token   # lido pelo app.main
    log_file = _setup_logging()

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
        api = JsApi()
        janela = webview.create_window("Escriba — transcrição de reuniões", url,
                                       width=1280, height=860, min_size=(900, 600),
                                       js_api=api)
        api._janela = janela   # o diálogo de salvar sai por aqui
        gui = "qt" if sys.platform.startswith("linux") else None   # Windows: WebView2
        webview.start(gui=gui, private_mode=False)

    server.should_exit = True


if __name__ == "__main__":
    main()
