"""Escriba como programa de desktop.

Abre o Escriba numa janela própria — sem navegador, sem precisar de internet.
É o ponto de entrada do executável (Escriba.exe / Escriba no Linux) e do
`run` (run.bat / run.sh).

Por dentro, a interface continua sendo a mesma página web, servida por um
servidor embutido que só escuta em 127.0.0.1, numa porta aleatória, e só
responde a quem tiver o token desta sessão (a própria janela). Nenhuma outra
máquina da rede — nem outra aba de navegador — consegue usá-lo.

No Windows a janela usa o Microsoft Edge WebView2. Sem ele, o pywebview cairia
no IE11 (mshtml) e a página quebraria sem aviso; por isso a checagem é feita
antes de abrir a janela (`webview2_instalado`).
"""
from __future__ import annotations

import logging
import os
import platform
import secrets
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable

log = logging.getLogger("escriba")

# Evergreen Bootstrapper do WebView2 (link oficial da Microsoft)
WEBVIEW2_LINK = "https://go.microsoft.com/fwlink/p/?LinkId=2124703"
# As mesmas chaves que webview.platforms.winforms._is_chromium usa (pywebview 6.2.1)
_WEBVIEW2_CLIENTES = (
    "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}",   # Runtime
    "{2CD8A007-E189-409D-A2C8-9AF4EF3C72AA}",   # Beta
    "{0D50BFEC-CD6A-4F9A-964C-C7416E3ACB10}",   # Dev
    "{65C35B14-6C1D-4122-AC46-7148CC9D6497}",   # Canary
)
_WEBVIEW2_MINIMO = (86, 0, 622, 0)
_DOTNET_462 = 394802                             # valor "Release" do .NET 4.6.2

_MB_OK = 0x0
_MB_SIM_NAO = 0x4
_MB_INFORMACAO = 0x40
_MB_AVISO = 0x30
_MB_TOPO = 0x40000
_IDYES = 6


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
    # Sem proxy: o servidor é 127.0.0.1, e um proxy configurado no sistema
    # faria a checagem falhar até estourar o tempo.
    abrir = urllib.request.build_opener(urllib.request.ProxyHandler({})).open
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with abrir(url, timeout=1):
                return
        except Exception:  # noqa: BLE001 — servidor ainda subindo
            time.sleep(0.2)
    raise RuntimeError("O servidor interno do Escriba não respondeu.")


# --------------------------------------------------------------------------
# WebView2 (Windows)
# --------------------------------------------------------------------------
def _ler_registro(raiz: str, caminho: str, valor: str) -> object:
    """Valor de uma chave do registro do Windows; None se a chave não existir."""
    import winreg

    try:
        with winreg.OpenKey(getattr(winreg, raiz), caminho) as chave:
            return winreg.QueryValueEx(chave, valor)[0]
    except OSError:
        return None


def _versao_minima(texto: str) -> bool:
    try:
        return tuple(int(p) for p in texto.strip().split(".")) >= _WEBVIEW2_MINIMO
    except ValueError:
        return False


def webview2_instalado(ler: Callable[[str, str, str], object] = _ler_registro) -> bool:
    """Diz se o pywebview vai conseguir abrir a janela com o WebView2.

    Replica a checagem de `winforms._is_chromium`: .NET 4.6.2 ou mais e o
    WebView2 com versão >= 86.0.622.0, em HKCU e em HKLM (WOW6432Node).

    Args:
        ler: Leitor do registro; injetável para teste.
    """
    release = ler("HKEY_LOCAL_MACHINE",
                  r"SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full", "Release")
    if not isinstance(release, int) or release < _DOTNET_462:
        return False
    x86 = platform.machine().lower() == "x86"
    for clsid in _WEBVIEW2_CLIENTES:
        for raiz in ("HKEY_CURRENT_USER", "HKEY_LOCAL_MACHINE"):
            if raiz == "HKEY_CURRENT_USER" or x86:
                caminho = rf"SOFTWARE\Microsoft\EdgeUpdate\Clients\{clsid}"
            else:
                caminho = rf"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{clsid}"
            versao = ler(raiz, caminho, "pv")
            if isinstance(versao, str) and _versao_minima(versao):
                return True
    return False


def _caixa(titulo: str, texto: str, botoes: int, icone: int) -> int:
    """Caixa de mensagem nativa do Windows (MessageBoxW). Devolve o botão escolhido."""
    import ctypes

    return ctypes.windll.user32.MessageBoxW(None, texto, titulo, botoes | icone | _MB_TOPO)


def _pedir_webview2() -> bool:
    """Sem WebView2: mostra o link do instalador e pergunta se abre no navegador."""
    texto = (
        "O Escriba precisa do Microsoft Edge WebView2 Runtime para abrir a janela, "
        "e ele não foi encontrado neste computador.\n\n"
        "Instale pelo link oficial da Microsoft (Evergreen Bootstrapper):\n"
        f"{WEBVIEW2_LINK}\n\n"
        "Depois, abra o Escriba de novo.\n\n"
        "Quer abrir o Escriba agora no navegador? Ele continua funcionando sem internet."
    )
    return _caixa("Escriba: falta o WebView2", texto, _MB_SIM_NAO, _MB_AVISO) == _IDYES


# --------------------------------------------------------------------------
# Modos de abertura
# --------------------------------------------------------------------------
def _modo_navegador(url: str, base: str) -> None:
    """Abre no navegador padrão e mantém o servidor vivo até o usuário encerrar."""
    import webbrowser

    webbrowser.open(url)
    if sys.platform == "win32":
        # pythonw / .exe não têm console: a caixa de OK é o "encerrar"
        _caixa("Escriba",
               f"O Escriba está aberto no navegador ({base}).\n\n"
               "Clique em OK para encerrar o programa.", _MB_OK, _MB_INFORMACAO)
        return
    print(f"Escriba aberto no navegador. Feche esta janela para encerrar.\n{base}")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


def _modo_janela(webview, url: str, base: str) -> None:
    api = JsApi()
    janela = webview.create_window("Escriba — transcrição de reuniões", url,
                                   width=1280, height=860, min_size=(900, 600),
                                   js_api=api)
    api._janela = janela   # o diálogo de salvar sai por aqui
    # Windows: forçado para o WebView2; o pywebview só o usa se webview2_instalado()
    gui = "qt" if sys.platform.startswith("linux") else "edgechromium"
    webview.start(gui=gui, private_mode=False)


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
        # o nome padrão também passa pela sanitização (pode vir de um upload)
        nome = salvar.nome_seguro(nome_sugerido or arquivo.nome, extensao, arquivo.nome)
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
    import webview

    return webview.FileDialog.SAVE


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

    usar_navegador = webview is None or os.getenv("ESCRIBA_NAVEGADOR") == "1"
    if not usar_navegador and sys.platform == "win32" and not webview2_instalado():
        log.error("WebView2 Runtime não encontrado; oferecendo o navegador (%s)", WEBVIEW2_LINK)
        if not _pedir_webview2():
            server.should_exit = True
            return
        usar_navegador = True

    if usar_navegador:
        # Sem componente de janela (ou escolhido): usa o navegador padrão, offline
        _modo_navegador(url, base)
    else:
        # Downloads passam pelo JsApi.salvar_arquivo (ver index.html); o WebView não baixa nada
        webview.settings["ALLOW_DOWNLOADS"] = False
        webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True
        try:
            _modo_janela(webview, url, base)
        except Exception:  # noqa: BLE001 — registra a falha e não deixa o usuário sem tela
            log.exception("A janela do Escriba falhou (log: %s); abrindo no navegador.", log_file)
            _modo_navegador(url, base)

    server.should_exit = True


if __name__ == "__main__":
    main()
