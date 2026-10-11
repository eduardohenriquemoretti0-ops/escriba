"""Programa de desktop (escriba_app.py): WebView2, proxy e nome de salvar.

Nada abre janela: o registro do Windows é simulado por um dicionário, e o
servidor de teste escuta só em 127.0.0.1.
"""
from __future__ import annotations

import http.server
import os
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import escriba_app  # noqa: E402
from app import config, jobs, salvar  # noqa: E402

_DOTNET = ("HKEY_LOCAL_MACHINE",
           r"SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full", "Release")
_RUNTIME_HKCU = ("HKEY_CURRENT_USER",
                 r"SOFTWARE\Microsoft\EdgeUpdate\Clients"
                 r"\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}", "pv")


def test_webview2_ok_com_dotnet_e_runtime_no_usuario():
    dados = {_DOTNET: 528040, _RUNTIME_HKCU: "130.0.2849.80"}
    assert escriba_app.webview2_instalado(lambda r, c, v: dados.get((r, c, v)))


def test_webview2_ausente_retorna_falso():
    assert not escriba_app.webview2_instalado(lambda r, c, v: None)


def test_webview2_versao_antiga_nao_conta():
    dados = {_DOTNET: 528040, _RUNTIME_HKCU: "85.0.622.0"}
    assert not escriba_app.webview2_instalado(lambda r, c, v: dados.get((r, c, v)))


def test_sem_dotnet_462_pywebview_nao_usa_chromium():
    dados = {_DOTNET: 393295, _RUNTIME_HKCU: "130.0.0.0"}   # .NET 4.6.0
    assert not escriba_app.webview2_instalado(lambda r, c, v: dados.get((r, c, v)))


def test_runtime_instalado_so_em_hklm_wow6432(monkeypatch):
    monkeypatch.setattr(escriba_app.platform, "machine", lambda: "AMD64")
    caminho = (r"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients"
               r"\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}")
    dados = {_DOTNET: 528040, ("HKEY_LOCAL_MACHINE", caminho, "pv"): "120.0.0.1"}
    assert escriba_app.webview2_instalado(lambda r, c, v: dados.get((r, c, v)))


def test_versao_minima_compara_por_componente():
    assert escriba_app._versao_minima("86.0.622.0")
    assert escriba_app._versao_minima("120.0.2210.91")
    assert not escriba_app._versao_minima("86.0.621.9")
    assert not escriba_app._versao_minima("lixo")


def test_wait_ready_ignora_proxy_do_sistema(monkeypatch):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):  # silencia o log do teste
            pass

    servidor = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    try:
        # um proxy inexistente no ambiente não pode atrapalhar a checagem local
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
        monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
        url = f"http://127.0.0.1:{servidor.server_address[1]}/api/health"
        escriba_app._wait_ready(url, timeout=5)
    finally:
        servidor.shutdown()
        servidor.server_close()


def test_tipo_salvar_usa_filedialog_save():
    import webview

    assert escriba_app._tipo_salvar() == webview.FileDialog.SAVE


class _JanelaFalsa:
    def __init__(self):
        self.nome_pedido = None

    def create_file_dialog(self, tipo, directory, save_filename, file_types):
        self.nome_pedido = save_filename
        return None                      # usuário cancelou: só interessa o nome sugerido


def test_nome_padrao_tambem_e_sanitizado(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    origem = tmp_path / "ata.pdf"
    origem.write_bytes(b"%PDF-1.4 teste")
    arquivo = salvar.Arquivo(origem=origem, extensao="pdf", nome="rel:atorio?.pdf")
    monkeypatch.setattr(salvar, "preparar", lambda job_id, kind: arquivo)

    janela = _JanelaFalsa()
    api = escriba_app.JsApi()
    api._janela = janela

    resposta = api.salvar_arquivo("a" * 12, "ata_pdf", "")

    assert resposta == {"status": "cancelado"}
    assert janela.nome_pedido == "rel_atorio_.pdf"
