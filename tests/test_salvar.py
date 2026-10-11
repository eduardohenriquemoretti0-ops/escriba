"""Testes do salvamento pela janela desktop (app/salvar.py e JsApi).

Não abrem diálogo: o caminho de destino é injetado, e a janela é um objeto
falso que devolve a escolha pronta. Os jobs ficam num diretório temporário.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import config, jobs, review, salvar  # noqa: E402
from app.engines.base import Segment  # noqa: E402


@pytest.fixture
def dados(tmp_path, monkeypatch):
    """Aponta os dados do Escriba para um diretório temporário."""
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(config, "PROFILE_DIR", tmp_path / "profile")
    (tmp_path / "jobs").mkdir()
    return tmp_path


def _job_concluido() -> jobs.Job:
    job = jobs.new_job(filename="reuniao_acme.wav", engine="local", language="pt",
                       min_speakers=None, max_speakers=None, want_summary=False,
                       participants=[])
    job.status = jobs.STATUS_DONE
    job.save()
    review.save_raw(job, [
        Segment(0.0, 4.0, "Falante 1", "Bom dia a todos, vamos começar a reunião."),
        Segment(4.5, 9.0, "Falante 2", "Perfeito, obrigado por reservar a sala."),
    ])
    job.path("transcricao.txt").write_text(
        "Falante 1: Bom dia a todos.\nFalante 2: Perfeito.\n", encoding="utf-8")
    return job


def test_grava_pdf_de_transcricao_com_cabecalho(dados, tmp_path):
    job = _job_concluido()
    destino = tmp_path / "saida" / "transcricao.pdf"
    destino.parent.mkdir()

    gravado = salvar.salvar_em(job.id, "transcricao_pdf", destino)

    assert gravado == destino
    conteudo = destino.read_bytes()
    assert len(conteudo) > 1000
    assert conteudo[:5] == b"%PDF-"


def test_grava_txt_nao_vazio(dados, tmp_path):
    job = _job_concluido()
    destino = tmp_path / "transcricao.txt"

    salvar.salvar_em(job.id, "txt", destino)

    texto = destino.read_text(encoding="utf-8")
    assert texto.strip()
    assert "Falante 1" in texto


def test_nao_deixa_temporario_para_tras(dados, tmp_path):
    job = _job_concluido()
    salvar.salvar_em(job.id, "txt", tmp_path / "a.txt")
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(".")) == []


def test_rejeita_kind_fora_da_lista_branca(dados, tmp_path):
    job = _job_concluido()
    with pytest.raises(salvar.SalvarError, match="Tipo de arquivo inválido"):
        salvar.salvar_em(job.id, "../../etc/passwd", tmp_path / "x")
    with pytest.raises(salvar.SalvarError, match="Tipo de arquivo inválido"):
        salvar.salvar_em(job.id, "meta", tmp_path / "x")


@pytest.mark.parametrize("job_id", ["../escriba", "B6B000BBF451", "abc", "", None, 123])
def test_rejeita_job_id_com_formato_errado(dados, tmp_path, job_id):
    with pytest.raises(salvar.SalvarError, match="Identificador"):
        salvar.salvar_em(job_id, "txt", tmp_path / "x.txt")


def test_job_inexistente_vira_erro_amigavel(dados, tmp_path):
    with pytest.raises(salvar.SalvarError, match="não encontrado"):
        salvar.salvar_em("000000000000", "txt", tmp_path / "x.txt")


def test_pasta_inexistente_nao_grava(dados, tmp_path):
    job = _job_concluido()
    with pytest.raises(salvar.SalvarError, match="A pasta não existe"):
        salvar.salvar_em(job.id, "txt", tmp_path / "nao_existe" / "x.txt")


def test_destino_que_e_pasta_nao_grava(dados, tmp_path):
    job = _job_concluido()
    with pytest.raises(salvar.SalvarError, match="é uma pasta"):
        salvar.salvar_em(job.id, "txt", tmp_path)


def test_erro_ao_gravar_nao_sobrescreve_arquivo_existente(dados, tmp_path):
    job = _job_concluido()
    existente = tmp_path / "ok.txt"
    existente.write_text("conteúdo anterior", encoding="utf-8")
    origem = job.path("transcricao.txt")
    origem.write_bytes(b"")   # origem vazia: a gravação deve falhar

    with pytest.raises(salvar.SalvarError, match="vazio"):
        salvar.gravar(origem, existente)

    assert existente.read_text(encoding="utf-8") == "conteúdo anterior"
    assert not (tmp_path / ".ok.txt.part").exists()


@pytest.mark.parametrize("sugerido, esperado", [
    ("../../../etc/passwd", "passwd.pdf"),
    ("C:\\Users\\x\\evil.exe", "evil.pdf"),
    ("reuniao\x00\x07 final?.pdf", "reuniao__ final_.pdf"),
    ("", "padrao.pdf"),
    ("...", "padrao.pdf"),
    ("CON", "_CON.pdf"),
    ("relatório de março", "relatório de março.pdf"),
])
def test_nome_seguro_remove_caminho_e_controle(sugerido, esperado):
    assert salvar.nome_seguro(sugerido, "pdf", "padrao.pdf") == esperado


def test_nome_seguro_forca_extensao_do_servidor():
    assert salvar.nome_seguro("planilha.exe", "txt", "x.txt") == "planilha.txt"


def test_nome_seguro_limita_tamanho():
    assert len(salvar.nome_seguro("a" * 500, "txt", "x.txt")) <= 124


class _JanelaFalsa:
    """Imita Window.create_file_dialog do pywebview: devolve a escolha pronta."""

    def __init__(self, escolha):
        self.escolha = escolha
        self.chamadas = []

    def create_file_dialog(self, tipo, directory="", allow_multiple=False,
                           save_filename="", file_types=()):
        self.chamadas.append({"tipo": tipo, "directory": directory,
                              "save_filename": save_filename, "file_types": file_types})
        return self.escolha


def _api(janela):
    from escriba_app import JsApi
    api = JsApi()
    api._janela = janela
    return api


def test_js_api_salvo_so_depois_de_gravar(dados, tmp_path):
    job = _job_concluido()
    destino = tmp_path / "escolhido.pdf"
    janela = _JanelaFalsa((str(destino),))

    r = _api(janela).salvar_arquivo(job.id, "transcricao_pdf", "../lixo")

    assert r == {"status": "salvo", "caminho": str(destino)}
    assert destino.read_bytes()[:5] == b"%PDF-"
    chamada = janela.chamadas[0]
    assert chamada["save_filename"] == "lixo.pdf"
    assert ".." not in chamada["save_filename"] and "/" not in chamada["save_filename"]


def test_js_api_acrescenta_extensao_quando_usuario_nao_digita(dados, tmp_path):
    job = _job_concluido()
    janela = _JanelaFalsa((str(tmp_path / "sem_extensao"),))

    r = _api(janela).salvar_arquivo(job.id, "txt", "")

    assert r["status"] == "salvo"
    assert r["caminho"].endswith("sem_extensao.txt")
    assert (tmp_path / "sem_extensao.txt").exists()


def test_js_api_cancelado_nao_grava_nada(dados, tmp_path):
    job = _job_concluido()
    r = _api(_JanelaFalsa(None)).salvar_arquivo(job.id, "txt", "")
    assert r == {"status": "cancelado"}
    assert list(tmp_path.glob("*.txt")) == []


def test_js_api_valida_antes_de_abrir_dialogo(dados):
    janela = _JanelaFalsa(("/nunca",))
    r = _api(janela).salvar_arquivo("zz", "txt", "")
    assert r["status"] == "erro"
    assert janela.chamadas == []   # nenhum diálogo para pedido inválido


def test_js_api_sem_janela_devolve_erro(dados):
    from escriba_app import JsApi
    job = _job_concluido()
    r = JsApi().salvar_arquivo(job.id, "txt", "")
    assert r["status"] == "erro"
    assert "não está pronta" in r["mensagem"]


def test_js_api_expoe_so_o_metodo_publico():
    from escriba_app import JsApi
    publicos = [n for n in dir(JsApi) if not n.startswith("_")]
    assert publicos == ["salvar_arquivo"]
