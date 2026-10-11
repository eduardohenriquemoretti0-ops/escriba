"""Gravação de arquivos de dados no Windows: nova tentativa e mensagem amigável.

Também cobre a leitura de JSON com BOM (arquivo salvo pelo Bloco de Notas) e o
handler que devolve 409 em vez de 500 quando um arquivo está bloqueado.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import arquivo_seguro, config, irmaos, jobs, profile, vozes  # noqa: E402
from app import main as servidor  # noqa: E402

BOM = b"\xef\xbb\xbf"
_REPLACE_ORIGINAL = Path.replace


@pytest.fixture
def dados(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(config, "PROFILE_DIR", tmp_path / "profile")
    monkeypatch.setattr(config, "DIARIZATION_MODEL", str(tmp_path / "nemo_novo.onnx"))
    monkeypatch.setattr(profile, "PROFILE_JSON", tmp_path / "profile" / "perfil.json")
    monkeypatch.setattr(arquivo_seguro.time, "sleep", lambda s: None)   # sem esperar no teste
    (tmp_path / "jobs").mkdir()
    (tmp_path / "profile").mkdir()
    return tmp_path


def _falha_n_vezes(monkeypatch, n: int) -> list[int]:
    """Faz `Path.replace` falhar `n` vezes com PermissionError (arquivo aberto)."""
    chamadas: list[int] = []

    def replace(self, destino):
        chamadas.append(1)
        if len(chamadas) <= n:
            raise PermissionError(13, "Arquivo em uso", str(self))
        return _REPLACE_ORIGINAL(self, destino)

    monkeypatch.setattr(Path, "replace", replace)
    return chamadas


def test_gravar_texto_tenta_de_novo_e_consegue(dados, monkeypatch):
    chamadas = _falha_n_vezes(monkeypatch, 2)
    alvo = dados / "nomes" / "nomes.json"

    arquivo_seguro.gravar_texto(alvo, '{"ok": true}')

    assert len(chamadas) == 3
    assert json.loads(alvo.read_text(encoding="utf-8")) == {"ok": True}


def test_gravar_texto_desiste_com_mensagem_amigavel(dados, monkeypatch):
    chamadas = _falha_n_vezes(monkeypatch, 99)
    alvo = dados / "nomes" / "nomes.json"

    with pytest.raises(arquivo_seguro.ArquivoBloqueado) as erro:
        arquivo_seguro.gravar_texto(alvo, "{}")

    assert len(chamadas) == arquivo_seguro.TENTATIVAS == 3
    assert "Feche o arquivo nomes.json" in str(erro.value)
    assert not alvo.with_name("nomes.json.tmp").exists()     # sem temporário para trás


def test_handler_responde_409_com_a_mensagem(dados):
    resposta = asyncio.run(servidor._arquivo_bloqueado(
        None, arquivo_seguro.ArquivoBloqueado("Feche o arquivo vozes.json.")))

    assert resposta.status_code == 409
    assert json.loads(resposta.body) == {"detail": "Feche o arquivo vozes.json."}


def test_vozes_modelo_antigo_vai_para_arquivo_novo_sem_sobrescrever(dados):
    arq = dados / "vozes" / "vozes.json"
    arq.parent.mkdir(parents=True)
    arq.write_text(json.dumps({"modelo": "nemo_antigo", "pessoas": {"Ana": {}}}),
                   encoding="utf-8")
    (dados / "vozes" / "vozes.nemo_antigo.json").write_text("backup anterior", encoding="utf-8")

    assert vozes.list_people() == []                       # começa do zero
    guardados = sorted(p.name for p in arq.parent.glob("vozes.nemo_antigo*.json"))
    assert "vozes.nemo_antigo.json" in guardados           # backup anterior intacto
    assert len(guardados) == 2                             # o antigo foi guardado, não apagado
    assert not arq.exists()


def test_vozes_le_json_com_bom(dados):
    arq = dados / "vozes" / "vozes.json"
    arq.parent.mkdir(parents=True)
    arq.write_bytes(BOM + json.dumps({"modelo": "nemo_novo", "pessoas": {
        "Ana": {"contribuicoes": [{"job": "j", "rotulo": "A", "vetor": [1.0, 0.0],
                                   "segundos": 3.0, "quando": 1.0}]}}},
        ensure_ascii=False).encode("utf-8"))

    assert [p["nome"] for p in vozes.list_people()] == ["Ana"]


def test_irmaos_le_json_com_bom_e_grava_sem_bom(dados):
    arq = dados / "nomes" / "nomes.json"
    arq.parent.mkdir(parents=True)
    arq.write_bytes(BOM + json.dumps({"nomes": [
        {"id": "a1", "nome": "Matheus", "observacao": "(1)", "grafias": []}]}).encode("utf-8"))

    assert [e.nome for e in irmaos.listar()] == ["Matheus"]
    irmaos.criar("José Eduardo")
    assert not arq.read_bytes().startswith(BOM)


def test_profile_le_json_com_bom(dados):
    profile.PROFILE_JSON.write_bytes(BOM + json.dumps(
        {"nome": "José Eduardo", "organizacao": "Congregação"}, ensure_ascii=False).encode())

    assert profile.load()["nome"] == "José Eduardo"


def test_jobs_le_meta_com_bom(dados):
    job = jobs.new_job(filename="reuniao.m4a", engine="local", language="pt",
                       min_speakers=None, max_speakers=None, want_summary=False)
    meta = job.path("meta.json")
    meta.write_bytes(BOM + meta.read_bytes())

    assert jobs.load(job.id).filename == "reuniao.m4a"
    assert [j.id for j in jobs.list_jobs()] == [job.id]
