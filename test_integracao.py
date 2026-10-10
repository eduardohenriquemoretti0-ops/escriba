"""Teste de integração do Escriba (sem chaves de API).

Sobe a app com TestClient, envia um WAV real de 5 s, substitui o engine por
um fake que devolve segmentos conhecidos e verifica: fila, conversão ffmpeg,
merge, exports (TXT/SRT/VTT/JSON), timeline e endpoints de download.

Os recursos da v2 (nomes, correções, assinatura, PDFs, e-mail) são testados
em test_v2.py.
"""
import os
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

# dados do teste ficam num diretório temporário, nunca no data/ de verdade
_TMP = tempfile.mkdtemp(prefix="escriba_test_")
os.environ["ESCRIBA_DATA_DIR"] = _TMP

from fastapi.testclient import TestClient  # noqa: E402

from app.engines.base import Segment  # noqa: E402
from app.main import app  # noqa: E402

BASE = Path(__file__).parent


def gerar_wav(path: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
         "-ac", "2", "-ar", "44100", str(path)],
        check=True, capture_output=True,
    )


FAKE_SEGMENTS = [
    Segment(0.0, 1.2, "A", "Bom dia, pessoal, vamos começar."),
    Segment(1.4, 2.0, "A", "Pauta de hoje é o cronograma."),
    Segment(2.3, 3.5, "B", "Perfeito, eu atualizo a planilha até sexta."),
    Segment(3.8, 5.0, "A", "Fechado, então."),
]


def fake_transcribe(self, audio_path, *, language, min_speakers,
                    max_speakers, progress, vocabulary=None):
    assert Path(audio_path).exists() and Path(audio_path).stat().st_size > 0
    progress("Transcrevendo (fake)")
    return list(FAKE_SEGMENTS)


def main() -> None:
    wav = Path(_TMP) / "_teste.wav"
    gerar_wav(wav)

    with patch("app.engines.assemblyai_engine.AssemblyAIEngine.transcribe",
               fake_transcribe):
        with TestClient(app) as client:  # dispara o startup (worker)
            # upload
            with wav.open("rb") as f:
                r = client.post(
                    "/api/jobs",
                    files={"file": ("reuniao_teste.wav", f, "audio/wav")},
                    data={"engine": "assemblyai", "language": "pt",
                          "min_speakers": "2", "max_speakers": "2"},
                )
            assert r.status_code == 200, r.text
            job_id = r.json()["id"]

            # formato rejeitado
            r2 = client.post("/api/jobs",
                             files={"file": ("nota.pdf", b"x", "application/pdf")})
            assert r2.status_code == 415, "extensão inválida deveria dar 415"

            # aguarda o worker
            for _ in range(60):
                job = client.get(f"/api/jobs/{job_id}").json()
                if job["status"] in ("concluido", "erro"):
                    break
                time.sleep(0.5)
            assert job["status"] == "concluido", job
            assert job["duration"] and 4.5 < job["duration"] < 5.5
            assert job["speakers"] == ["Falante 1", "Falante 2"]
            assert len(job["timeline"]) == 3, "merge deveria juntar as 2 falas de A"

            # downloads
            txt = client.get(f"/api/jobs/{job_id}/file/txt").text
            assert "Falante 1 · 00:00:00" in txt
            assert "Bom dia, pessoal, vamos começar. Pauta de hoje é o cronograma." in txt
            srt = client.get(f"/api/jobs/{job_id}/file/srt").text
            assert "00:00:02,300 --> 00:00:03,500" in srt
            vtt = client.get(f"/api/jobs/{job_id}/file/vtt").text
            assert vtt.startswith("WEBVTT") and "<v Falante 2>" in vtt
            js = client.get(f"/api/jobs/{job_id}/file/json").json()
            assert js["segmentos"][1]["falante"] == "Falante 2"
            assert client.get(f"/api/jobs/{job_id}/file/ata").status_code == 404

            # UI servida
            home = client.get("/")
            assert home.status_code == 200 and "Escriba" in home.text

            # exclusão
            assert client.delete(f"/api/jobs/{job_id}").json()["ok"] is True
            assert client.get(f"/api/jobs/{job_id}").status_code == 404

    print("TODOS OS TESTES PASSARAM ✔")


if __name__ == "__main__":
    main()
