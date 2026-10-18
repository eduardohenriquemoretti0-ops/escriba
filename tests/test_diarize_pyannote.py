"""Falantes vindos dos turnos do pyannote: frases inteiras, corte só com evidência."""
import numpy as np

from app.engines import diarize
from app.engines.diarize import Word


class _Ex:
    def embed(self, samples):
        v = np.ones(4, dtype=np.float32)
        return v / np.linalg.norm(v)


def _palavras(texto, ini, passo=0.4):
    out, t = [], ini
    for p in texto.split():
        out.append(Word(t, t + passo - 0.05, " " + p))
        t += passo
    return out


AUDIO = np.zeros(diarize.SAMPLE_RATE * 60, dtype=np.float32)


def test_frase_inteira_fica_com_o_falante_dominante():
    ws = _palavras("bom dia irmãos sejam todos bem vindos.", 0.0)
    # fronteira do pyannote cai no meio da última palavra: não parte a frase
    turnos = [(0.0, 2.6, "SPK_1"), (2.6, 10.0, "SPK_2")]
    frases, _ = diarize.diarize_por_turnos(AUDIO, ws, _Ex(), turnos)
    assert len(frases) == 1 and frases[0].speaker == 0


def test_troca_real_de_falante_dentro_da_frase_parte():
    ws = _palavras("obrigado irmão agora quem lê é o irmão Wellington lendo", 0.0)
    turnos = [(0.0, 2.0, "A"), (2.0, 10.0, "B")]
    frases, cent = diarize.diarize_por_turnos(AUDIO, ws, _Ex(), turnos)
    assert [f.speaker for f in frases] == [0, 1]
    assert set(cent) == {0, 1}


def test_rotulos_na_ordem_de_aparicao_e_sem_turno_herda_anterior():
    ws = _palavras("primeira frase aqui.", 0.0) + _palavras("segunda frase longe.", 20.0)
    turnos = [(0.0, 2.0, "SPK_9")]
    frases, _ = diarize.diarize_por_turnos(AUDIO, ws, _Ex(), turnos)
    assert [f.speaker for f in frases] == [0, 0]


def test_disponivel_respeita_onnx(monkeypatch):
    from app import config
    from app.engines import pyannote_diar
    monkeypatch.setattr(config, "DIARIZADOR", "onnx")
    assert pyannote_diar.disponivel()[0] is False
