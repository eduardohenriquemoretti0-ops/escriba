# PyInstaller: gera dist/Escriba/ (Escriba.exe no Windows, Escriba no Linux).
# Use:  python build.py   (roda isto e copia os modelos para junto do programa)
import os
import sys
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = [("app/static", "app/static"), ("app/fonts", "app/fonts")]   # fonts: PDF de reserva
binaries, hiddenimports = [], []
for pkg in ("faster_whisper", "ctranslate2", "sherpa_onnx", "onnxruntime",
            "webview", "weasyprint", "pydyf", "tinycss2", "cssselect2", "pyphen",
            "tokenizers", "av", "fpdf"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h
hiddenimports += collect_submodules("app") + collect_submodules("uvicorn")
hiddenimports += ["assemblyai", "anthropic", "markdown", "PIL", "faster_whisper.vad"]

excludes = ["torch", "torchaudio", "whisperx", "pyannote", "tkinter", "matplotlib"]
# ícone só se o arquivo existir (sem ele o PyInstaller no Windows aborta)
_ICONE = os.path.join(SPECPATH, "escriba.ico")
if sys.platform != "linux":
    excludes += ["PyQt6", "qtpy"]

a = Analysis(["escriba_app.py"], datas=datas, binaries=binaries,
             hiddenimports=hiddenimports, excludes=excludes)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Escriba",
          console=False,
          icon=_ICONE if sys.platform == "win32" and os.path.exists(_ICONE) else None)
coll = COLLECT(exe, a.binaries, a.datas, name="Escriba")
