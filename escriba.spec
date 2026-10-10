# PyInstaller: gera dist/Escriba/ (Escriba.exe no Windows, Escriba no Linux).
# Use:  python build.py   (roda isto e copia os modelos para junto do programa)
import sys
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = [("app/static", "app/static")]
binaries, hiddenimports = [], []
for pkg in ("faster_whisper", "ctranslate2", "sherpa_onnx", "onnxruntime",
            "webview", "weasyprint", "pydyf", "tinycss2", "cssselect2", "pyphen",
            "tokenizers", "av"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h
hiddenimports += collect_submodules("app") + collect_submodules("uvicorn")
hiddenimports += ["assemblyai", "anthropic", "markdown", "PIL"]

excludes = ["torch", "torchaudio", "whisperx", "pyannote", "tkinter", "matplotlib"]
if sys.platform != "linux":
    excludes += ["PyQt6", "qtpy"]

a = Analysis(["escriba_app.py"], datas=datas, binaries=binaries,
             hiddenimports=hiddenimports, excludes=excludes)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Escriba",
          console=False, icon="escriba.ico" if sys.platform == "win32" else None)
coll = COLLECT(exe, a.binaries, a.datas, name="Escriba")
