@echo off
rem Abre o Escriba (programa de desktop). Primeira vez: cria o ambiente e baixa os modelos.
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Primeira execucao: preparando o ambiente ^(precisa de internet so agora^)...
  py -3 -m venv .venv || python -m venv .venv
  ".venv\Scripts\python.exe" -m pip install -q --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt
)
if not exist "models\whisper\large-v3-turbo\model.bin" ".venv\Scripts\python.exe" baixar_modelos.py
if not exist "models\diarizacao\nemo_en_titanet_small.onnx" ".venv\Scripts\python.exe" baixar_modelos.py
start "" ".venv\Scripts\pythonw.exe" escriba_app.py
