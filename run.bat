@echo off
rem Abre o Escriba (programa de desktop). Primeira vez: cria o ambiente e baixa os modelos.
rem Os erros do pip ficam na tela (sem -q): se algo falhar, a mensagem aparece aqui.
setlocal
cd /d "%~dp0"

if exist ".venv\pronto.ok" goto modelos
if exist ".venv\Scripts\python.exe" goto instalar
echo Primeira execucao: preparando o ambiente ^(precisa de internet so agora^)...
py -3 -m venv .venv
if errorlevel 1 python -m venv .venv
if errorlevel 1 (
  echo Nao consegui criar o ambiente .venv. Verifique a instalacao do Python ^(3.11 ou 3.12^).
  goto erro
)

:instalar
echo.
echo Instalando dependencias ^(pode demorar^)...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto erro
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo O pip falhou ao instalar requirements.txt. Veja o erro acima.
  goto erro
)
type nul > ".venv\pronto.ok"

:modelos
if not exist "models\whisper\large-v3-turbo\model.bin" ".venv\Scripts\python.exe" baixar_modelos.py
if errorlevel 1 goto erro
if not exist "models\diarizacao\nemo_en_titanet_small.onnx" ".venv\Scripts\python.exe" baixar_modelos.py
if errorlevel 1 goto erro

rem so o programa roda sem console (pythonw); o resto usa python.exe para mostrar erros
start "" ".venv\Scripts\pythonw.exe" escriba_app.py
exit /b 0

:erro
echo.
echo *** Nao foi possivel abrir o Escriba. Veja as mensagens acima. ***
pause
exit /b 1
