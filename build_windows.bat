@echo off
rem Gera o Escriba.exe (rodar numa maquina Windows 10/11 com Python 3.11+ e internet).
cd /d "%~dp0"
py -3 -m venv .venv-build || python -m venv .venv-build
".venv-build\Scripts\python.exe" -m pip install --upgrade pip
".venv-build\Scripts\python.exe" -m pip install -r requirements.txt pyinstaller
".venv-build\Scripts\python.exe" build.py
echo.
echo Programa em dist\Escriba\Escriba.exe  (entregue a pasta inteira ou o .zip)
pause
