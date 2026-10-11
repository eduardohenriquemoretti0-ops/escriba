@echo off
rem Gera o Escriba.exe: rodar numa maquina Windows 10/11 com Python 3.11 ou 3.12 e internet.
rem (3.13 ainda nao foi verificado; o script avisa, mas segue.)
rem Saida: dist\Escriba\Escriba.exe e dist\Escriba-windows-amd64.zip
setlocal
cd /d "%~dp0"

rem Trava: refazer dist\ com o Escriba aberto quebra o programa em uso
tasklist /FI "IMAGENAME eq Escriba.exe" 2>nul | find /I "Escriba.exe" >nul
if not errorlevel 1 (
  echo.
  echo Feche o Escriba antes de gerar o build ^(Escriba.exe esta em execucao^).
  goto erro
)

rem Python: prefere 3.12, depois 3.11, depois o padrao do launcher
set "PY=py -3"
py -3.11 -V >nul 2>&1 && set "PY=py -3.11"
py -3.12 -V >nul 2>&1 && set "PY=py -3.12"
echo Usando: %PY%
if not exist ".venv-build\Scripts\python.exe" (
  %PY% -m venv .venv-build
  if errorlevel 1 (
    echo Nao consegui criar o ambiente .venv-build com %PY%. Verifique a instalacao do Python.
    goto erro
  )
)
".venv-build\Scripts\python.exe" -c "import sys; sys.exit(0 if sys.version_info[:2] in ((3, 11), (3, 12)) else 1)"
if errorlevel 1 echo AVISO: o Python deste ambiente nao e 3.11 nem 3.12 ^(3.13 nao foi verificado^).

echo.
echo [1/3] Atualizando o pip...
".venv-build\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto erro

echo.
echo [2/3] Instalando dependencias (requirements.txt) e PyInstaller 6.22.3...
".venv-build\Scripts\python.exe" -m pip install -r requirements.txt pyinstaller==6.22.3
if errorlevel 1 goto erro

echo.
echo [3/3] Gerando o programa (build.py: modelos, PyInstaller e zip)...
".venv-build\Scripts\python.exe" build.py
if errorlevel 1 goto erro

if not exist "dist\Escriba\Escriba.exe" (
  echo Nao encontrei dist\Escriba\Escriba.exe. Veja as mensagens acima.
  goto erro
)
echo.
echo Pronto: dist\Escriba\Escriba.exe  ^(entregue a pasta inteira ou o .zip^)
pause
exit /b 0

:erro
echo.
echo *** Falhou. Veja as mensagens acima. ***
pause
exit /b 1
