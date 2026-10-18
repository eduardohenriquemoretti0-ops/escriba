"""Gera o programa pronto para entregar: dist/Escriba/ + modelos + .zip.

Rode no MESMO sistema do destino (o PyInstaller não faz build cruzado):
  - Windows -> dist/Escriba/Escriba.exe   (build_windows.bat faz tudo)
  - Linux   -> dist/Escriba/Escriba
"""
from __future__ import annotations

import csv
import io
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist" / "Escriba"


def _pids_linux() -> list[int]:
    exe = str(DIST / "Escriba")
    pids = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            # depois de um rebuild o link vira ".../Escriba (deleted)"
            alvo = os.readlink(proc / "exe").removesuffix(" (deleted)")
        except OSError:
            continue
        if alvo == exe:
            pids.append(int(proc.name))
    return pids


def _pids_windows() -> list[int]:
    """PIDs de Escriba.exe pelo tasklist (o Windows não expõe o caminho aqui)."""
    saida = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq Escriba.exe", "/FO", "CSV", "/NH"],
        capture_output=True, text=True, errors="replace", check=False).stdout
    pids = []
    for linha in csv.reader(io.StringIO(saida)):
        if len(linha) >= 2 and linha[0].lower() == "escriba.exe" and linha[1].isdigit():
            pids.append(int(linha[1]))
    return pids


def _programa_aberto() -> list[int]:
    """PIDs do Escriba empacotado em execução. Refazer o build por cima de um
    programa aberto quebra imports tardios dele com
    "Error -3 while decompressing data: incorrect header check"."""
    if sys.platform.startswith("win"):
        return _pids_windows()
    return _pids_linux()


def main() -> None:
    if pids := _programa_aberto():
        sys.exit(f"Feche o Escriba antes de gerar o build (aberto: PID {pids}).")
    subprocess.run([sys.executable, "baixar_modelos.py"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
                    "escriba.spec"], cwd=ROOT, check=True)
    # Modelos ficam AO LADO do programa (não dentro): grandes e substituíveis
    shutil.copytree(ROOT / "models" / "whisper" / "large-v3-turbo",
                    DIST / "models" / "whisper" / "large-v3-turbo", dirs_exist_ok=True)
    shutil.copytree(ROOT / "models" / "diarizacao", DIST / "models" / "diarizacao",
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("*.tar.bz2", "sherpa-onnx-*",
                                                  "wespeaker*"))
    # pyannote community-1 (CC-BY-4.0: o README com a atribuição vai junto)
    shutil.copytree(ROOT / "models" / "pyannote-community-1",
                    DIST / "models" / "pyannote-community-1", dirs_exist_ok=True)
    shutil.copy2(ROOT / ".env.example", DIST / ".env.example")
    shutil.copy2(ROOT / "LEIA-ME.txt", DIST / "LEIA-ME.txt")
    nome = f"Escriba-{platform.system().lower()}-{platform.machine().lower()}"
    arquivo = shutil.make_archive(str(ROOT / "dist" / nome), "zip", DIST.parent, "Escriba")
    print(f"\nPronto: {DIST}\nPacote: {arquivo}")


if __name__ == "__main__":
    main()
