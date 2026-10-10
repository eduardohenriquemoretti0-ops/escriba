"""Gera o programa pronto para entregar: dist/Escriba/ + modelos + .zip.

Rode no MESMO sistema do destino (o PyInstaller não faz build cruzado):
  - Windows -> dist/Escriba/Escriba.exe   (build_windows.bat faz tudo)
  - Linux   -> dist/Escriba/Escriba
"""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist" / "Escriba"


def main() -> None:
    subprocess.run([sys.executable, "baixar_modelos.py"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
                    "escriba.spec"], cwd=ROOT, check=True)
    # Modelos ficam AO LADO do programa (não dentro): grandes e substituíveis
    shutil.copytree(ROOT / "models" / "whisper" / "large-v3-turbo",
                    DIST / "models" / "whisper" / "large-v3-turbo", dirs_exist_ok=True)
    shutil.copytree(ROOT / "models" / "diarizacao", DIST / "models" / "diarizacao",
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("*.tar.bz2", "sherpa-onnx-*",
                                                  "3dspeaker*", "wespeaker*"))
    shutil.copy2(ROOT / ".env.example", DIST / ".env.example")
    shutil.copy2(ROOT / "LEIA-ME.txt", DIST / "LEIA-ME.txt")
    nome = f"Escriba-{platform.system().lower()}-{platform.machine().lower()}"
    arquivo = shutil.make_archive(str(ROOT / "dist" / nome), "zip", DIST.parent, "Escriba")
    print(f"\nPronto: {DIST}\nPacote: {arquivo}")


if __name__ == "__main__":
    main()
