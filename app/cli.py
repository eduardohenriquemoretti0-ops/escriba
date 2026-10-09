"""CLI do escriba — uso imediato: transcrever um áudio (e opcionalmente gerar ata).

Exemplos:
    python -m app.cli transcrever reuniao.m4a
    python -m app.cli transcrever reuniao.m4a --ata --saida ~/atas
    python -m app.cli transcrever reuniao.m4a --modelo medium --idioma pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import process_audio
from .transcribe.base import ProviderUnavailableError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="escriba", description="Transcrição de áudio + ata.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcrever", help="Transcreve um arquivo de áudio.")
    t.add_argument("audio", type=Path, help="Caminho do arquivo de áudio/vídeo.")
    t.add_argument("--saida", type=Path, default=None, help="Diretório de saída.")
    t.add_argument("--ata", action="store_true", help="Também gerar a ata (via Claude).")
    t.add_argument("--sem-ata", action="store_true", help="Não gerar ata.")
    t.add_argument("--idioma", default=None, help="Idioma (ex.: pt). Padrão: auto.")
    t.add_argument("--modelo", default=None, help="Modelo Whisper (tiny..large-v3).")

    args = parser.parse_args(argv)

    if args.cmd == "transcrever":
        if args.modelo:
            import os
            os.environ["ESCRIBA_WHISPER_MODEL"] = args.modelo
        make_ata: bool | None = None
        if args.ata:
            make_ata = True
        if args.sem_ata:
            make_ata = False

        if not args.audio.exists():
            print(f"erro: arquivo não encontrado: {args.audio}", file=sys.stderr)
            return 2
        try:
            res = process_audio(
                args.audio, out_dir=args.saida, make_ata=make_ata, language=args.idioma
            )
        except ProviderUnavailableError as e:
            print(f"erro de transcrição: {e}", file=sys.stderr)
            return 1

        print("\n=== TRANSCRIÇÃO ===")
        print(res.transcript[:2000] + ("..." if len(res.transcript) > 2000 else ""))
        print(f"\n[salvo] {res.transcript_path}")
        if res.ata_path:
            print(f"[salvo] {res.ata_path}")
        if res.warnings:
            print("\nAvisos:")
            for w in res.warnings:
                print(f"  - {w}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
