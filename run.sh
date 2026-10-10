#!/usr/bin/env bash
# Abre o Escriba (programa de desktop). Primeira vez: cria o ambiente e baixa os modelos.
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Primeira execução: preparando o ambiente (precisa de internet só agora)..."
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi
[[ -f models/diarizacao/nemo_en_titanet_small.onnx && -f models/whisper/large-v3-turbo/model.bin ]] \
  || .venv/bin/python baixar_modelos.py
exec .venv/bin/python escriba_app.py
