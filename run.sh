#!/usr/bin/env bash
set -euo pipefail

# run.sh - convenience script to bootstrap and run the Samisk Transkribering app
# Usage:
#   ./run.sh           # create venv (if necessary), install deps, run server
#   TORCH_CU=129 ./run.sh   # after venv created, install torch+torchaudio for CUDA 12.9

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

activate_venv() {
  if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "Virtualenv not found at $VENV_DIR. Create it manually and install requirements."
    exit 1
  fi
  # shellcheck source=/dev/null
  source "$VENV_DIR/bin/activate"
}

run_server() {
  echo "Exporting PYANNOTE_AUTH_TOKEN from ./hf_token if present"
  if [ -f "$ROOT_DIR/hf_token" ]; then
    token=$(sed -n '1p' "$ROOT_DIR/hf_token" | tr -d '\r\n')
    if [ -n "$token" ]; then
      export PYANNOTE_AUTH_TOKEN="$token"
      echo "PYANNOTE_AUTH_TOKEN set from hf_token"
    fi
  fi

  echo "Starting uvicorn using the venv python"
  python -m uvicorn app.main:app
}

main() {
  activate_venv
  run_server
}

main "$@"
