#!/usr/bin/env bash
set -euo pipefail

# run.sh - convenience script to bootstrap and run the Samisk Transkribering app
# Usage:
#   ./run.sh           # create venv (if necessary), install deps, run server
#   TORCH_CU=129 ./run.sh   # after venv created, install torch+torchaudio for CUDA 12.9

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

ensure_venv() {
  if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtualenv at $VENV_DIR"
    python -m venv "$VENV_DIR"
  fi
}

activate_venv() {
  # shellcheck source=/dev/null
  source "$VENV_DIR/bin/activate"
}

install_requirements() {
  echo "Upgrading pip and installing wheel/build tools"
  python -m pip install --upgrade pip setuptools wheel
  echo "Installing project requirements"
  python -m pip install -r "$ROOT_DIR/requirements.txt"
}

install_torch_cu129() {
  echo "Installing torch + torchaudio for CUDA 12.9"
  python -m pip install --index-url https://download.pytorch.org/whl/cu129 \
    --extra-index-url https://download.pytorch.org/whl/torch_stable.html \
    torch torchaudio --upgrade
}

run_server() {
  echo "Starting uvicorn using the venv python"
  python -m uvicorn app.main:app --reload
}

main() {
  ensure_venv
  activate_venv

  # Install base requirements (idempotent)
  install_requirements

  # Optional: install CUDA-specific torch if TORCH_CU is set to 129
  if [ "${TORCH_CU:-}" = "129" ]; then
    install_torch_cu129
  fi

  run_server
}

main "$@"
