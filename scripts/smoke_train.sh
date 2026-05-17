#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/venv/.sigmoidz}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/venv/.sigmoidz/bin/python}"

$PYTHON_BIN src/train.py \
  --preset tiny \
  --max_steps "${MAX_STEPS:-20}" \
  --out_dir "${OUT_DIR:-runs/tiny}" \
  "$@"
