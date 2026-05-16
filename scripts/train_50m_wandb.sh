#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.sigmoidz}"
PYTHON_BIN="${PYTHON_BIN:-uv run python}"

$PYTHON_BIN src/train.py \
  --config "${CONFIG:-configs/50m_sigmoidz.json}" \
  --wandb \
  "$@"
