#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

torchrun \
  --nproc_per_node="$NPROC_PER_NODE" \
  src/train.py \
  --config "${CONFIG:-configs/50m_sigmoidz.json}" \
  "$@"
