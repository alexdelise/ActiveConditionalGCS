#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" run_ktilde_convergence.py \
  --config ktilde/config_convergence.json \
  --name Ktilde_SD15__fft__k0_512x512_S10000_ns20 \
  "$@"
