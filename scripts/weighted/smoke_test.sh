#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif [[ -x "/opt/anaconda3/bin/python" ]]; then
    PYTHON_BIN="/opt/anaconda3/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

exec "$PYTHON_BIN" run_conditioning_regression.py \
  --suite-config configs/weighted/prompt_matched/sunset/sample_k0_unconditioned_suite.json \
  --tag weighted/smoke/prompt_matched/sunset/sample_k0_unconditioned \
  --cases sample_k0_unconditioned__recover_unprompted \
  --sampling-methods cs \
  --sampling-percentages 0.00125 \
  --repeats-per-setting 1 \
  "$@"
