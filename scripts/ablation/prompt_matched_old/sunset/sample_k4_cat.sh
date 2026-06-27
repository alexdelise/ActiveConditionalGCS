#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif [[ -x "/opt/anaconda3/bin/python" ]]; then
    PYTHON_BIN="/opt/anaconda3/bin/python"
  elif command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="python3.12"
  elif command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="python3.11"
  else
    PYTHON_BIN="python3"
  fi
fi

target_root="results/ablation/prompt_matched_old/sunset/sample_k4_cat"
reference_root="results/prompt_matched_old/sunset/sample_k4_cat"
mkdir -p "${target_root}"

copy_reference() {
  local label="$1"
  local src="$2"
  local dst="$3"
  if [[ -d "${src}" ]]; then
    mkdir -p "${dst}"
    cp -a "${src}/." "${dst}/"
    echo "[copy sync] ${label}: ${src} -> ${dst}"
  else
    echo "[copy warn] ${label}: missing source ${src}" >&2
  fi
}

copy_reference \
  "unconditioned reference" \
  "${reference_root}/sample_k4_cat__recover_unprompted" \
  "${target_root}/reference_unconditioned"
copy_reference \
  "CFG 7.5 reference" \
  "${reference_root}/sample_k4_cat__recover_prompt_sunset_beach" \
  "${target_root}/reference_cfg7p5"

"${PYTHON_BIN}" run_conditioning_regression.py \
  --suite-config configs/ablation/prompt_matched_old/sunset/sample_k4_cat_suite.json \
  --sampling-methods cs
