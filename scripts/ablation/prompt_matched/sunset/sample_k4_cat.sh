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

target_root="results/ablation/prompt_matched/sunset/sample_k4_cat/diffusion_backprop"
first4_root="results/prompt_matched/sunset/first4_sample_k4_cat/diffusion_backprop"
last3_root="results/prompt_matched/sunset/last3_sample_k4_cat/diffusion_backprop"
mkdir -p "${target_root}"

copy_sample_dir() {
  local label="$1"
  local src="$2"
  local dst="$3"
  if [[ -d "${src}" ]]; then
    mkdir -p "$(dirname "${dst}")"
    mkdir -p "${dst}"
    cp -a "${src}/." "${dst}/"
    echo "[copy sync] ${label}: ${src} -> ${dst}"
  else
    echo "[copy warn] ${label}: missing source ${src}" >&2
  fi
}

sync_reference_case() {
  local label="$1"
  local source_case="$2"
  local dest_case="$3"
  copy_sample_dir "${label} samp_0p00016" "${first4_root}/${source_case}/cs/item_000/samp_0p00016" "${target_root}/${dest_case}/cs/item_000/samp_0p00016"
  copy_sample_dir "${label} samp_0p00031" "${first4_root}/${source_case}/cs/item_000/samp_0p00031" "${target_root}/${dest_case}/cs/item_000/samp_0p00031"
  copy_sample_dir "${label} samp_0p00063" "${first4_root}/${source_case}/cs/item_000/samp_0p00063" "${target_root}/${dest_case}/cs/item_000/samp_0p00063"
  copy_sample_dir "${label} samp_0p00125" "${first4_root}/${source_case}/cs/item_000/samp_0p00125" "${target_root}/${dest_case}/cs/item_000/samp_0p00125"
  copy_sample_dir "${label} samp_0p00250" "${last3_root}/${source_case}/cs/item_000/samp_0p00250" "${target_root}/${dest_case}/cs/item_000/samp_0p00250"
  copy_sample_dir "${label} samp_0p00500" "${last3_root}/${source_case}/cs/item_000/samp_0p00500" "${target_root}/${dest_case}/cs/item_000/samp_0p00500"
  copy_sample_dir "${label} samp_0p01000" "${last3_root}/${source_case}/cs/item_000/samp_0p01000" "${target_root}/${dest_case}/cs/item_000/samp_0p01000"
}

sync_reference_case "unconditioned reference" "sample_k4_cat__recover_unprompted" "reference_unconditioned"
sync_reference_case "CFG 7.5 reference" "sample_k4_cat__recover_prompt_sunset_beach" "reference_cfg7p5"

"${PYTHON_BIN}" run_conditioning_regression.py \
  --suite-config configs/ablation/prompt_matched/sunset/sample_k4_cat_suite.json \
  --dc-methods diffusion_backprop \
  --sampling-methods cs
