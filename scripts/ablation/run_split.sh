#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 ]]; then
  echo "usage: $0 <experiment-family> <first4|last3> <sample-prior> [runner args...]" >&2
  exit 2
fi

family="$1"
split="$2"
prior="$3"
shift 3

case "$family" in
  prompt_matched|prompt_mismatched|out_of_range) ;;
  *)
    echo "unsupported CFG-ablation experiment family: $family" >&2
    exit 2
    ;;
esac

case "$split" in
  first4|last3) ;;
  *)
    echo "unsupported CFG-ablation split: $split" >&2
    exit 2
    ;;
esac

case "$prior" in
  sample_k0_unconditioned|sample_k1_daytime_beach|sample_k2_sunset_beach|sample_k4_cat) ;;
  *)
    echo "unsupported CFG-ablation sampling prior: $prior" >&2
    exit 2
    ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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

source_root="results/${family}/sunset/${split}_${prior}"
target_root="results/ablation/${family}/sunset/${split}_${prior}"
mkdir -p "$target_root"

copy_reference_case() {
  local source_case="$1"
  local target_case="$2"
  local src="${source_root}/${source_case}"
  local dst="${target_root}/${target_case}"

  if [[ ! -d "$src" ]]; then
    echo "[copy warn] compatible reference is not available yet: $src" >&2
    return
  fi

  mkdir -p "$dst"
  cp -a "${src}/." "${dst}/"
  echo "[copy sync] ${src} -> ${dst}"
}

copy_reference_case "${prior}__recover_unprompted" "reference_unconditioned"
copy_reference_case "${prior}__recover_prompt_sunset_beach" "reference_cfg7p5"

"${PYTHON_BIN}" run_conditioning_regression.py \
  --suite-config "configs/ablation/${family}/sunset/${split}_${prior}_suite.json" \
  --sampling-methods cs \
  "$@"
