#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 ]]; then
  echo "usage: $0 <main|ablation> <prompt_matched|prompt_mismatched|out_of_range> <prior> [runner args...]" >&2
  exit 2
fi

mode="$1"
family="$2"
prior="$3"
shift 3

case "$mode" in
  main|ablation) ;;
  *) echo "unsupported weighted mode: $mode" >&2; exit 2 ;;
esac
case "$family" in
  prompt_matched|prompt_mismatched|out_of_range) ;;
  *) echo "unsupported weighted experiment family: $family" >&2; exit 2 ;;
esac
case "$prior" in
  k0_unconditioned|k1_daytime_beach|k2_sunset_beach|k4_cat) ;;
  *) echo "unsupported weighted sampling prior: $prior" >&2; exit 2 ;;
esac

ROOT_DIR="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif [[ -x "/opt/anaconda3/bin/python" ]]; then
    PYTHON_BIN="/opt/anaconda3/bin/python"
  elif command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="python3.12"
  else
    PYTHON_BIN="python3"
  fi
fi

prior_tag="sample_${prior}"
config_prefix="configs/weighted"
if [[ "$mode" == "ablation" ]]; then
  config_prefix="${config_prefix}/ablation"
  "${PYTHON_BIN}" scripts/weighted/sync_ablation_references.py \
    --family "$family" \
    --prior "$prior_tag" \
    --copy
fi

exec "${PYTHON_BIN}" run_conditioning_regression.py \
  --suite-config "${config_prefix}/${family}/sunset/${prior_tag}_suite.json" \
  --sampling-methods cs \
  "$@"
