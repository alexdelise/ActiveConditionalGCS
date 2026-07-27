#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 4 ]]; then
  echo "usage: $0 <main|ablation> <prompt_matched|prompt_mismatched|out_of_range> <first3|last2> <sampling-prior> [runner args...]" >&2
  exit 2
fi

mode="$1"
family="$2"
split="$3"
prior="$4"
shift 4

case "$mode" in
  main|ablation) ;;
  *) echo "unsupported weighted mode: $mode" >&2; exit 2 ;;
esac
case "$family" in
  prompt_matched|prompt_mismatched|out_of_range) ;;
  *) echo "unsupported weighted experiment family: $family" >&2; exit 2 ;;
esac
case "$split" in
  first3) sampling_percentages="0.00125,0.0025,0.005" ;;
  last2) sampling_percentages="0.01,0.025" ;;
  *) echo "unsupported weighted split: $split" >&2; exit 2 ;;
esac

# Accept the same sample_k* spelling used by the original split launchers while
# keeping compatibility with weighted/run_suite.sh's internal prior names.
prior="${prior#sample_}"
case "$prior" in
  k0_unconditioned|k1_daytime_beach|k2_sunset_beach|k4_cat) ;;
  *) echo "unsupported weighted sampling prior: $prior" >&2; exit 2 ;;
esac

# Some class machines retain an open checkout whose absolute mount path cannot
# be traversed again. When launched from the project root, stay relative to the
# already-open directory instead of reconstructing it from $PWD or BASH_SOURCE.
if [[ -f ./run_conditioning_regression.py ]]; then
  ROOT_DIR="."
else
  SCRIPT_DIR="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
  ROOT_DIR="$(cd -P -- "$SCRIPT_DIR/../.." && pwd -P)"
fi
if [[ ! -f "$ROOT_DIR/run_conditioning_regression.py" || ! -d "$ROOT_DIR/configs/weighted" ]]; then
  echo "could not find run_conditioning_regression.py and configs/weighted in the project root" >&2
  exit 2
fi
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
result_prefix="weighted"
if [[ "$mode" == "ablation" ]]; then
  config_prefix="${config_prefix}/ablation"
  result_prefix="${result_prefix}/ablation"
  "${PYTHON_BIN}" scripts/weighted/sync_ablation_references.py \
    --family "$family" \
    --prior "$prior_tag" \
    --split "$split" \
    --copy
fi

exec "${PYTHON_BIN}" run_conditioning_regression.py \
  --suite-config "${config_prefix}/${family}/sunset/${prior_tag}_suite.json" \
  --tag "${result_prefix}/${family}/sunset/${split}_${prior_tag}" \
  --sampling-methods cs \
  --sampling-percentages "$sampling_percentages" \
  "$@"
