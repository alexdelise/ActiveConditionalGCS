#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 4 ]]; then
  echo "usage: $0 <main|ablation> <prompt_matched|prompt_mismatched|out_of_range> <first4|last3> <sampling-prior> [runner args...]" >&2
  exit 2
fi

mode="$1"
family="$2"
split="$3"
prior="$4"
shift 4

case "$mode" in
  main|ablation) ;;
  *) echo "unsupported unweighted mode: $mode" >&2; exit 2 ;;
esac
case "$family" in
  prompt_matched|prompt_mismatched|out_of_range) ;;
  *) echo "unsupported unweighted experiment family: $family" >&2; exit 2 ;;
esac
case "$split" in
  first4) sampling_percentages="0.00015625,0.0003125,0.000625,0.00125" ;;
  last3) sampling_percentages="0.0025,0.005,0.01" ;;
  *) echo "unsupported unweighted split: $split" >&2; exit 2 ;;
esac

prior="${prior#sample_}"
case "$prior" in
  unprompted|k0) prior="k0_unconditioned" ;;
  daytime_beach|k1) prior="k1_daytime_beach" ;;
  sunset_beach|k2) prior="k2_sunset_beach" ;;
  cat|k4) prior="k4_cat" ;;
  k0_unconditioned|k1_daytime_beach|k2_sunset_beach|k4_cat) ;;
  *) echo "unsupported unweighted sampling prior: $prior" >&2; exit 2 ;;
esac

if [[ -f ./run_conditioning_regression.py ]]; then
  ROOT_DIR="."
else
  SCRIPT_DIR="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
  ROOT_DIR="$(cd -P -- "$SCRIPT_DIR/../.." && pwd -P)"
fi
if [[ ! -f "$ROOT_DIR/run_conditioning_regression.py" || ! -d "$ROOT_DIR/configs/unweighted" ]]; then
  echo "could not find run_conditioning_regression.py and configs/unweighted in the project root" >&2
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
config_prefix="configs/unweighted"
result_prefix="unweighted"
if [[ "$mode" == "ablation" ]]; then
  config_prefix="${config_prefix}/ablation"
  result_prefix="${result_prefix}/ablation"
  sync_args=()
  list_only=false
  for runner_arg in "$@"; do
    if [[ "$runner_arg" == "--list-cases" ]]; then
      list_only=true
      break
    fi
  done
  if [[ "$list_only" == false ]]; then
    sync_args+=(--copy)
  fi
  "${PYTHON_BIN}" scripts/unweighted/sync_ablation_references.py \
    --family "$family" \
    --prior "$prior_tag" \
    --split "$split" \
    "${sync_args[@]}"
fi

exec "${PYTHON_BIN}" run_conditioning_regression.py \
  --suite-config "${config_prefix}/${family}/sunset/${split}_${prior_tag}_suite.json" \
  --tag "${result_prefix}/${family}/sunset/${split}_${prior_tag}" \
  --sampling-methods cs \
  --sampling-percentages "$sampling_percentages" \
  "$@"
