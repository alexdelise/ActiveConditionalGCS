#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "usage: $0 <k0|k1|k2|k4|mcs|inverse_square> <unprompted|daytime_beach|sunset_beach|cat> [all|first3|last2] [--dry-run]" >&2
  exit 2
fi

law="$1"
recovery="$2"
split="all"
mode=""
for argument in "${@:3}"; do
  case "$argument" in
    all|first3|last2) split="$argument" ;;
    --dry-run) mode="$argument" ;;
    *) echo "unknown option: $argument" >&2; exit 2 ;;
  esac
done
case "$law" in
  k0|k1|k2|k4|mcs|inverse_square) ;;
  *) echo "unsupported sampling law: $law" >&2; exit 2 ;;
esac
case "$recovery" in
  unprompted|daytime_beach|sunset_beach|cat) ;;
  *) echo "unsupported recovery prompt: $recovery" >&2; exit 2 ;;
esac

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
project_root="$(cd -P "$script_dir/../../../.." && pwd -P)"
cd -P "$project_root"

case "$law" in
  k0) prefix="sample_k0_unconditioned__recover_" ;;
  k1) prefix="sample_k1_daytime_beach__recover_" ;;
  k2) prefix="sample_k2_sunset_beach__recover_" ;;
  k4) prefix="sample_k4_cat__recover_" ;;
  mcs) prefix="baseline_mcs__recover_" ;;
  inverse_square) prefix="baseline_inverse_square__recover_" ;;
esac
if [[ "$law" == k0 || "$law" == k1 || "$law" == k2 || "$law" == k4 ]]; then
  if [[ "$recovery" == "unprompted" ]]; then
    case_name="${prefix}unprompted"
  else
    case_name="${prefix}prompt_${recovery}"
  fi
  sampling_method="cs"
else
  case_name="${prefix}${recovery}"
  sampling_method="$law"
fi

python_bin="${PYTHON_BIN:-python}"
suite="configs/weighted/diagnostics/out_of_range_fixed_learning_rate/${law}_suite.json"
results_root="results/weighted/diagnostics/out_of_range_fixed_learning_rate"
command=(
  "$python_bin" run_conditioning_regression.py
  --suite-config "$suite"
  --sampling-methods "$sampling_method"
  --cases "$case_name"
  --results-root "$results_root"
)
case "$split" in
  all) expected=25 ;;
  first3)
    command+=(--sampling-percentages "0.01,0.02,0.03")
    expected=15
    ;;
  last2)
    command+=(--sampling-percentages "0.04,0.05")
    expected=10
    ;;
esac

if [[ "$mode" == "--dry-run" ]]; then
  printf 'sampling_law=%s\nrecovery=%s\nsplit=%s\ncase=%s\nexpected_reconstructions=%d\n' \
    "$law" "$recovery" "$split" "$case_name" "$expected"
  "${command[@]}" --list-cases
  exit 0
fi

exec "${command[@]}"
