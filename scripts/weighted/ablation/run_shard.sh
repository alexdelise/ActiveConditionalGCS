#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 <prompt_matched|prompt_mismatched|out_of_range> <k0|k1|k2|k4> <unconditioned|cfg1|cfg1p5|cfg3|cfg5|cfg7p5> <first3|last2|all> [--dry-run]" >&2
  exit 2
fi

scenario="$1"
law="$2"
line="$3"
split="$4"
mode="${5:-}"

case "$scenario" in prompt_matched|prompt_mismatched|out_of_range) ;; *) echo "unsupported scenario: $scenario" >&2; exit 2 ;; esac
case "$law" in k0|k1|k2|k4) ;; *) echo "unsupported sampling law: $law" >&2; exit 2 ;; esac
case "$line" in unconditioned|cfg1|cfg1p5|cfg3|cfg5|cfg7p5) ;; *) echo "unsupported recovery line: $line" >&2; exit 2 ;; esac
case "$split" in first3|last2|all) ;; *) echo "unsupported split: $split" >&2; exit 2 ;; esac
if [[ -n "$mode" && "$mode" != "--dry-run" ]]; then
  echo "unsupported option: $mode" >&2
  exit 2
fi

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
project_root="$(cd -P "$script_dir/../../.." && pwd -P)"
cd -P "$project_root"

case "$law" in
  k0) prefix="sample_k0_unconditioned" ;;
  k1) prefix="sample_k1_daytime_beach" ;;
  k2) prefix="sample_k2_sunset_beach" ;;
  k4) prefix="sample_k4_cat" ;;
esac
if [[ "$line" == "unconditioned" ]]; then
  case_name="${prefix}__recover_unprompted"
else
  case_name="${prefix}__recover_prompt_sunset_beach_${line}"
fi

rates=""
case "$split" in
  first3) rates="0.01,0.02,0.03"; expected=6 ;;
  last2) rates="0.04,0.05"; expected=4 ;;
  all) expected=10 ;;
esac

python_bin="${PYTHON_BIN:-python}"
suite="configs/weighted/ablation/${scenario}/${law}_suite.json"
results_root="results/weighted/ablation/${scenario}"
command=(
  "$python_bin" run_conditioning_regression.py
  --suite-config "$suite"
  --sampling-methods cs
  --cases "$case_name"
  --results-root "$results_root"
)
if [[ -n "$rates" ]]; then
  command+=(--sampling-percentages "$rates")
fi

if [[ "$mode" == "--dry-run" ]]; then
  printf 'scenario=%s\nsampling_law=%s\nrecovery_line=%s\nsplit=%s\ncase=%s\nexpected_reconstructions=%d\n' \
    "$scenario" "$law" "$line" "$split" "$case_name" "$expected"
  "${command[@]}" --list-cases
  exit 0
fi

exec "${command[@]}"
