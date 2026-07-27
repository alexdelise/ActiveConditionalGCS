#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 4 ]]; then
  echo "usage: $0 <mcs|inverse_square|vdhh> <family> <first3|last2> <recovery> [--dry-run] [runner args...]" >&2
  exit 2
fi

scheme="$1"
family="$2"
split="$3"
recovery="$4"
shift 4

case "$scheme" in
  mcs|inverse_square|vdhh) ;;
  *) echo "unsupported baseline sampling scheme: $scheme" >&2; exit 2 ;;
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
case "$recovery" in
  unprompted|daytime_beach|sunset_beach|cat) ;;
  *) echo "unsupported recovery prompt: $recovery" >&2; exit 2 ;;
esac

dry_run=false
runner_args=()
for argument in "$@"; do
  if [[ "$argument" == "--dry-run" ]]; then
    dry_run=true
  else
    runner_args+=("$argument")
  fi
done

# Stay in an already-open checkout when possible; this also works on class
# machines whose physical mount cannot be traversed again from BASH_SOURCE.
if [[ -f ./run_conditioning_regression.py && -d ./configs/weighted ]]; then
  root_dir="."
else
  script_dir="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
  root_dir="$(cd -P -- "$script_dir/../../.." && pwd -P)"
fi
if [[ ! -f "$root_dir/run_conditioning_regression.py" ]]; then
  echo "could not locate the ActiveConditionalGCS project root" >&2
  exit 2
fi
cd "$root_dir"

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

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

suite_config="configs/weighted/baselines/${family}/sunset/${scheme}_suite.json"
case_name="baseline_${scheme}__recover_${recovery}"
result_tag="weighted/${family}/sunset/${split}_${scheme}_recover_${recovery}"

command=(
  "$PYTHON_BIN"
  run_conditioning_regression.py
  --suite-config "$suite_config"
  --tag "$result_tag"
  --cases "$case_name"
  --sampling-methods "$scheme"
  --sampling-percentages "$sampling_percentages"
)
if [[ "$dry_run" == true ]]; then
  echo "scheme=$scheme family=$family split=$split recovery=$recovery"
  echo "rates=$sampling_percentages tag=$result_tag"
  "${command[@]}" --list-cases
  exit 0
fi

exec "${command[@]}" "${runner_args[@]}"
