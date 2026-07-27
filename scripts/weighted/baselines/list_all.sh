#!/usr/bin/env bash
set -euo pipefail

if [[ -f ./run_conditioning_regression.py ]]; then
  root_dir="."
else
  script_dir="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
  root_dir="$(cd -P -- "$script_dir/../../.." && pwd -P)"
fi
cd "$root_dir"

families=(prompt_matched prompt_mismatched out_of_range)
schemes=(mcs inverse_square vdhh)
recoveries=(unprompted daytime_beach sunset_beach cat)
splits=(first3 last2)

job_count=0
expected_rows=0
declare -A computer_jobs=()
declare -A computer_rows=()
declare -A method_rows=()
for scheme in "${schemes[@]}"; do
  echo "=== $scheme ==="
  launcher="./scripts/weighted/baselines/run_${scheme}_split.sh"
  for family_index in "${!families[@]}"; do
    family="${families[$family_index]}"
    for recovery_index in "${!recoveries[@]}"; do
      recovery="${recoveries[$recovery_index]}"
      combination_index=$((family_index * ${#recoveries[@]} + recovery_index))
      for split in "${splits[@]}"; do
        if [[ "$split" == "first3" ]]; then
          computer=$((107 + combination_index % 6))
          rows=15
        else
          computer=$((107 + (combination_index + 3) % 6))
          rows=10
        fi
        command=("$launcher" "$family" "$split" "$recovery")
        printf 'class%s: ' "$computer"
        printf '%q ' "${command[@]}"
        printf '\n'
        "${command[@]}" --dry-run
        job_count=$((job_count + 1))
        expected_rows=$((expected_rows + rows))
        computer_jobs[$computer]=$(( ${computer_jobs[$computer]:-0} + 1 ))
        computer_rows[$computer]=$(( ${computer_rows[$computer]:-0} + rows ))
        method_rows[$scheme]=$(( ${method_rows[$scheme]:-0} + rows ))
      done
    done
  done
done

echo "validated_jobs=$job_count expected_rows=$expected_rows"
if [[ "$job_count" -ne 72 || "$expected_rows" -ne 900 ]]; then
  echo "baseline job-grid validation failed" >&2
  exit 1
fi
for scheme in "${schemes[@]}"; do
  echo "method=$scheme expected_rows=${method_rows[$scheme]:-0}"
  if [[ "${method_rows[$scheme]:-0}" -ne 300 ]]; then
    echo "unexpected method row count for $scheme" >&2
    exit 1
  fi
done
for computer in 107 108 109 110 111 112; do
  echo "class$computer jobs=${computer_jobs[$computer]:-0} expected_rows=${computer_rows[$computer]:-0}"
  if [[ "${computer_jobs[$computer]:-0}" -ne 12 || "${computer_rows[$computer]:-0}" -ne 150 ]]; then
    echo "unexpected job allocation for class$computer" >&2
    exit 1
  fi
done
echo "per_method_rows=300 per_computer_rows=150"
