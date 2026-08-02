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
schemes=(mcs inverse_square)
recoveries=(unprompted daytime_beach sunset_beach cat)
splits=(first4 last3)

jobs=0
rows=0
for scheme in "${schemes[@]}"; do
  echo "=== $scheme ==="
  launcher="./scripts/unweighted/baselines/run_${scheme}_split.sh"
  for family in "${families[@]}"; do
    for recovery in "${recoveries[@]}"; do
      for split in "${splits[@]}"; do
        if [[ "$split" == "first4" ]]; then
          expected=20
        else
          expected=15
        fi
        printf '%q ' "$launcher" "$family" "$split" "$recovery"
        printf '\n'
        "$launcher" "$family" "$split" "$recovery" --dry-run
        jobs=$((jobs + 1))
        rows=$((rows + expected))
      done
    done
  done
done

echo "validated_jobs=$jobs expected_rows=$rows"
[[ "$jobs" -eq 48 && "$rows" -eq 840 ]]
