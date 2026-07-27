#!/usr/bin/env bash
set -euo pipefail

if [[ -f ./run_ktilde_convergence.py ]]; then
  ROOT_DIR="."
else
  SCRIPT_DIR="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
  ROOT_DIR="$(cd -P -- "$SCRIPT_DIR/../../.." && pwd -P)"
fi
cd "$ROOT_DIR"

for trial in 1 2 3 4 5; do
  for prior in k0 k1 k2 k4; do
    ./scripts/weighted/ktilde_convergence/run_trial.sh "$prior" "$trial" --dry-run
  done
done
