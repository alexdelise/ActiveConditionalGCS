#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 ]]; then
  echo "usage: $0 <main|ablation> <prompt_matched|prompt_mismatched|out_of_range> <sampling-prior> [runner args...]" >&2
  exit 2
fi

mode="$1"
family="$2"
prior="$3"
shift 3

if [[ -f ./run_conditioning_regression.py ]]; then
  ROOT_DIR="."
else
  SCRIPT_DIR="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
  ROOT_DIR="$(cd -P -- "$SCRIPT_DIR/../.." && pwd -P)"
fi
cd "$ROOT_DIR"

scripts/unweighted/run_split.sh "$mode" "$family" first4 "$prior" "$@"
scripts/unweighted/run_split.sh "$mode" "$family" last3 "$prior" "$@"
