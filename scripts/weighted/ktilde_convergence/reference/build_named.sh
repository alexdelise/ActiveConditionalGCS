#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <ktilde-name> [build_ktilde.py arguments...]" >&2
  exit 2
fi

KTILDE_NAME="$1"
shift

SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -P "$SCRIPT_DIR/../../../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" build_ktilde.py \
  --config ktilde/weighted/config_convergence.json \
  --name "$KTILDE_NAME" \
  --output-dir ktilde/weighted \
  "$@"
