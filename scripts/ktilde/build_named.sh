#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: $0 <config-path> <ktilde-name> [build_ktilde.py arguments...]" >&2
  exit 2
fi

CONFIG_PATH="$1"
KTILDE_NAME="$2"
shift 2

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in python python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "Could not find Python. Set PYTHON_BIN to the environment interpreter." >&2
  exit 1
fi

"$PYTHON_BIN" build_ktilde.py \
  --config "$CONFIG_PATH" \
  --name "$KTILDE_NAME" \
  "$@"
