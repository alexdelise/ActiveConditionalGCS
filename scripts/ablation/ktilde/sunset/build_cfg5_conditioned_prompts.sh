#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in python python3.12 python3.11 python3.10 python3.9 python3.8 python3.7 python3; do
    if ! command -v "$candidate" >/dev/null 2>&1; then
      continue
    fi
    if "$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 7) else 1)
PY
    then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "Could not find a Python 3.7+ interpreter. Set PYTHON_BIN to your environment python." >&2
  exit 1
fi

CONFIG_PATH="ktilde/config_cfg_ablation.json"
names=(
  "Ktilde_SD15__fft__k1daytimebeach_cfg5_512x512_S500_ns20"
  "Ktilde_SD15__fft__k2sunsetbeach_cfg5_512x512_S500_ns20"
  "Ktilde_SD15__fft__k4cat_cfg5_512x512_S500_ns20"
)

for name in "${names[@]}"; do
  echo "[ktilde cfg5] ${name}"
  "$PYTHON_BIN" build_ktilde.py --config "$CONFIG_PATH" --name "$name" "$@"
done
