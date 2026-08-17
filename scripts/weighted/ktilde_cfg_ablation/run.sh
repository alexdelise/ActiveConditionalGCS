#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: $0 <k1|k2|k4> <1|3|5> [--dry-run] [--force]" >&2
  exit 2
fi

PRIOR="$1"
CFG="$2"
shift 2

case "$PRIOR" in
  k1) STEM="k1daytimebeach" ;;
  k2) STEM="k2sunsetbeach" ;;
  k4) STEM="k4cat" ;;
  *) echo "Unknown prior '$PRIOR'; expected k1, k2, or k4" >&2; exit 2 ;;
esac

case "$CFG" in
  1|3|5) ;;
  *) echo "Unknown sampling CFG '$CFG'; expected 1, 3, or 5" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -P -- "$SCRIPT_DIR/../../.." && pwd)"
CONFIG="ktilde/weighted/config_cfg_ablation_s10000.json"
NAME="Ktilde_SD15__fft__${STEM}_cfg${CFG}_512x512_S10000_ns20"
OUTPUT_DIR="ktilde/weighted"

DRY_RUN=false
FORWARD_ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    DRY_RUN=true
  else
    FORWARD_ARGS+=("$arg")
  fi
done

if [[ "$DRY_RUN" == true ]]; then
  printf 'project_root=%s\nconfig=%s\nname=%s\noutput_dir=%s\n' \
    "$PROJECT_ROOT" "$CONFIG" "$NAME" "$OUTPUT_DIR"
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
cd -P -- "$PROJECT_ROOT"
exec "$PYTHON_BIN" build_ktilde.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --name "$NAME" \
  "${FORWARD_ARGS[@]}"
