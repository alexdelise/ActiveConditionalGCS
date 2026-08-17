#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <ca|uc|db> [--dry-run] [--force]" >&2
  exit 2
fi

LEFT_CLASS="$1"
shift

case "$LEFT_CLASS" in
  ca) STEM="k4cat" ;;
  uc) STEM="k0unconditioned" ;;
  db) STEM="k1daytimebeach" ;;
  *) echo "Unknown first class '$LEFT_CLASS'; expected ca, uc, or db" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -P -- "$SCRIPT_DIR/../../.." && pwd)"
CONFIG="ktilde/weighted/config_cross_class_s10000.json"
NAME="Ktilde_SD15__fft__cross_${STEM}_minus_k2sunsetbeach_512x512_S10000_ns20"
OUTPUT_DIR="ktilde/weighted"

DRY_RUN=false
FORWARD_ARGS=()
for argument in "$@"; do
  if [[ "$argument" == "--dry-run" ]]; then
    DRY_RUN=true
  else
    FORWARD_ARGS+=("$argument")
  fi
done

if [[ "$DRY_RUN" == true ]]; then
  printf 'first_class=%s\nsecond_class=sb\nconfig=%s\nname=%s\noutput_dir=%s\n' \
    "$LEFT_CLASS" "$CONFIG" "$NAME" "$OUTPUT_DIR"
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
cd -P -- "$PROJECT_ROOT"
exec "$PYTHON_BIN" build_ktilde.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --name "$NAME" \
  "${FORWARD_ARGS[@]}"
