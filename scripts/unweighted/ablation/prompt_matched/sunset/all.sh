#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
"$SCRIPT_DIR/sample_k0_unconditioned.sh" "$@"
"$SCRIPT_DIR/sample_k1_daytime_beach.sh" "$@"
"$SCRIPT_DIR/sample_k2_sunset_beach.sh" "$@"
"$SCRIPT_DIR/sample_k4_cat.sh" "$@"
