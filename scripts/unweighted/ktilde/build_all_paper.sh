#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/main/build_all.sh" "$@"
"$SCRIPT_DIR/cfg_ablation/build_all.sh" "$@"
