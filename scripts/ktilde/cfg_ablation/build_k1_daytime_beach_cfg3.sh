#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/../build_named.sh" \
  ktilde/config_cfg_ablation.json \
  Ktilde_SD15__fft__k1daytimebeach_cfg3_512x512_S500_ns20 \
  "$@"
