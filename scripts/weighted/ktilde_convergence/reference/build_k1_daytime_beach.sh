#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/build_named.sh" \
  Ktilde_SD15__fft__k1daytimebeach_512x512_S10000_ns20 \
  "$@"
