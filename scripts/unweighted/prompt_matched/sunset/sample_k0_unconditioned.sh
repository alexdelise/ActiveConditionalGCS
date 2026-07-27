#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec "$SCRIPT_DIR/../../run_suite.sh" main prompt_matched k0_unconditioned "$@"
