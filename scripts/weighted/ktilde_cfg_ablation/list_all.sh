#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

count=0
for cfg in 1 3 5; do
  for prior in k1 k2 k4; do
    "$SCRIPT_DIR/run.sh" "$prior" "$cfg" --dry-run
    count=$((count + 1))
  done
done

echo "Validated $count CFG-ablation S10000 jobs"
