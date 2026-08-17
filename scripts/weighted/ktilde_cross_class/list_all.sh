#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

for first_class in ca uc db; do
  "$SCRIPT_DIR/run.sh" "$first_class" --dry-run
done

echo "Validated 3 ordered cross-class S10000 jobs"
