#!/usr/bin/env bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
runner="$script_dir/run_shard.sh"
count=0
for law in k0 k1 k2 k4 mcs inverse_square; do
  for recovery in unprompted daytime_beach sunset_beach cat; do
    "$runner" "$law" "$recovery" --dry-run
    count=$((count + 25))
  done
done
printf 'validated_shards=24\nexpected_reconstructions=%d\n' "$count"
