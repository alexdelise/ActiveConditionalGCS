#!/usr/bin/env bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
runner="$script_dir/run_shard.sh"
commands=0
reconstructions=0

for scenario in prompt_matched prompt_mismatched out_of_range; do
  for law in k0 k1 k2 k4; do
    for line in unconditioned cfg1 cfg1p5 cfg3 cfg5 cfg7p5; do
      for split in first3 last2; do
        "$runner" "$scenario" "$law" "$line" "$split" --dry-run
        commands=$((commands + 1))
        if [[ "$split" == "first3" ]]; then
          reconstructions=$((reconstructions + 6))
        else
          reconstructions=$((reconstructions + 4))
        fi
      done
    done
  done
done

printf 'validated_commands=%d\nexpected_reconstructions=%d\n' "$commands" "$reconstructions"
