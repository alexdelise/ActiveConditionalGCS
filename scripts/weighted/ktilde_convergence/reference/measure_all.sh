#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

./measure_k0_unconditioned.sh "$@"
./measure_k1_daytime_beach.sh "$@"
./measure_k2_sunset_beach.sh "$@"
./measure_k4_cat.sh "$@"
