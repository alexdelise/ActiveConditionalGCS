#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

./build_k0_unconditioned.sh "$@"
./build_k1_daytime_beach.sh "$@"
./build_k2_sunset_beach.sh "$@"
./build_k4_cat.sh "$@"
