#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

./build_k1_daytime_beach_cfg3.sh "$@"
./build_k2_sunset_beach_cfg3.sh "$@"
./build_k4_cat_cfg3.sh "$@"
