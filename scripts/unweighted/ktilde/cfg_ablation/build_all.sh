#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

./build_cfg1_all.sh "$@"
./build_cfg3_all.sh "$@"
./build_cfg5_all.sh "$@"
