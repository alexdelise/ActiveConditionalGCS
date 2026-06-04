#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

./sample_k0_unconditioned.sh
./sample_k1_daytime_beach.sh
./sample_k2_sunset_beach.sh
./sample_k4_cat.sh
