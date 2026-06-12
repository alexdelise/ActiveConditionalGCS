#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

./first4_sample_k0_unconditioned.sh
./first4_sample_k1_daytime_beach.sh
./first4_sample_k2_sunset_beach.sh
./first4_sample_k4_cat.sh
