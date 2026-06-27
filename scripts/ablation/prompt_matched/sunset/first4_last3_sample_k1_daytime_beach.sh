#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

./first4_sample_k1_daytime_beach.sh "$@"
./last3_sample_k1_daytime_beach.sh "$@"
