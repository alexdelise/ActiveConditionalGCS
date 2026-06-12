#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

./first4_sample_k0_unconditioned.sh
./last3_sample_k0_unconditioned.sh
