#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

./run_out_of_range_first4_all.sh
./run_out_of_range_last3_all.sh
