#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

./run_prompt_mismatched_in_range_first4_sample_k1_daytime_beach.sh
./run_prompt_mismatched_in_range_last3_sample_k1_daytime_beach.sh
