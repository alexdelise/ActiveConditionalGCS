#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

./run_prompt_mismatched_in_range_sample_k0_unconditioned.sh
./run_prompt_mismatched_in_range_sample_k1_daytime_beach.sh
./run_prompt_mismatched_in_range_sample_k2_sunset_beach.sh
./run_prompt_mismatched_in_range_sample_k4_cat.sh
