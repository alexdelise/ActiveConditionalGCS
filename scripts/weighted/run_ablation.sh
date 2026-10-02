#!/usr/bin/env bash
set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
exec "${PYTHON_BIN:-python}" "$script_dir/run_experiment.py" ablation "$@"
