#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "usage: $0 <k0|k1|k2|k4> <trial-1-through-5> [runner arguments...]" >&2
  exit 2
fi

prior="$1"
trial="$2"
shift 2

case "$prior" in
  k0) reference_name="Ktilde_SD15__fft__k0_512x512_S10000_ns20" ;;
  k1) reference_name="Ktilde_SD15__fft__k1daytimebeach_512x512_S10000_ns20" ;;
  k2) reference_name="Ktilde_SD15__fft__k2sunsetbeach_512x512_S10000_ns20" ;;
  k4) reference_name="Ktilde_SD15__fft__k4cat_512x512_S10000_ns20" ;;
  *) echo "unsupported K-tilde prior alias: $prior" >&2; exit 2 ;;
esac

case "$trial" in
  1|2|3|4|5) ;;
  *) echo "trial must be one of: 1, 2, 3, 4, 5" >&2; exit 2 ;;
esac

# Stay inside an already-open physical checkout when possible. This avoids the
# stale-path behavior seen on some class machines while still allowing the
# launcher to resolve the repository when invoked elsewhere.
if [[ -f ./run_ktilde_convergence.py ]]; then
  ROOT_DIR="."
else
  SCRIPT_DIR="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
  ROOT_DIR="$(cd -P -- "$SCRIPT_DIR/../../.." && pwd -P)"
fi
if [[ ! -f "$ROOT_DIR/run_ktilde_convergence.py" ]]; then
  echo "could not find run_ktilde_convergence.py in the project root" >&2
  exit 2
fi
cd "$ROOT_DIR"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif [[ -x "/opt/anaconda3/bin/python" ]]; then
    PYTHON_BIN="/opt/anaconda3/bin/python"
  elif command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="python3.12"
  else
    PYTHON_BIN="python3"
  fi
fi

exec "$PYTHON_BIN" run_ktilde_convergence.py \
  --config ktilde/weighted/config_convergence.json \
  --trial-manifest ktilde/weighted/config_convergence_trials.json \
  --name "$reference_name" \
  --trial "$trial" \
  "$@"
