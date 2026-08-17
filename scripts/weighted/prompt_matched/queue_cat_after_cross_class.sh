#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [--dry-run]" >&2
  exit 2
fi
mode="${1:-}"
if [[ -n "$mode" && "$mode" != "--dry-run" ]]; then
  echo "unsupported option: $mode" >&2
  exit 2
fi

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
project_root="$(cd -P "$script_dir/../../../../.." && pwd -P)"
cross_artifact="$project_root/ktilde/weighted/Ktilde_SD15__fft__cross_k0unconditioned_minus_k2sunsetbeach_512x512_S10000_ns20.npz"
runner="$project_root/scripts/weighted/prompt_matched/run_shard.sh"

sessions=(recon recon2 recon3 recon4)
laws=(k0 k1 k2 k4)

if [[ "$mode" == "--dry-run" ]]; then
  printf 'completion_gate=%s\n' "$cross_artifact"
  for index in "${!sessions[@]}"; do
    printf '%s: %s %s cat\n' "${sessions[$index]}" "$runner" "${laws[$index]}"
  done
  exit 0
fi

for session in "${sessions[@]}"; do
  if ! tmux has-session -t "$session" 2>/dev/null; then
    tmux new-session -d -s "$session"
  fi
done

echo "Waiting for the final cross-class artifact: $cross_artifact"
while [[ ! -f "$cross_artifact" ]]; do
  sleep 60
done

for index in "${!sessions[@]}"; do
  session="${sessions[$index]}"
  law="${laws[$index]}"
  while true; do
    pane_command="$(tmux display-message -p -t "$session" '#{pane_current_command}')"
    case "$pane_command" in
      bash|zsh|sh) break ;;
    esac
    echo "$session is still running $pane_command; waiting"
    sleep 60
  done
  launch="cd -P '$project_root' && conda activate testWH && '$runner' '$law' cat"
  tmux send-keys -t "$session" "$launch" Enter
  echo "Submitted $law / cat to $session"
done
