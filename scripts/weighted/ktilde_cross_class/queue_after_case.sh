#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 || "$#" -gt 4 ]]; then
  echo "Usage: $0 <source-case-root> <destination-session> <ca|uc|db> [--dry-run]" >&2
  exit 2
fi

SOURCE_CASE_ROOT="$1"
DESTINATION_SESSION="$2"
LEFT_CLASS="$3"
MODE="${4:-}"

case "$LEFT_CLASS" in
  ca) ARTIFACT_STEM="k4cat" ;;
  uc) ARTIFACT_STEM="k0unconditioned" ;;
  db) ARTIFACT_STEM="k1daytimebeach" ;;
  *) echo "Unknown first class '$LEFT_CLASS'; expected ca, uc, or db" >&2; exit 2 ;;
esac
if [[ -n "$MODE" && "$MODE" != "--dry-run" ]]; then
  echo "Unknown option '$MODE'" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -P -- "$SCRIPT_DIR/../../.." && pwd)"
SOURCE_PATH="$PROJECT_ROOT/$SOURCE_CASE_ROOT"
ARTIFACT="$PROJECT_ROOT/ktilde/weighted/Ktilde_SD15__fft__cross_${ARTIFACT_STEM}_minus_k2sunsetbeach_512x512_S10000_ns20.npz"
LAUNCH_COMMAND="cd -P '$PROJECT_ROOT' && conda activate testWH && ./scripts/weighted/ktilde_cross_class/run.sh '$LEFT_CLASS'"

if ! tmux has-session -t "$DESTINATION_SESSION" 2>/dev/null; then
  echo "Missing destination tmux session '$DESTINATION_SESSION'" >&2
  exit 1
fi

if [[ "$MODE" == "--dry-run" ]]; then
  printf 'source=%s\nexpected_completed_rows=25\ndestination_session=%s\nartifact=%s\ncommand=%s\n' \
    "$SOURCE_PATH" "$DESTINATION_SESSION" "$ARTIFACT" "$LAUNCH_COMMAND"
  exit 0
fi

echo "Waiting for 25 completed reconstructions under $SOURCE_PATH"
last_count=-1
while true; do
  completed=0
  if [[ -d "$SOURCE_PATH" ]]; then
    completed="$(find "$SOURCE_PATH" -type f -name run_data.npz | wc -l)"
  fi
  if [[ "$completed" -ne "$last_count" ]]; then
    printf 'Completed source rows: %d / 25\n' "$completed"
    last_count="$completed"
  fi
  if [[ "$completed" -ge 25 ]]; then
    break
  fi
  sleep 60
done

if [[ -f "$ARTIFACT" ]]; then
  echo "Cross-class artifact already exists; no launch needed: $ARTIFACT"
  exit 0
fi

while true; do
  pane_command="$(tmux display-message -p -t "$DESTINATION_SESSION" '#{pane_current_command}')"
  case "$pane_command" in
    bash|zsh|sh) break ;;
  esac
  echo "Destination $DESTINATION_SESSION is still running $pane_command; waiting"
  sleep 60
done

tmux send-keys -t "$DESTINATION_SESSION" "$LAUNCH_COMMAND" Enter
echo "Submitted cross-class '$LEFT_CLASS' to tmux session '$DESTINATION_SESSION'"
