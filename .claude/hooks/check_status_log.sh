#!/usr/bin/env bash
# Stop hook: keep PROJECT_STATUS.md current.
#
# Blocks Claude from finishing a turn when a file in the git working tree
# (tracked and changed, or untracked) was modified after PROJECT_STATUS.md,
# and tells it to update the status log first. Allows the stop when nothing
# changed, when the log is already newer than every change, or when Claude is
# already continuing because of this hook (stop_hook_active), so it never loops.

set -euo pipefail

input="$(cat)"
if [ "$(printf '%s' "$input" | jq -r '.stop_hook_active // false')" = "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel)}"
status="PROJECT_STATUS.md"

block() {
  jq -n --arg reason "$1" '{decision: "block", reason: $reason}'
  exit 0
}

if [ ! -f "$status" ]; then
  block "PROJECT_STATUS.md is missing. Create it (see the Project status log section of CLAUDE.md) before finishing."
fi

newer=()
while IFS= read -r line; do
  path="${line:3}"
  path="${path##* -> }"
  path="${path%\"}"
  path="${path#\"}"
  case "$path" in
    "$status" | runs/* | *.pyc | */__pycache__/*) continue ;;
  esac
  if [ -f "$path" ] && [ "$path" -nt "$status" ]; then
    newer+=("$path")
  fi
done < <(git status --porcelain --untracked-files=all)

if [ "${#newer[@]}" -gt 0 ]; then
  list="$(printf '%s, ' "${newer[@]:0:8}")"
  block "These files changed after PROJECT_STATUS.md was last updated: ${list%, }. Update PROJECT_STATUS.md (status, current performance, what was done, assumptions, next steps) to reflect this turn's work, then finish."
fi
exit 0
