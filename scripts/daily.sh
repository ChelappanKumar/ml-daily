#!/usr/bin/env bash
# ml-daily — daily entrypoint called by launchd.
#
# Flow:
#   1) Sleep a random 0-540 minutes (skip with --now) so the commit time looks natural.
#   2) Run pick_task.py to scaffold today's folder.
#   3) Commit and push.
#
# Logs go to scripts/daily.log (gitignored).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_FILE="$SCRIPT_DIR/daily.log"

cd "$REPO_ROOT"

# Tee all output to the log with timestamps.
exec > >(while IFS= read -r line; do printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$line"; done | tee -a "$LOG_FILE") 2>&1

SKIP_SLEEP=0
for arg in "$@"; do
  case "$arg" in
    --now) SKIP_SLEEP=1 ;;
    *) echo "unknown arg: $arg"; exit 64 ;;
  esac
done

if [[ "$SKIP_SLEEP" -eq 0 ]]; then
  # 0-540 minutes => 0-32400 seconds. 09:00 launchd + this sleep lands inside 09:00-18:00.
  SLEEP_SECS=$((RANDOM % 32400))
  echo "sleeping $SLEEP_SECS seconds ($((SLEEP_SECS / 60)) minutes) before run"
  sleep "$SLEEP_SECS"
fi

# Pick venv python if present, else system python3.
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PY="$REPO_ROOT/.venv/bin/python"
else
  PY="$(command -v python3)"
fi

echo "using python: $PY"

FOLDER="$("$PY" "$SCRIPT_DIR/pick_task.py")"
echo "scaffolded: $FOLDER"

# Stage scaffold + updated progress log.
git add "$FOLDER" progress.md

if git diff --cached --quiet; then
  echo "nothing to commit (already scaffolded today). exiting clean."
  exit 0
fi

REL="${FOLDER#$REPO_ROOT/}"
TOPIC="$(basename "$FOLDER")"
git commit -m "scaffold: $TOPIC

Auto-scaffolded daily exercise.
Path: $REL"

# Push only if a remote is configured.
if git remote get-url origin >/dev/null 2>&1; then
  BRANCH="$(git rev-parse --abbrev-ref HEAD)"
  echo "pushing $BRANCH to origin"
  git push origin "$BRANCH"
else
  echo "no 'origin' remote configured — committed locally only. run setup.sh to add one."
fi

echo "done."
