#!/usr/bin/env bash
# ml-daily — one-time setup.
#
# Idempotent: safe to re-run. It will:
#   1) git init (if needed) and set REPO-LOCAL author identity.
#   2) Create a Python venv with pyyaml.
#   3) Make the initial commit (if no commits yet).
#   4) Help you create the GitHub remote (via `gh` if available, otherwise prints instructions).
#   5) Install the launchd plist at ~/Library/LaunchAgents/com.chelappan.mldaily.plist
#      and load it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GIT_NAME="Chelappan Kumar"
GIT_EMAIL="chelappankumar23@gmail.com"
REPO_NAME="ml-daily"
PLIST_NAME="com.chelappan.mldaily.plist"
PLIST_SRC="$REPO_ROOT/$PLIST_NAME"
PLIST_DST="$HOME/Library/LaunchAgents/$PLIST_NAME"

cd "$REPO_ROOT"

echo ">>> 1/5  git init + repo-local author"
if [[ ! -d .git ]]; then
  git init -b main
fi
git config user.name  "$GIT_NAME"
git config user.email "$GIT_EMAIL"
echo "    author = $GIT_NAME <$GIT_EMAIL>"

echo ">>> 2/5  Python venv + pyyaml"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet pyyaml
deactivate

echo ">>> 3/5  initial commit"
chmod +x "$SCRIPT_DIR/daily.sh" "$SCRIPT_DIR/pick_task.py"
git add .
if git diff --cached --quiet && git rev-parse --verify HEAD >/dev/null 2>&1; then
  echo "    nothing new to commit."
else
  if git rev-parse --verify HEAD >/dev/null 2>&1; then
    git commit -m "chore: setup updates"
  else
    git commit -m "chore: scaffold ml-daily repo

Daily ML pipelines + AI agents practice repo.
See README.md for how it works."
  fi
fi

echo ">>> 4/5  GitHub remote"
if git remote get-url origin >/dev/null 2>&1; then
  echo "    origin already set: $(git remote get-url origin)"
else
  if command -v gh >/dev/null 2>&1; then
    echo "    creating GitHub repo via gh..."
    read -r -p "    Create public repo '$REPO_NAME' under your GitHub account now? [y/N] " ans
    if [[ "$ans" =~ ^[Yy]$ ]]; then
      gh repo create "$REPO_NAME" --public --source=. --remote=origin --push
    else
      echo "    skipped. To add manually later:"
      echo "      gh repo create $REPO_NAME --public --source=. --remote=origin --push"
    fi
  else
    echo "    'gh' CLI not found. To add the remote manually:"
    echo "      1) Create an empty repo named '$REPO_NAME' on https://github.com/new"
    echo "      2) cd $REPO_ROOT"
    echo "      3) git remote add origin git@github.com:ChelappanKumar/$REPO_NAME.git"
    echo "      4) git push -u origin main"
    echo
    echo "    Or install gh first:  brew install gh && gh auth login"
  fi
fi

echo ">>> 5/5  launchd"
mkdir -p "$HOME/Library/LaunchAgents"
if [[ ! -f "$PLIST_SRC" ]]; then
  echo "    plist source missing at $PLIST_SRC — aborting launchd step."
  exit 1
fi

# Render the plist with the actual repo path substituted in.
sed "s|__REPO_ROOT__|$REPO_ROOT|g" "$PLIST_SRC" > "$PLIST_DST"
echo "    installed $PLIST_DST"

# Unload first (ignore failure if not loaded), then load.
launchctl unload "$PLIST_DST" >/dev/null 2>&1 || true
launchctl load "$PLIST_DST"
echo "    loaded. Next run: 09:00 local (then a random 0-540 min delay)."

echo
echo "Setup complete."
echo
echo "Try a dry run now:"
echo "  bash $SCRIPT_DIR/daily.sh --now"
echo
echo "Disable later with:"
echo "  launchctl unload $PLIST_DST"
