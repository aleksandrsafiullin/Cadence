#!/bin/bash
# Double-click launcher for Excavation Video Lab (localhost only).
cd "$(dirname "$0")" || exit 1

export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not on PATH."
  echo "Install it from https://docs.astral.sh/uv/getting-started/installation/ and open this file again."
  read -r -p "Press Enter to close..."
  exit 1
fi

uv run excavation ui
status=$?
if [[ $status -ne 0 ]]; then
  echo
  echo "The interface exited with status ${status}."
  read -r -p "Press Enter to close..."
fi
exit "$status"
