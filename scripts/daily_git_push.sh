#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR/.." rev-parse --show-toplevel)"
cd "$REPO_ROOT"

SCRIPT_PATHS=()
while IFS= read -r -d '' path; do
  case "$path" in
    *.py|*.ipynb|*.sh|*.R|*.r|*.js|*.ts|*.sql)
      SCRIPT_PATHS+=("$path")
      ;;
  esac
done < <(git ls-files --cached --others --exclude-standard -z)

if [[ "${1:-}" == "--dry-run" ]]; then
  if [[ "${#SCRIPT_PATHS[@]}" -gt 0 ]]; then
    git status --short --untracked-files=all -- "${SCRIPT_PATHS[@]}"
  fi
  exit 0
elif [[ "$#" -gt 0 ]]; then
  printf 'Usage: %s [--dry-run]\n' "$0" >&2
  exit 2
fi

branch="$(git branch --show-current)"
if [[ -z "$branch" ]]; then
  printf 'Cannot publish script changes from a detached HEAD.\n' >&2
  exit 1
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  printf 'Git remote "origin" is not configured.\n' >&2
  exit 1
fi

if [[ -z "$(git config user.name)" || -z "$(git config user.email)" ]]; then
  printf 'Git author name and email must be configured for this repository.\n' >&2
  exit 1
fi

if [[ "${#SCRIPT_PATHS[@]}" -gt 0 ]]; then
  git add -A -- "${SCRIPT_PATHS[@]}"
  if ! git diff --cached --quiet -- "${SCRIPT_PATHS[@]}"; then
    git commit -m "Automated daily script sync: $(date +%F)" -- "${SCRIPT_PATHS[@]}"
  fi
fi

git push origin "$branch"
