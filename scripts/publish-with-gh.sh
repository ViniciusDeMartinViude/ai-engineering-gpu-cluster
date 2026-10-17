#!/usr/bin/env bash
# Run on YOUR computer after `gh auth login`, not on a worker or the manager.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v gh >/dev/null || { echo 'Install GitHub CLI: https://cli.github.com/'; exit 1; }
gh auth status
repo_name="${1:-ai-engineering-gpu-cluster}"
visibility="${2:---private}"
[[ "$visibility" == '--private' || "$visibility" == '--public' ]] || { echo 'Second argument: --private or --public'; exit 1; }
if [[ ! -d .git ]]; then git init -b main; fi
git add .
git diff --cached --quiet || git commit -m 'Initial GPU cluster manager and workers deployment kit'
gh repo create "$repo_name" "$visibility" --source=. --remote=origin --push
