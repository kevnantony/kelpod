#!/usr/bin/env bash
# Deploy the sandbox API to the Hugging Face Space, pinned to a pushed commit of this repo.
# Usage: bash deploy/hf-space/deploy.sh [commit]   (default: origin's tip of the current branch)
# Needs `hf auth login` with write access to the Space.
set -euo pipefail
SPACE=${KELPOD_SPACE:-kevinantony/kelpod}
cd "$(dirname "$0")"
git fetch -q origin
REF=${1:-$(git rev-parse "origin/$(git rev-parse --abbrev-ref HEAD)")}
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
cp app.py README.md "$TMP/"
sed "s/__REF__/$REF/" Dockerfile > "$TMP/Dockerfile"
uvx --from huggingface_hub hf repo create "$SPACE" --repo-type space --space-sdk docker --exist-ok >/dev/null
uvx --from huggingface_hub hf upload "$SPACE" "$TMP" . --repo-type space --commit-message "Deploy kelpod@${REF:0:7}"
echo "Deployed kelpod@${REF:0:7} to https://huggingface.co/spaces/$SPACE"
