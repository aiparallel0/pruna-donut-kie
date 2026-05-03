#!/usr/bin/env bash
# fetch_checkpoint.sh — download the fine-tuned DONUT checkpoint from the
# companion repo (aiparallel0/kaggle2).
#
# Usage: bash scripts/fetch_checkpoint.sh
# Output: checkpoints/donut-sroie/
#
# Edit the CHECKPOINT_URL variable below before running.

set -euo pipefail

# TODO: paste checkpoint URL so I edit it manually before running
CHECKPOINT_URL="# TODO: paste checkpoint URL so I edit it manually before running"

DEST="checkpoints/donut-sroie"

if [[ "$CHECKPOINT_URL" == \#* ]]; then
    echo "ERROR: CHECKPOINT_URL has not been set in scripts/fetch_checkpoint.sh."
    echo "       Open the file and replace the placeholder with the actual URL."
    exit 1
fi

echo "==> Downloading fine-tuned DONUT checkpoint from ${CHECKPOINT_URL}"
mkdir -p "${DEST}"
curl --fail --location --progress-bar "${CHECKPOINT_URL}" \
    | tar -xz -C "${DEST}"

echo "==> Checkpoint downloaded to ${DEST}/"
