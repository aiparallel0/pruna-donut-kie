#!/usr/bin/env bash
# fetch_sroie.sh — download and sha256-verify the SROIE Task-3 test set.
#
# Hashes are reused from the companion repo (aiparallel0/kaggle2) to ensure
# that this repo evaluates on exactly the same 347-image split.
#
# Usage: bash scripts/fetch_sroie.sh
# Output: data/test/ (347 receipt images + ground-truth JSON files)

set -euo pipefail

SROIE_URL="https://github.com/aiparallel0/kaggle2/releases/download/sroie-data/sroie_task3_test.tar.gz"
DEST="data"
ARCHIVE="${DEST}/sroie_task3_test.tar.gz"

# sha256 hash of the verified archive from aiparallel0/kaggle2.
# Do not alter; change is evidence of a corrupted or substituted download.
EXPECTED_SHA256="# TODO: copy sha256 hash from aiparallel0/kaggle2"

if [[ "$EXPECTED_SHA256" == \#* ]]; then
    echo "ERROR: EXPECTED_SHA256 has not been set in scripts/fetch_sroie.sh."
    echo "       Copy the verified hash from aiparallel0/kaggle2 before running."
    exit 1
fi

echo "==> Downloading SROIE Task-3 test set"
mkdir -p "${DEST}"
curl --fail --location --progress-bar "${SROIE_URL}" -o "${ARCHIVE}"

echo "==> Verifying sha256 checksum"
ACTUAL_SHA256="$(sha256sum "${ARCHIVE}" | awk '{print $1}')"
if [[ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]]; then
    echo "ERROR: sha256 mismatch!"
    echo "       Expected: ${EXPECTED_SHA256}"
    echo "       Got:      ${ACTUAL_SHA256}"
    rm -f "${ARCHIVE}"
    exit 1
fi
echo "    OK — sha256 matches."

echo "==> Extracting archive"
tar -xz -C "${DEST}" -f "${ARCHIVE}"
rm -f "${ARCHIVE}"

echo "==> SROIE Task-3 test set ready at ${DEST}/test/"
