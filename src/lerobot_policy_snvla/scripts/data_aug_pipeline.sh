#!/usr/bin/env bash
set -euo pipefail

HF_USER=0xNOY
SRC=${HF_USER}/so101_wn

SRC_SCOOPS=10

cd "$(dirname "$0")/../.."

TARGET_SCOOPS=$(seq -s, 1 $SRC_SCOOPS)

snvla-generate-partial-scoop-episodes "$SRC" "outputs/${SRC}_gen" --source-scoops "$SRC_SCOOPS" --target-object "soybeans,red beans" --target-scoops "$TARGET_SCOOPS" &&

hf upload --repo-type dataset --revision main "${SRC}_gen" "outputs/${SRC}_gen" &&

lerobot-edit-dataset --repo_id "${SRC}_mix" --operation.type merge --operation.repo_id="['${SRC}', '${SRC}_gen']" &&

snvla-augment-narrations "${SRC}_mix" "outputs/${SRC}_aug" --window-size 8 &&

hf upload --repo-type dataset --revision main "${SRC}_aug" "outputs/${SRC}_aug"
