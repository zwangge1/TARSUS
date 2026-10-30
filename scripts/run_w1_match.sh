#!/usr/bin/env bash
# W1 task 1: official keypoint matching + SA leg-point distances on Duong labels.
# Usage (repo root): bash scripts/run_w1_match.sh      [GPU via CUDA_VISIBLE_DEVICES, default 0]
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO/scripts/env.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OUT="$REPO/outputs/w1_match"
mkdir -p "$OUT"
python "$REPO/scripts/w1_superbird_match.py" \
  --duong-project /mnt/zihanw/data/duong_data/deeplabcut/bird-landing-Duong-2025-06-29 \
  --out "$OUT" 2>&1 | tee "$OUT/run.log"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "FAILED, see $OUT/run.log"; exit 1; fi
echo "done: $OUT/summary.md"
