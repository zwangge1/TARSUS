#!/usr/bin/env bash
# Group A: official DLC 3 top_down_resnet_50, SA backbone init, species leave-one-out, 2 seeds.
# Usage (repo root):  GPUS="0,1" bash scripts/run_groupA.sh        (default GPUS="0")
# Runs are distributed round-robin over the listed GPUs and executed in parallel per GPU.
# Each run writes outputs/runs/A_<fold>_s<seed>/metrics.json and a log next to it.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO/scripts/env.sh"
GPUS="${GPUS:-0}"
EPOCHS="${EPOCHS:-200}"
DUONG=/mnt/zihanw/data/duong_data/deeplabcut/bird-landing-Duong-2025-06-29
mkdir -p "$REPO/outputs/runs"

if [ ! -f "$REPO/outputs/dlc_project/split.json" ]; then
  python "$REPO/scripts/make_dlc_project.py" --duong-project "$DUONG" --out "$REPO/outputs/dlc_project" || exit 1
fi

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
jobs=()
for fold in Duck Goose Pelican Swan; do for seed in 0 1; do jobs+=("$fold $seed"); done; done

run_on_gpu() {  # $1 = gpu id, $2.. = job indices
  local gpu=$1; shift
  for j in "$@"; do
    local fold seed; read -r fold seed <<< "${jobs[$j]}"
    local dir="$REPO/outputs/runs/A_${fold}_s${seed}"; mkdir -p "$dir"
    if [ -f "$dir/metrics.json" ]; then echo "skip $dir (done)"; continue; fi
    echo "[gpu $gpu] start A $fold seed $seed  $(date '+%F %T')"
    CUDA_VISIBLE_DEVICES=$gpu python "$REPO/scripts/train_eval.py" --group A --fold "$fold" --seed "$seed" \
      --epochs "$EPOCHS" --device cuda:0 > "$dir/train.log" 2>&1 \
      && echo "[gpu $gpu] done  A $fold seed $seed  $(date '+%F %T')" \
      || echo "[gpu $gpu] FAILED A $fold seed $seed, see $dir/train.log"
  done
}

n=${#GPU_ARR[@]}
for ((g=0; g<n; g++)); do
  idx=(); for ((j=g; j<${#jobs[@]}; j+=n)); do idx+=("$j"); done
  run_on_gpu "${GPU_ARR[$g]}" "${idx[@]}" &
done
wait
python - <<PY
import sys; sys.path.insert(0, "$REPO/scripts")
from pathlib import Path; import metrics
df = metrics.summarize_runs(sorted(Path("$REPO/outputs/runs").glob("A_*")))
print(df.to_string(index=False)); df.to_csv("$REPO/outputs/runs/summary_A.csv", index=False)
PY
echo "done: $REPO/outputs/runs/summary_A.csv"
