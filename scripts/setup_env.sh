#!/usr/bin/env bash
# TARSUS: one-time environment setup on the A100 server.
# Usage (from the repo root):  bash scripts/setup_env.sh
# - creates conda env /mnt/zihanw/envs/tarsus (cloned from evowm if present, so torch is already there)
# - installs the vendored official DeepLabCut (third_party/DeepLabCut) in editable mode
# - downloads the official superanimal_bird weights through DLC's own function
# - checks that the Duong data is in place
# Log: outputs/setup_env.log
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ENV=/mnt/zihanw/envs/tarsus
EVOWM_ENV=/mnt/zihanw/envs/evowm
DUONG_ROOT="${DUONG_ROOT:-/mnt/zihanw/data/duong_data}"
export PIP_CACHE_DIR=/mnt/zihanw/.cache/pip
export TMPDIR=/mnt/zihanw/tmp
export HF_HOME=/mnt/zihanw/.cache/huggingface
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$HF_HOME" "$REPO/outputs"
LOG="$REPO/outputs/setup_env.log"
exec > >(tee "$LOG") 2>&1
echo "[$(date '+%F %T')] repo=$REPO env=$ENV"

# 1. conda env
if [ ! -x "$ENV/bin/python" ]; then
  if [ -x "$EVOWM_ENV/bin/python" ]; then
    echo "== clone env from $EVOWM_ENV"
    conda create -y -p "$ENV" --clone "$EVOWM_ENV" || exit 1
  else
    echo "== create fresh env (python 3.11)"
    conda create -y -p "$ENV" python=3.11 || exit 1
    "$ENV/bin/python" -m pip install -i https://pypi.org/simple torch==2.7.1 torchvision==0.22.1 || exit 1
  fi
fi
PY="$ENV/bin/python"

# 2. official DeepLabCut (vendored, editable)
echo "== pip install -e third_party/DeepLabCut"
"$PY" -m pip install -i https://pypi.org/simple -e "$REPO/third_party/DeepLabCut" || exit 1

# 3. import check
echo "== import check"
"$PY" - <<'EOF' || exit 1
import torch, deeplabcut
print("deeplabcut", deeplabcut.__version__, "from", deeplabcut.__file__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
EOF

# 4. superanimal_bird weights (HF; fall back to hf-mirror.com if huggingface.co is unreachable)
if ! timeout 15 curl -sI https://huggingface.co >/dev/null; then
  echo "== huggingface.co unreachable, using https://hf-mirror.com"
  export HF_ENDPOINT=https://hf-mirror.com
fi
echo "== download superanimal_bird (pose resnet_50 + detector fasterrcnn_mobilenet_v3_large_fpn)"
"$PY" - <<'EOF' || exit 1
from deeplabcut.pose_estimation_pytorch.modelzoo.utils import get_super_animal_snapshot_path
for name in ["resnet_50", "fasterrcnn_mobilenet_v3_large_fpn"]:
    p = get_super_animal_snapshot_path("superanimal_bird", name)
    print(name, p, round(p.stat().st_size / 1e6, 1), "MB")
EOF

# 5. data check
echo "== data check: $DUONG_ROOT"
PROJ="$DUONG_ROOT/deeplabcut/bird-landing-Duong-2025-06-29"
if [ -f "$PROJ/config.yaml" ]; then
  echo "config.yaml OK; labeled folders: $(ls "$PROJ/labeled-data" | wc -l); videos: $(ls "$PROJ/videos" | wc -l)"
  echo "raw videos: $(find "$DUONG_ROOT/Raw datasets" -name '*.mp4' 2>/dev/null | wc -l)"
else
  echo "MISSING: $PROJ/config.yaml  (upload duong_min.tar.gz and extract into $DUONG_ROOT)"
fi
echo "[$(date '+%F %T')] done"
