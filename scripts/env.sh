# TARSUS runtime environment. Usage: source scripts/env.sh
# Uses the tarsus conda env without `conda activate`, and puts the env's own
# libnvJitLink first on LD_LIBRARY_PATH (an older system copy breaks `import torch`:
# "undefined symbol __nvJitLinkCreate_12_8").
TARSUS_ENV=/mnt/zihanw/envs/tarsus
_NVJ=$(ls -d "$TARSUS_ENV"/lib/python3.*/site-packages/nvidia/nvjitlink/lib 2>/dev/null | head -1)
export LD_LIBRARY_PATH="${_NVJ:+$_NVJ:}${LD_LIBRARY_PATH:-}"
export PATH="$TARSUS_ENV/bin:$PATH"
export PIP_CACHE_DIR=/mnt/zihanw/.cache/pip
export TMPDIR=/mnt/zihanw/tmp
export HF_HOME=/mnt/zihanw/.cache/huggingface
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$HF_HOME"
