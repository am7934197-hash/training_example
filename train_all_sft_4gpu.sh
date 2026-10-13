#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-lerobot_new}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_PROCESSES="${NUM_PROCESSES:-${#GPU_IDS[@]}}"
if (( NUM_PROCESSES != 4 || ${#GPU_IDS[@]} != 4 )); then
    echo "ERROR: this launcher requires exactly 4 visible GPUs; got ${CUDA_VISIBLE_DEVICES}." >&2
    exit 2
fi

CONFIG_PATH="${SCRIPT_DIR}/pi05_train_all_sft.yaml"
EXPECTED_EFFECTIVE_BATCH_SIZE=64
read -r PER_DEVICE_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS < <(
    python - "${CONFIG_PATH}" <<'PY'
import sys
import yaml
with open(sys.argv[1], encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
print(cfg['batch_size'], cfg.get('gradient_accumulation_steps', 1))
PY
)
EFFECTIVE_BATCH_SIZE=$((PER_DEVICE_BATCH_SIZE * NUM_PROCESSES * GRADIENT_ACCUMULATION_STEPS))
if (( EFFECTIVE_BATCH_SIZE != EXPECTED_EFFECTIVE_BATCH_SIZE )); then
    echo "ERROR: effective batch size ${EFFECTIVE_BATCH_SIZE}; expected ${EXPECTED_EFFECTIVE_BATCH_SIZE}." >&2
    exit 2
fi

echo "GPUs: ${CUDA_VISIBLE_DEVICES}; effective batch: ${EFFECTIVE_BATCH_SIZE}"
accelerate launch --multi_gpu --num_processes="${NUM_PROCESSES}" --num_machines=1 --machine_rank=0 --mixed_precision=bf16 \
    -m lerobot.scripts.lerobot_train --config "${CONFIG_PATH}" "$@"
