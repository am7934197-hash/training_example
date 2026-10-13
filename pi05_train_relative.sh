#!/bin/bash
# G1 place - PI0.5 relative-action training launcher

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-lerobot_new}"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

ACCELERATE="$(command -v accelerate)"
CONFIG_PATH="${SCRIPT_DIR}/pi05_train_relative.yaml"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

if [[ -z "${SWANLAB_API_KEY:-}" ]]; then
    echo "WARNING: SWANLAB_API_KEY is not set; SwanLab logging may fail."
fi

IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_PROCESSES="${NUM_PROCESSES:-${#GPU_IDS[@]}}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
EXPECTED_EFFECTIVE_BATCH_SIZE="${EXPECTED_EFFECTIVE_BATCH_SIZE:-64}"

if (( NUM_PROCESSES != ${#GPU_IDS[@]} )); then
    echo "ERROR: NUM_PROCESSES=${NUM_PROCESSES}, but CUDA_VISIBLE_DEVICES contains ${#GPU_IDS[@]} GPUs." >&2
    exit 2
fi

read -r PER_DEVICE_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS < <(
    python - "${CONFIG_PATH}" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

print(config["batch_size"], config.get("gradient_accumulation_steps", 1))
PY
)
EFFECTIVE_BATCH_SIZE=$((PER_DEVICE_BATCH_SIZE * NUM_PROCESSES * GRADIENT_ACCUMULATION_STEPS))

if (( EFFECTIVE_BATCH_SIZE != EXPECTED_EFFECTIVE_BATCH_SIZE )); then
    echo "ERROR: effective batch size is ${EFFECTIVE_BATCH_SIZE}, expected ${EXPECTED_EFFECTIVE_BATCH_SIZE}." >&2
    echo "Formula: ${PER_DEVICE_BATCH_SIZE} per GPU x ${NUM_PROCESSES} GPUs x ${GRADIENT_ACCUMULATION_STEPS} accumulation steps." >&2
    exit 2
fi

echo "Conda env: ${CONDA_ENV}"
echo "Python: $(command -v python)"
echo "Visible GPUs: ${CUDA_VISIBLE_DEVICES}; processes: ${NUM_PROCESSES}"
echo "Effective batch size: ${PER_DEVICE_BATCH_SIZE} x ${NUM_PROCESSES} x ${GRADIENT_ACCUMULATION_STEPS} = ${EFFECTIVE_BATCH_SIZE}"

LAUNCH_ARGS=(
    --num_processes="${NUM_PROCESSES}"
    --mixed_precision="${MIXED_PRECISION}"
    --num_machines=1
    --machine_rank=0
)
if (( NUM_PROCESSES > 1 )); then
    LAUNCH_ARGS+=(--multi_gpu)
fi

"${ACCELERATE}" launch "${LAUNCH_ARGS[@]}" \
    -m lerobot.scripts.lerobot_train \
    --config "${CONFIG_PATH}" \
    "$@"
