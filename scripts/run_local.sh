#!/bin/bash
# Compress a model on a single node, then assemble the HF checkpoint.
#
#   CONFIG=configs/qwen3_30b/ours_gw2.yaml bash scripts/run_local.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG=... bash scripts/run_local.sh --resume
#
# Extra arguments are forwarded to main.py (e.g. --resume, --max-layers 1).
set -euo pipefail
ulimit -c 0

CONFIG="${CONFIG:-configs/qwen3_30b/ours_gw2.yaml}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

IFS="," read -r -a _gpus <<< "${CUDA_VISIBLE_DEVICES}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#_gpus[@]}}"
MASTER_PORT="${MASTER_PORT:-$((20000 + RANDOM % 20000))}"

echo "[INFO] config=${CONFIG} nproc=${NPROC_PER_NODE} gpus=${CUDA_VISIBLE_DEVICES}"

torchrun --nnodes=1 --nproc-per-node="${NPROC_PER_NODE}" --master-port="${MASTER_PORT}" \
  main.py --config "${CONFIG}" "$@"

python save_model.py --config "${CONFIG}"
