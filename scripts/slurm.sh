#!/bin/bash
# Generic Slurm launcher: one torchrun per node, c10d rendezvous, then save_model.py.
#
# Resources are passed on the command line, since partition / account / GRES names are
# cluster-specific. Request one task per node and all of that node's GPUs:
#   CONFIG=configs/kimi_k25/ours_gw2.yaml \
#     sbatch --nodes=2 --ntasks-per-node=1 --gres=gpu:8 --time=72:00:00 scripts/slurm.sh
#   CONFIG=... RESUME=latest sbatch ... scripts/slurm.sh
#
# Environment knobs:
#   CONFIG     config to run (default configs/qwen3_30b/ours_gw2.yaml)
#   VENV       virtualenv to activate on every node
#   RUN_ROOT   redirect checkpoint_dir / log_dir here (a patched copy of the config is
#              written to ${RUN_ROOT}/config.yaml); the in-tree config is left untouched
#   RESUME     forwarded as --resume (latest or a run id)
#   MAIN_ARGS  extra arguments for main.py, e.g. "--max-layers 1"
#
# checkpoint_dir must be on a filesystem shared by all nodes: every rank reads the
# finished layer shards.
#SBATCH --job-name=moe-sq
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail
ulimit -c 0

CONFIG="${CONFIG:-configs/qwen3_30b/ours_gw2.yaml}"
RESUME="${RESUME:-}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_DIR}"

if [[ -n "${VENV:-}" ]]; then
  source "${VENV}/bin/activate"
fi

if [[ -n "${RUN_ROOT:-}" ]]; then
  mkdir -p "${RUN_ROOT}"
  python - "${CONFIG}" "${RUN_ROOT}" <<'PY'
import os, sys, yaml
src, root = sys.argv[1:3]
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg.setdefault("training", {})
cfg["training"]["checkpoint_dir"] = os.path.join(root, "checkpoints")
cfg["training"]["log_dir"] = os.path.join(root, "logs")
with open(os.path.join(root, "config.yaml"), "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
  CONFIG="${RUN_ROOT}/config.yaml"
fi

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
GPUS_PER_NODE="${GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}"
echo "[INFO] config=${CONFIG} nodes=${SLURM_NNODES} gpus/node=${GPUS_PER_NODE} master=${MASTER_ADDR}:${MASTER_PORT}"

read -r -a EXTRA_ARGS <<< "${MAIN_ARGS:-}"
if [[ -n "${RESUME}" ]]; then
  EXTRA_ARGS+=(--resume "${RESUME}")
fi

export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# One task per node that owns every GPU and CPU on it. Stated explicitly because some
# sites bind one GPU per task by default, which would put all local ranks on one device.
srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
  --gpus-per-task="${GPUS_PER_NODE}" --cpus-per-task="${SLURM_CPUS_ON_NODE}" --kill-on-bad-exit=1 \
  torchrun \
    --nnodes="${SLURM_NNODES}" \
    --nproc-per-node="${GPUS_PER_NODE}" \
    --rdzv-id="${SLURM_JOB_ID}" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv-conf=timeout=600 \
    --max-restarts=0 \
    main.py --config "${CONFIG}" "${EXTRA_ARGS[@]}"

python save_model.py --config "${CONFIG}"
