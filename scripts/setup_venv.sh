#!/bin/bash
# Create the MoESQ Python environment.
#
# torch and flash-attn are NOT installed here: every cluster pins a different
# CUDA backend / GPU arch, so you install those two by hand before running this
# script (see pyproject.toml [project.optional-dependencies] for the URLs/wheels).
# This script installs only the cluster-agnostic dependencies.
#
# Usage:
#   bash scripts/setup_venv.sh             # default: .venv, core deps only (no torch/flash)
#   VENV_DIR=venv-moe-sq bash scripts/setup_venv.sh   # custom venv path
#   EXTRAS="torch,flash" bash scripts/setup_venv.sh   # opt back in to the torch/flash extras
#   USE_UV=1 bash scripts/setup_venv.sh              # use `uv` (faster, requires uv installed)
#
# Notes:
#   - On HPC clusters where the venv lives on shared NFS, point VENV_DIR
#     somewhere shared by all nodes.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
EXTRAS="${EXTRAS:-}"
USE_UV="${USE_UV:-0}"
PY="${PYTHON:-python3}"

echo "[setup_venv] repo:    ${REPO_ROOT}"
echo "[setup_venv] venv:    ${VENV_DIR}"
echo "[setup_venv] extras:  ${EXTRAS}"
echo "[setup_venv] uv:      ${USE_UV}"

if [[ -d "${VENV_DIR}" ]]; then
  echo "[setup_venv] venv already exists at ${VENV_DIR}; activating + upgrading."
else
  if [[ "${USE_UV}" == "1" ]]; then
    command -v uv >/dev/null 2>&1 || {
      echo "[setup_venv] ERROR: USE_UV=1 but 'uv' is not on PATH." >&2
      exit 1
    }
    uv venv "${VENV_DIR}" --python "${PY}"
  else
    "${PY}" -m venv "${VENV_DIR}"
  fi
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

if ! python -c "import torch" >/dev/null 2>&1; then
  echo "[setup_venv] WARNING: torch is not installed in this venv." >&2
  echo "[setup_venv]          Install torch (and flash-attn) manually for this cluster's" >&2
  echo "[setup_venv]          CUDA backend before running training. See pyproject.toml." >&2
fi

if [[ "${USE_UV}" == "1" ]]; then
  uv pip install --upgrade pip wheel setuptools
  if [[ -n "${EXTRAS}" ]]; then
    uv pip install -e "${REPO_ROOT}[${EXTRAS}]"
  else
    uv pip install -e "${REPO_ROOT}"
  fi
else
  python -m pip install --upgrade pip wheel setuptools
  if [[ -n "${EXTRAS}" ]]; then
    python -m pip install -e "${REPO_ROOT}[${EXTRAS}]"
  else
    python -m pip install -e "${REPO_ROOT}"
  fi
fi

echo "[setup_venv] done. activate with: source ${VENV_DIR}/bin/activate"
