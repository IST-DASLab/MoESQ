#!/usr/bin/env bash
# Install vLLM v0.30.0 + the MoESQ paired48_nvfp4 MoE backend + paired_nvfp4_kernels
# into a dedicated serving venv (separate from the MoESQ training venv).
#
#   bash integrations/vllm/install.sh
#
# Requirements: SM100 (B200/GB200) or SM120 (RTX 5090 / RTX PRO 6000) GPU to run; `uv`;
# CUDA toolkit >= 12.8 with nvcc on PATH or CUDA_HOME set (only for the kernel build). Nothing in vLLM itself is compiled:
# vLLM installs from the precompiled v0.30.0 wheel and only our Python changes are
# patched in.
#
# Knobs (environment variables):
#   VENV_DIR           serving venv                (default: <repo>/.venv-vllm)
#   VLLM_DIR           vLLM source checkout        (default: <repo>/build/vllm)
#   TORCH_BACKEND      torch CUDA variant          (default: cu130; must match nvcc)
#   ALLOW_CUDA_MISMATCH=1  proceed even if nvcc and TORCH_BACKEND differ
#   PAIRED_NVFP4_ARCHS kernel target archs         (default: this machine's GPU, 100a or
#                      120a; "100a;120a" builds one wheel for both; 100a without a GPU)
#   MAX_JOBS           kernel build parallelism    (default: 8)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv-vllm}"
VLLM_DIR="${VLLM_DIR:-$REPO_ROOT/build/vllm}"
TORCH_BACKEND="${TORCH_BACKEND:-cu130}"
KERNEL_DIR="$REPO_ROOT/third_party/grouped-sparse-GEMM"
# The kernel release the vLLM patch is pinned to (the submodule commit).
KERNEL_TAG=v0.13.0
KERNEL_COMMIT=db70eebaae8fdca2325a9997dfa040fc0cce225e
PATCH="$HERE/moe-sq-v0.30.0.patch"
VLLM_TAG=v0.30.0
VLLM_COMMIT=ced6857afa0ea7b2e3f0846a62e1394e90f15607

say() { printf '\n==> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

command -v uv >/dev/null || die "uv not found (https://docs.astral.sh/uv/)"
# Resolve nvcc the way torch's extension builder does: CUDA_HOME first, then PATH.
NVCC="${CUDA_HOME:+$CUDA_HOME/bin/nvcc}"
[ -x "${NVCC:-}" ] || NVCC="$(command -v nvcc || true)"
[ -x "${NVCC:-}" ] || die "nvcc not found: set CUDA_HOME or put the CUDA toolkit on PATH"
export CUDA_HOME="$(dirname "$(dirname "$NVCC")")"
NVCC_VER="$("$NVCC" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
# cu130 -> 13.0, cu128 -> 12.8: the kernel links against torch, so they must agree.
TORCH_CUDA="$(echo "$TORCH_BACKEND" | sed -n 's/^cu\([0-9]*\)\([0-9]\)$/\1.\2/p')"
if [ -n "$TORCH_CUDA" ] && [ "$TORCH_CUDA" != "$NVCC_VER" ] \
    && [ "${ALLOW_CUDA_MISMATCH:-0}" != 1 ]; then
    die "nvcc $NVCC_VER != torch CUDA $TORCH_CUDA; set TORCH_BACKEND to match" \
        "(or ALLOW_CUDA_MISMATCH=1)"
fi
git -C "$REPO_ROOT" submodule update --init --recursive third_party/grouped-sparse-GEMM \
    || die "could not fetch the kernel submodule (clone MoESQ with git, not as an archive)"
[ -f "$KERNEL_DIR/setup.py" ] || die "kernel sources missing at $KERNEL_DIR"
[ "$(git -C "$KERNEL_DIR" rev-parse HEAD)" = "$KERNEL_COMMIT" ] \
    || die "$KERNEL_DIR is not at grouped-sparse-GEMM $KERNEL_TAG ($KERNEL_COMMIT);" \
        "run: git submodule update --init third_party/grouped-sparse-GEMM"

# Kernel archs: default to the GPU in this machine (SM100 -> 100a, SM120 -> 120a).
if [ -z "${PAIRED_NVFP4_ARCHS:-}" ]; then
    CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | head -n1 | tr -d ' .' || true)"
    case "$CAP" in
        100|120) PAIRED_NVFP4_ARCHS="${CAP}a" ;;
        "") PAIRED_NVFP4_ARCHS=100a
            echo "no GPU visible; building the kernels for 100a (set PAIRED_NVFP4_ARCHS)" ;;
        *) die "GPU compute capability $CAP is not supported (SM100 or SM120 only);" \
            "set PAIRED_NVFP4_ARCHS to build anyway" ;;
    esac
fi

say "vLLM $VLLM_TAG source -> $VLLM_DIR"
if [ ! -d "$VLLM_DIR/.git" ]; then
    git clone --depth 1 --branch "$VLLM_TAG" https://github.com/vllm-project/vllm.git "$VLLM_DIR"
fi
[ "$(git -C "$VLLM_DIR" rev-parse HEAD)" = "$VLLM_COMMIT" ] \
    || die "$VLLM_DIR is not at $VLLM_TAG ($VLLM_COMMIT)"
if git -C "$VLLM_DIR" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "patch already applied"
else
    git -C "$VLLM_DIR" apply "$PATCH"
    echo "patch applied"
fi

say "serving venv -> $VENV_DIR"
# A uv-managed Python ships the headers (Python.h) the kernel build needs.
[ -x "$VENV_DIR/bin/python" ] || uv venv --managed-python --python 3.12 "$VENV_DIR"
PY="$VENV_DIR/bin/python"

say "vLLM (precompiled $VLLM_TAG wheel + patched Python sources, torch $TORCH_BACKEND)"
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT="$VLLM_COMMIT" \
    uv pip install --python "$PY" -e "$VLLM_DIR" --torch-backend="$TORCH_BACKEND"

say "paired_nvfp4_kernels $KERNEL_TAG for $PAIRED_NVFP4_ARCHS (third_party/grouped-sparse-GEMM, built against that torch)"
uv pip install --python "$PY" setuptools wheel ninja
# --no-build-isolation/--no-deps: the extension must link against the installed torch.
( cd "$KERNEL_DIR" && PAIRED_NVFP4_ARCHS="$PAIRED_NVFP4_ARCHS" \
    MAX_JOBS="${MAX_JOBS:-8}" uv pip install --python "$PY" . --no-deps --no-build-isolation )

say "check"
# Run from a neutral directory: `python -` puts the cwd first on sys.path, so a
# vllm/ source tree in the caller's cwd would shadow the installed package.
(cd "$VENV_DIR" && "$PY" -) <<'EOF'
import torch, vllm, paired_nvfp4_kernels as pnk
import vllm._custom_ops as ops
print(f"vllm {vllm.__version__} | torch {torch.__version__} | paired_nvfp4_kernels {pnk.__version__}"
      f" (built for SM{pnk.built_archs()})")
assert ops.paired_nvfp4_available(), "vLLM cannot load paired_nvfp4_kernels"
EOF

cat <<EOF

Done. Serve a MoESQ checkpoint (the paired48_nvfp4 backend is selected automatically):
  source $VENV_DIR/bin/activate
  vllm serve ISTA-DASLab/Qwen3-30B-A3B-P48NVFP4-MoESQ
EOF
