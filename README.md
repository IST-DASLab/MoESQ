# Hardware-Native Joint Sparse-Quantization for Trillion-Scale Mixture-of-Experts

[![arXiv](https://img.shields.io/badge/arXiv-2610.02241-b31b1b.svg)](https://arxiv.org/abs/2610.02241)
[![Hugging Face: MoESQ checkpoints](https://img.shields.io/badge/Hugging_Face-Released_checkpoints-FFD21E.svg)](https://huggingface.co/collections/ISTA-DASLab/moesq)

**MoESQ** jointly sparsifies and quantizes the expert weights of large Mixture-of-Experts
models. The routed-expert linears get **paired-4:8 structured sparsity** with **NVFP4**
weights and NVFP4 activations (**W4A4**), targeting NVIDIA Blackwell sparse tensor cores.
Masks and weight values are learned jointly, layer by layer, so the method scales to
trillion-parameter models that don't fit in GPU memory. Router, attention and shared experts
stay at full precision.

This repository contains:

- **Compression** (`main.py`, `src/`, [`configs/`](configs/)): MoESQ and the SparseGPTQ, OBR,
  JSQ and GSQ baselines, plus export to compressed-tensors checkpoints (`save_model.py`).
- **Serving** ([`integrations/vllm/`](integrations/vllm/)): a patch to vLLM v0.30.0 that adds
  the `paired48_nvfp4` MoE backend, with a one-command installer.
- **Kernels** ([`third_party/grouped-sparse-GEMM/`](third_party/grouped-sparse-GEMM/), git
  submodule): the paired-4:8 sparse NVFP4 grouped GEMM that the backend calls.

## Supported models

| Model | Wrapper | Notes |
|---|---|---|
| Qwen3-MoE (30B-A3B, 235B-A22B) | `Qwen3MoeWrapper`, `Qwen3MoeFusedWrapper` | 128 experts, 8 active |
| Qwen3.5-MoE (397B-A17B) | `Qwen35MoeWrapper` | 512 experts, hybrid attention, shared experts |
| Kimi K2 / K2.5 | `KimiK2Wrapper`, `KimiK25Wrapper`, `KimiK25FusedWrapper` | up to 384 experts |
| Qwen3.8-Flash-Next | `Qwen4ExpWrapper` | 512 experts, hyper-connections, per-layer n-gram embedding, sparse attention |
| Kimi-K3 | `KimiK3Wrapper` | 896 latent experts, Attention Residuals, SiTU, MXFP4 source; remote code (`fla-core`); compression only |

The wrappers are expert-parallel. They run at any `world_size`, including a single GPU.

Compression runs on any CUDA GPU (it is fake-quantized PyTorch), so Hopper nodes work as
well as Blackwell. Qwen3.8-Flash-Next checkpoints serve with the `paired48_nvfp4` backend.
Serving Kimi-K3 is not wired up yet: it needs a SiTU path in the kernel.

## Installation

```bash
git clone --recurse-submodules https://github.com/IST-DASLab/MoESQ.git && cd MoESQ
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130   # match your CUDA
bash scripts/setup_venv.sh     # reuses .venv, installs the pinned dependencies
cp .env.example .env           # HF_TOKEN, WANDB_API_KEY
```

The dependencies are pinned to `transformers==5.17.0` and `compressed-tensors==0.19.0`,
which cover every model family above. `flash-attn` 2.8.3 is optional; without it,
attention falls back to `sdpa`. Kimi-K3 additionally needs the `kimi-k3` extra
(`fla-core`, for its linear-attention layers).

Serving uses its own environment (see [below](#serving-with-vllm)).

## Usage

```bash
# compress (single GPU / single node / Slurm multi-node)
python main.py --config configs/qwen3_30b/ours_gw2.yaml [--max-layers 1]
torchrun --nproc-per-node=8 main.py --config configs/qwen3_30b/ours_gw2.yaml
CONFIG=configs/kimi_k25/ours_gw2.yaml VENV=$PWD/.venv RUN_ROOT=/shared/runs/kimi \
  sbatch --nodes=2 --ntasks-per-node=1 --gres=gpu:8 scripts/slurm.sh

# resume an interrupted run, then assemble the HF checkpoint
python main.py --config <config.yaml> --resume [<run_id>]
python save_model.py --config <config.yaml> [--run-id <id>] [--out-dir ./out]
```

- **Configs:** [`configs/`](configs/) has every paper arm and ablation, and
  [`configs/README.md`](configs/README.md) documents the keys.
- **Output:** runs are written to `<checkpoint_dir>/<run_id>/`. The assembled compressed-tensors
  model goes to `assembled/`. Paired-4:8 NVFP4 experts are stored in the paired48 sparse
  storage format (`weight_sparse_packed` + `weight_sparse_mask`, about 0.65x the expert
  bytes of dense packed NVFP4), which the patched vLLM loads directly.
  `--no-sparse-storage` writes dense packed NVFP4 with the pruned elements zeroed instead;
  the tool in the vLLM integration converts losslessly between the two.

**Evaluation:** serve the checkpoint (see below), then run lm-eval against the server from the
training venv. The default tasks are GSM8K, ARC-Challenge, ARC-Easy, Winogrande and PIQA.

```bash
python eval_model.py --config <config.yaml> --base-url http://localhost:8000/v1/completions
```

## Serving with vLLM

MoESQ checkpoints do not load in upstream vLLM. `install.sh` sets up a separate `.venv-vllm`
in four steps:

1. Clone vLLM v0.30.0.
2. Apply the `paired48_nvfp4` patch.
3. Install the precompiled vLLM wheel; vLLM itself is not compiled.
4. Build the kernels against the same torch.

It requires an SM100 (B200/GB200) or SM120 (RTX 5090 / RTX PRO 6000) GPU, [`uv`](https://docs.astral.sh/uv/), and a CUDA
toolkit >= 12.8.

```bash
bash integrations/vllm/install.sh && source .venv-vllm/bin/activate
vllm serve ISTA-DASLab/Kimi-K2.5-P48NVFP4-MoESQ --trust-remote-code \
  --tensor-parallel-size 8 --enable-expert-parallel --kv-cache-dtype bfloat16   # 8x B200
```

The backend is selected automatically, and kernel tactics are autotuned during warmup.

- **Multi-GPU layouts:**
  - **TP + EP** (`--tensor-parallel-size N --enable-expert-parallel`) needs no extra
    dependencies and is tested on all four model families.
  - **DP + EP** (`--data-parallel-size N --enable-expert-parallel --all2all-backend
    flashinfer_nvlink_two_sided`, or `deepep_low_latency` with
    [DeepEP](https://github.com/deepseek-ai/DeepEP)). For Kimi-K2.5 prefer TP + EP; see
    the integration README.
  - Always keep `--enable-expert-parallel` on with TP.
- **More detail:** [`integrations/vllm/README.md`](integrations/vllm/README.md) covers
  options, limitations and validation results.

## Citation

If you use MoESQ, please cite the paper ([DOI: 10.48550/arXiv.2610.02241](https://doi.org/10.48550/arXiv.2610.02241)):

```bibtex
@misc{lee2026hardwarenativejointsparsequantizationtrillionscale,
      title={Hardware-Native Joint Sparse-Quantization for Trillion-Scale Mixture-of-Experts},
      author={Kwanhee Lee and Namhoon Lee and Dan Alistarh},
      year={2026},
      eprint={2610.02241},
      archivePrefix={arXiv},
      primaryClass={cs.AR},
      doi={10.48550/arXiv.2610.02241},
      url={https://arxiv.org/abs/2610.02241},
}
```

## Contact

[kwanhee.lee@postech.ac.kr](mailto:kwanhee.lee@postech.ac.kr)
