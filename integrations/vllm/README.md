# Serving MoESQ checkpoints with vLLM

MoESQ checkpoints (paired-4:8 sparse NVFP4 experts, `group_size: 32`) need a MoE
backend that upstream vLLM does not have yet. This directory packages it as a small
patch on [vLLM v0.30.0](https://github.com/vllm-project/vllm/releases/tag/v0.30.0). The
expert GEMMs run on Blackwell (SM100 or SM120) sparse tensor cores through the kernels in
[`third_party/grouped-sparse-GEMM`](https://github.com/IST-DASLab/grouped-sparse-GEMM),
which are included as a git submodule pinned to release
[`v0.14.0`](https://github.com/IST-DASLab/grouped-sparse-GEMM/releases/tag/v0.14.0).

| File | Purpose |
|---|---|
| `moe-sq-v0.30.0.patch` | The `paired48_nvfp4` backend for vLLM v0.30.0: about 4.7k lines of Python in 27 files, no C++ |
| `install.sh` | Clones vLLM v0.30.0, applies the patch, installs the precompiled vLLM wheel, and builds the kernels against the same torch |

## Install

Requirements: an SM100 (B200/GB200) or SM120 (RTX 5090 / RTX PRO 6000) GPU,
[`uv`](https://docs.astral.sh/uv/), and a CUDA toolkit >= 12.8 for the kernel build.

```bash
git clone --recurse-submodules https://github.com/IST-DASLab/MoESQ.git && cd MoESQ
bash integrations/vllm/install.sh        # creates .venv-vllm, separate from the training venv
source .venv-vllm/bin/activate
```

`install.sh` installs torch 2.13 built for CUDA 13.0 (`TORCH_BACKEND=cu130`). If your
toolkit is another version, set `TORCH_BACKEND` to match it (for example `cu128` for
CUDA 12.8): the kernels link against the installed torch and must be built with the
same CUDA. The kernels are built for the GPU in the machine (`100a` or `120a`); set
`PAIRED_NVFP4_ARCHS` to override it, for example `"100a;120a"` for one build that runs on
both. vLLM stops at startup with a rebuild hint if the build lacks the GPU's arch.

## Serve

The backend is selected automatically for `group_size: 32` NVFP4 MoE checkpoints:

```bash
vllm serve ISTA-DASLab/Qwen3-30B-A3B-P48NVFP4-MoESQ
```

For multi-GPU runs:

| Layout | Flags | Tested on (GSM8K 5-shot, strict-match) |
|---|---|---|
| **TP + EP** | `--tensor-parallel-size N --enable-expert-parallel` | • Qwen3-30B-A3B, TP2: 0.834<br>• Qwen3.5-397B-A17B, TP8: 0.799<br>• Kimi-K2.5, TP8: 0.924<br>• Qwen3.8-Flash-Next, TP2: 0.943 (BF16 TP4: 0.969; chat template) |
| DP + EP | `--data-parallel-size N --enable-expert-parallel --all2all-backend flashinfer_nvlink_two_sided` (or `deepep_low_latency`) | Measured before the w1/w3 scale fold (below), so lower than TP + EP:<br>• Qwen3-30B-A3B, DP2: 0.793<br>• Qwen3.5-397B-A17B, DP4: 0.732<br>• Kimi-K2.5, DP8: 0.79-0.83 (open issue, see below) |

- Qwen3.8-Flash-Next is chat-only: plain few-shot completions end immediately (BF16
  too), so evaluate it through the chat template, e.g. `vllm serve ...
  --default-chat-template-kwargs '{"enable_thinking": false}'` and lm-eval
  `local-chat-completions --apply_chat_template --fewshot_as_multiturn`.
- TP + EP numbers: 3-seed means of the MoESQ (gw2) checkpoints, except Qwen3-30B-A3B
  (the released checkpoint, one run).
- MoESQ checkpoints quantize gate and up with separate global scales. Upstream vLLM
  collapses w13 to w1's scale, which rescales every expert's up_proj by g1/g3; the patch
  folds the ratio into the w13 block scales instead. On the released Qwen3-30B-A3B
  checkpoint this alone moves GSM8K from 0.795 to 0.834.

**TP + EP**
- Each GPU holds whole experts, and our CUDA-graph-safe dispatch handles token routing.
- No extra dependencies, so this is the recommended layout.
- Always pass `--enable-expert-parallel` with TP. Plain TP would split every expert's
  inner dimension and is not tested.

**DP + EP**
- `flashinfer_nvlink_two_sided` needs no extra dependencies and is the faster
  all2all: the experts take token-order activations and the dispatch carries FP4.
  `deepep_low_latency` also works; it requires
  [DeepEP](https://github.com/deepseek-ai/DeepEP) and caps the batch
  (`--max-num-batched-tokens` <= 511 unless `NVSHMEM_QP_DEPTH` is raised).
- Known issue: Kimi-K2.5 under DP8 with CUDA graphs scores about 5 points lower on
  GSM8K than under TP8 (0.79-0.83 vs 0.89), with either all2all and with dense or
  sparse storage. With `--enforce-eager` DP8 matches TP8 (0.89-0.90), so the gap comes
  from CUDA-graph replay under DP. The dense NVFP4 Kimi-K2.5 and the Qwen models do
  not show it. Use TP + EP for Kimi.

Kernel tactics are autotuned during warmup. To skip that and use the defaults, pass
`--kernel-config '{"enable_paired_nvfp4_autotune": false}'`.

Sparse-storage checkpoints (`paired48_sparse`, e.g. `ISTA-DASLab/Kimi-K2.5-P48NVFP4-MoESQ`)
load the same way. To convert between dense and sparse storage, run this inside
`.venv-vllm` (CPU only):

```bash
python build/vllm/tools/paired48_sparse_checkpoint.py {to-sparse,to-dense} SRC DST [--jobs N]
```

Replace `build/vllm` if you set `VLLM_DIR`. vLLM is an editable install from that
checkout, so don't delete it.

## Limitations

- SM100 or SM120. Other GPUs, including SM103 (B300) and SM121 (DGX Spark), are not
  supported.
- On SM120, clusters are always 1x1. GEMM1 uses the fused SwiGLU epilogue on SM120 too
  (kernels v0.13.0; the autotuner keeps it where it is faster). All released checkpoints
  serve on SM120 as well as SM100; the results below are from B200.
- bf16 model dtype; SiLU-gated experts, or SiTU-gated for Kimi-K3 (kernels v0.14.0, in both
  the unfused activation quantizer and the fused GEMM1 epilogue). The SiTU path is tested on
  SM100 only so far.
- EPLB and the `nixl_ep` all2all backend are not supported.
- DP > 1 needs an all2all backend (`flashinfer_nvlink_two_sided` or `deepep_low_latency`).

## Validation

All results below come from a fresh `install.sh` install on B200 with torch 2.13.0+cu130
and default flags (backend auto-selected, autotune on).

GSM8K (5-shot, strict-match, ±0.011) on [`ISTA-DASLab/Qwen3-30B-A3B-P48NVFP4-MoESQ`](https://huggingface.co/ISTA-DASLab/Qwen3-30B-A3B-P48NVFP4-MoESQ), measured before the w1/w3 scale fold, so lower than the 0.834 above. "Dense" is the same checkpoint before conversion to sparse storage; the two formats load to identical weights (on Qwen3.8-Flash-Next, sparse and the dense round trip score alike):

| Storage | 1× B200 | TP2 + EP (2× B200) |
|---|---|---|
| Dense packed NVFP4 | 0.789 | 0.792 |
| Sparse (`paired48_sparse`, converted with the tool above) | 0.798 | 0.794 |

Serving throughput on 1× B200 (Qwen3-30B-A3B P48 GS32, SparseGPTQ-initialized):

| Workload | Result |
|---|---|
| Decode-heavy (random 1024 in / 256 out, concurrency 64) | ~5.47k output tok/s |
| Prefill-heavy (random 4096 in / 8 out) | ~107k total tok/s |

Multi-GPU results are in the layout table above.
