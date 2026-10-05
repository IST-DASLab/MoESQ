# Configurations

Configs are strict YAML loaded by `load_config()` in [`../src/config.py`](../src/config.py).
Unknown sections or keys raise an error.

[`p48_nvfp4_refine_actq.yaml`](p48_nvfp4_refine_actq.yaml) is the generic starting point: paired-4:8 + NVFP4 W4A4 with refinement, on Qwen3-30B-A3B.

## Paper configurations

| Arm | Qwen3-30B-A3B | Qwen3.5-397B-A17B | Kimi-K2.5 |
|---|---|---|---|
| **Ours** (paired-4:8 NVFP4 W4A4, gate-weighted refinement, p=2) | `qwen3_30b/ours_gw2.yaml` | `qwen35_397b/ours_gw2.yaml` | `kimi_k25/ours_gw2.yaml` |
| Ours, unweighted loss (p=0) | `qwen3_30b/ours_gw0.yaml` | | |
| SparseGPTQ + NVFP4 (one-shot) | `qwen3_30b/sgptq.yaml` | `qwen35_397b/sgptq.yaml` | `kimi_k25/sgptq.yaml` |
| OBR | `qwen3_30b/obr.yaml` | `qwen35_397b/obr.yaml` | `kimi_k25/obr.yaml` |
| JSQ | `qwen3_30b/jsq.yaml` | `qwen35_397b/jsq.yaml` | `kimi_k25/jsq.yaml` |
| GSQ dense 2-bit (reference) | `qwen3_30b/gsq2_dense.yaml` | `qwen35_397b/gsq2_dense.yaml` | |
| Dense NVFP4 W4A4 (reference) | `qwen3_30b/dense_nvfp4_actq.yaml` | | |

Starting points for newer models, with the settings of the closest paper arm (not tuned):
`qwen38_flash_next/ours_gw2.yaml` (from Qwen3.5) and `kimi_k3/ours_gw2.yaml` (from Kimi-K2.5,
with fewer calibration samples because Attention Residuals multiply the activation caches).

Ablations on Qwen3-30B-A3B are in [`qwen3_30b/ablations/`](qwen3_30b/ablations/):

- gate-weight exponent: `gw1`, `gw0_lrhalf`
- support × values study: `sqabl_arm1_init` … `sqabl_arm4_joint`, `sqabl_arm4_joint_tuned`

## Sections

```yaml
model:         # HF model id or local path + dtype
data:          # calibration dataset, batch_size, num_samples, ...
compression:   # what we compress to
init:          # how we initialize (GPTQ / OBR / JSQ / RTN)
refine:        # Gumbel-Softmax + fake-quant refinement
training:      # pipeline + checkpointing
eval:          # PPL eval split / lm-eval task defaults
wandb:         # WandB tracking
logging:       # console + log_dir verbosity
distributed:   # NCCL timeout
```

## Key knobs

| Key | Meaning |
|---|---|
| `compression.prunen` / `prunem` | Sparsity pattern: `4 / 8` (paired) or `2 / 4`. |
| `compression.quant_type` | One of:<br>• `"nvfp4"`: sparse + quantized, the main path.<br>• `"gsq"`: dense 2-bit reference. It exports `quant_method: "humming"`, which needs the Humming kernels and is not served by the paired48 vLLM integration.<br>• `null`: sparsity only.<br>`gsq` and `null` need `fake_quantize_activations: false`. |
| `compression.groupsize` | NVFP4 FP8-scale block on the *compressed* weight (16). With paired-4:8 the dense scale group is 32, which the exported `quantization_config` records (`group_size: 32`, hence the `GS32` model names). |
| `compression.learn_weight_values` | Learn dense expert weight values jointly with the masks. Requires `refine.enabled`. |
| `compression.fake_quantize_activations` | Default `true`. Applies NVFP4 dynamic per-group fake-quant to expert activations during training, as at inference, so the export is W4A4. The scales are recomputed each step and have no learnable parameters. `false` gives weight-only (W4A16). |
| `init.method` | One of:<br>• `"gptq"` (default). For sparse schemes it also picks the N:M mask.<br>• `"obr"` or `"jsq"`: one-shot baselines.<br>• `"rtn"`, `"random"`. |
| `refine.enabled` | `false` runs init only (SparseGPTQ / GPTQ baseline). `true` adds the Gumbel-Softmax refinement. |
| `refine.temperature`, `refine.scale` | Gumbel temperature and logit-scale schedules, `[start, end]`, annealed linearly per step. |
| `data.dataset_name` | Loader in [`../src/data/dataset.py`](../src/data/dataset.py): `c4`, `fineweb_edu`, `open_thoughts`, `mixed` or `nemotron_mix`.<br>The paper configs use `mixed`. It splits the calibration chunks across `neuralmagic/LLM_compression_calibration`, `open-thoughts/OpenThoughts-114k` and `HuggingFaceFW/fineweb-edu` (`sample-10BT`), weighted by `data.mixed_source_weights` (default `[0.1, 0.45, 0.45]`). |
| `training.checkpoint_dir` | Run directory root. For multi-node runs it must be on a filesystem visible to every node. `RUN_ROOT` in `scripts/slurm.sh` overrides it. |
