"""Assemble quantized per-expert checkpoint files into a HuggingFace-compatible model.

Usage:
    python save_model.py --config configs/p48_nvfp4_refine_actq.yaml                     # uses latest run
    python save_model.py --config configs/p48_nvfp4_refine_actq.yaml --run-id <run_id>   # specific run
    python save_model.py --config configs/p48_nvfp4_refine_actq.yaml --out-dir ./output  # custom output dir
    python save_model.py --config configs/p48_nvfp4_refine_actq.yaml --no-sparse-storage # dense experts

Paired-4:8 NVFP4 experts are written in paired48 sparse storage by default
(``weight_sparse_packed`` + ``weight_sparse_mask``, about 0.65x the expert bytes);
``--no-sparse-storage`` keeps dense ``weight_packed`` with the pruned pairs zeroed.
"""
import os
import re
import json
import math
import struct
import shutil
import argparse
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from safetensors.torch import safe_open, save_file
from transformers import AutoConfig
from src.config import load_config
from src.compression.quant.nvfp4 import dense_scale_groupsize
from src.compression import paired48_sparse

EXPERT_WEIGHT_PACKED_RE = re.compile(r"\.experts\.\d+\.[A-Za-z0-9_]+\.weight_packed$")


def resolve_model_path(model_name):
    p = Path(model_name)
    if p.exists():
        return str(p.parent.resolve()) if p.is_file() else str(p.resolve())
    try:
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(repo_id=model_name)
        return str(Path(local_dir).resolve())
    except Exception as e:
        raise FileNotFoundError(
            f"Could not resolve '{model_name}' as a local path or HF Hub repo. "
            f"Original error: {e}"
        )


def resolve_decompress_device(device_arg):
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested --decompress-device={device_arg}, but CUDA is not available."
        )
    return device_arg


def decompress_sparse_to_dense(compressed, bitmask, shape, device):
    target_shape = tuple(int(x) for x in shape.flatten().tolist())
    unpacked = np.unpackbits(
        bitmask.cpu().numpy(),
        axis=-1,
        count=target_shape[-1],
        bitorder="little",
    )
    mask = torch.from_numpy(unpacked.reshape(target_shape).astype(bool)).to(device)
    values = compressed.to(device).flatten()
    dense = torch.zeros(target_shape, dtype=compressed.dtype, device=device)
    dense[mask] = values
    return dense


def to_paired48_sparse_storage(tensors):
    """Re-encode one shard's dense per-expert ``weight_packed`` as paired48 sparse.

    Each ``*.experts.<e>.<proj>.weight_packed`` becomes ``weight_sparse_packed`` +
    ``weight_sparse_mask`` (see src/compression/paired48_sparse.py) and is round-trip
    checked; every other tensor, scales included, is kept as is.

    Returns:
        (tensors, number of expert projections converted).
    """
    out = {}
    n_converted = 0
    for key, t in tensors.items():
        if not EXPERT_WEIGHT_PACKED_RE.search(key):
            out[key] = t
            continue
        values, mask = paired48_sparse.compress_pair_bitmask(t)
        rebuilt, _ = paired48_sparse.decompress_pair_bitmask(values, mask, strict=True)
        if not torch.equal(rebuilt, t):
            raise RuntimeError(f"paired48 sparse round trip mismatch on {key}")
        base = key[: -len("weight_packed")]
        out[base + paired48_sparse.PACKED_SUFFIX] = values
        out[base + paired48_sparse.MASK_SUFFIX] = mask
        n_converted += 1
    return out, n_converted


def find_latest_run(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        return None
    candidates = []
    for name in os.listdir(checkpoint_dir):
        run_path = os.path.join(checkpoint_dir, name)
        prog_path = os.path.join(run_path, "progress.json")
        if os.path.isdir(run_path) and os.path.exists(prog_path):
            mtime = os.path.getmtime(prog_path)
            candidates.append((mtime, name))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


_ST_DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}


def _safetensors_header(path):
    """Read a safetensors header without mapping any tensor data."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


def collect_base_model_entries(base_model_dir):
    """Collect ALL tensors from the original base model.

    Sizes come from the safetensors HEADER (shape and dtype), not from loading each
    tensor, so collecting entries does not read the base model's weights off disk.
    """
    entries = {}
    for fname in sorted(os.listdir(base_model_dir)):
        if not fname.endswith(".safetensors"):
            continue
        path = os.path.join(base_model_dir, fname)
        for k, meta in _safetensors_header(path).items():
            m = re.search(r"\.layers\.(\d+)\.", k)
            nbytes = _ST_DTYPE_BYTES.get(meta["dtype"], 2)
            for d in meta["shape"]:
                nbytes *= d
            entries[k] = {
                "file": path,
                "key": k,
                "nbytes": nbytes,
                "layer_idx": int(m.group(1)) if m else None,
            }
    return entries


def collect_quantized_entries(per_expert_dir, nvfp4=False):
    """Collect quantized tensors from the run checkpoint directory.

    Picks up all .safetensors files (experts, shared_experts, etc.) except
    progress.json and other non-tensor files. Each file's layer index is
    extracted from its name.
    """
    entries = {}
    compression_formats = set()
    compressed_weight_keys = set()
    sparse_dense_modules = set()
    for fname in sorted(os.listdir(per_expert_dir)):
        if not fname.endswith(".safetensors"):
            continue
        m = re.search(r"layers_(\d+)", fname)
        if m is None:
            continue
        layer_idx = int(m.group(1))
        path = os.path.join(per_expert_dir, fname)
        with safe_open(path, framework="pt", device="cpu") as f:
            tensor_names = set(f.keys())
            ## `weight_shape` belongs to the INTEGER pack-quantized format. An
            ## nvfp4-pack-quantized checkpoint has no parameter for it -- vLLM's NVFP4
            ## MoE method registers only w13/w2_{weight_packed, weight_scale,
            ## weight_global_scale, input_global_scale} -- so shipping one fails the
            ## load with KeyError on `experts.w2_weight_shape`.
            ##
            ## Co-location with `.weight_global_scale` is NOT a reliable NVFP4 test:
            ## some wrappers (kimi_k25) keep the global scale out of the shard until
            ## assembly. Key on the run's quant_type instead, and keep the co-location
            ## test as an independent fallback for callers that do not pass it.
            nvfp4_weight_shape_keys = {
                f"{k[: -len('.weight_packed')]}.weight_shape"
                for k in tensor_names
                if k.endswith(".weight_packed")
                and (
                    nvfp4
                    or f"{k[: -len('.weight_packed')]}.weight_global_scale" in tensor_names
                )
            }
            for k in f.keys():
                if (
                    k.endswith(".bitmask")
                    or k.endswith(".row_offsets")
                    or k.endswith(".shape")
                    or k in nvfp4_weight_shape_keys
                ):
                    continue

                if k.endswith(".compressed"):
                    base = k[: -len(".compressed")]
                    shape_key = base + ".shape"
                    shape = tuple(int(x) for x in f.get_tensor(shape_key).flatten().tolist())
                    dense_key = base + ".weight"
                    compressed_weight_keys.add(dense_key)
                    sparse_dense_modules.add(base)
                    if f"{base}.row_offsets" in tensor_names:
                        compression_formats.add("sparse-bitmask")
                        row_offsets_key = f"{base}.row_offsets"
                    else:
                        compression_formats.add("sparse-24-bitmask")
                        row_offsets_key = None
                    t = f.get_tensor(k)
                    entries[dense_key] = {
                        "file": path,
                        "key": dense_key,
                        "nbytes": math.prod(shape) * t.element_size(),
                        "layer_idx": layer_idx,
                        "source_kind": "sparse_dense",
                        "compressed_key": k,
                        "bitmask_key": base + ".bitmask",
                        "shape_key": shape_key,
                        "row_offsets_key": row_offsets_key,
                    }
                    continue

                t = f.get_tensor(k)
                if k.endswith(".weight_packed"):
                    base = k[: -len(".weight_packed")]
                    if f"{base}.weight_global_scale" in tensor_names:
                        compression_formats.add("nvfp4-pack-quantized")
                    else:
                        compression_formats.add("pack-quantized")
                    compressed_weight_keys.add(base + ".weight")
                elif (
                    k.endswith(".weight")
                    and t.dtype in (torch.int32, torch.int64)
                    and f"{k}_scale" in tensor_names
                ):
                    # GSQ ships Humming's uint2 layout -- int32 [out, in/16] grid
                    # codes plus a bf16 group scale -- under the plain `.weight`
                    # key rather than NVFP4's weight_packed set. It must be
                    # recorded here, or inject_compression_config would drop
                    # quantization_config and declare the int32 codes as bf16.
                    compression_formats.add("humming-uint2")
                    compressed_weight_keys.add(k)
                entries[k] = {
                    "file": path,
                    "key": k,
                    "nbytes": t.numel() * t.element_size(),
                    "layer_idx": layer_idx,
                }
    return entries, compression_formats, compressed_weight_keys, sparse_dense_modules


def align_expert_namespace(base_entries, quantized_entries, compressed_weight_keys):
    """Reconcile the wrapper's expert naming with the checkpoint's, for nested MoE models.

    Qwen3.5-397B (Qwen3_5MoeForConditionalGeneration) stores its experts FUSED as two 3D
    parameters per layer, under a `language_model` nesting:

        model.language_model.layers.N.mlp.experts.gate_up_proj   [E, 2*inter, hidden]
        model.language_model.layers.N.mlp.experts.down_proj      [E, hidden, inter]

    while the expert-parallel wrapper writes its shards per expert, without that nesting:

        model.layers.N.mlp.experts.E.gate_proj.weight_packed

    Neither name matches the other, with two consequences that compound. The base model's
    bf16 fused experts are never filtered out (compressed_weight_keys holds per-expert
    names), and the packed tensors land in a namespace the architecture does not have --
    vLLM's WeightsMapper remaps `model.language_model.` to `language_model.model.` and
    leaves `model.layers.` alone, so it finds no parameter for them. Without the remap
    the assembled checkpoint would load as the original bf16 model with the compression
    silently ignored.

    Dropping the fused tensors is also load-correctness, not just size: vLLM sets
    is_fused_expert on seeing ANY `experts.gate_up_proj`, which would switch the whole
    load onto the fused path.
    """
    NEST = "model.language_model.layers."
    FLAT = "model.layers."
    base_nested = any(k.startswith(NEST) for k in base_entries)
    quant_flat = any(k.startswith(FLAT) for k in quantized_entries)
    if not (base_nested and quant_flat):
        return quantized_entries, compressed_weight_keys, set()

    remapped = {}
    for k, v in quantized_entries.items():
        nk = NEST + k[len(FLAT):] if k.startswith(FLAT) else k
        v = dict(v)
        # `key` is the name we WRITE; `src_key` is the name to READ from the shard,
        # which only knows the original name.
        v["src_key"] = v.get("src_key", k)
        v["key"] = nk
        remapped[nk] = v
    new_cwk = {NEST + k[len(FLAT):] if k.startswith(FLAT) else k
               for k in compressed_weight_keys}

    # Fused base tensors for any layer that now has per-expert replacements.
    layers_with_experts = {
        m.group(1)
        for k in remapped
        if (m := re.match(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.\d+\.", k))
    }
    fused_re = re.compile(
        r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$")
    superseded = {
        k for k in base_entries
        if (m := fused_re.match(k)) and m.group(1) in layers_with_experts
    }
    print(f"  Namespace: remapped {len(remapped)} quantized keys "
          f"{FLAT}* -> {NEST}*")
    print(f"  Namespace: dropping {len(superseded)} fused bf16 expert tensors "
          f"superseded by per-expert compressed weights ({len(layers_with_experts)} layers)")
    return remapped, new_cwk, superseded


def copy_non_weight_files(base_model_dir, out_dir):
    """Copy config, tokenizer, and other metadata files from the base model."""
    patterns = [
        "config.json", "generation_config.json",
        "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "vocab.json", "merges.txt",
        "tokenizer.model", "tiktoken.model",
        "preprocessor_config.json",
        "chat_template.jinja",
    ]
    copied = []
    for fname in os.listdir(base_model_dir):
        if fname in patterns or fname.endswith(".py") or fname.endswith(".jinja"):
            src = os.path.join(base_model_dir, fname)
            dst = os.path.join(out_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                copied.append(fname)
    return copied


def _build_ignore_list(model_config):
    """Build the quantization ignore list based on model architecture.

    Only MLP expert projections (gate_proj, up_proj, down_proj) are quantized.
    Everything else -- attention, norms, embeddings, routing gates, shared
    experts, vision towers -- must be excluded.
    """
    ignore = [
        "lm_head",
        "re:.*embed_tokens.*",
        "re:.*self_attn.*",
        "re:.*input_layernorm.*",
        "re:.*post_attention_layernorm.*",
        r"re:.*\.norm$",
    ]

    text_config = model_config.get("text_config", {})
    model_type = model_config.get("model_type", "")

    if "vision_config" in model_config or "vision_tower" in model_type:
        ignore.append("re:.*vision_tower.*")
        ignore.append("re:.*mm_projector.*")
        # Qwen3.5 / Qwen3-VL name the tower `visual.`, not `vision_tower.`. The tower
        # is never quantized (only routed experts are), so it must be excluded or
        # `targets: ["Linear"]` matches its Linears and vLLM finds no scheme for them.
        # Both spellings are kept because the mapper strips the `model.` prefix.
        ignore.append("re:.*visual\\..*")
        ignore.append("re:.*\\bvisual\\.merger\\..*")

    merged = {**model_config, **text_config}

    has_linear_attn = "layer_types" in merged
    if has_linear_attn:
        ignore.append("re:.*linear_attn.*")

    is_moe = merged.get("num_experts", 0) > 0 or merged.get("n_routed_experts", 0) > 0
    if is_moe:
        ignore.append(r"re:.*mlp\.gate$")
        ignore.append(r"re:.*block_sparse_moe\.gate$")

    has_shared_expert = (merged.get("n_shared_experts", 0) > 0
                         or (merged.get("num_shared_experts") or 0) > 0
                         or merged.get("shared_expert_intermediate_size", 0) > 0
                         or "qwen3_5" in model_type.lower())
    if has_shared_expert:
        ignore.append("re:.*shared_expert.*")

    # Kimi-K3: the latent-MoE projections around the routed experts and the Attention
    # Residual projections stay bf16.
    if merged.get("routed_expert_hidden_size"):
        ignore.append(r"re:.*routed_expert_(down|up)_proj$")
    if merged.get("attn_res_block_size"):
        ignore.append(r"re:.*_res_proj$")
        # its lm_head sits at `language_model.lm_head`, which the literal entry misses
        ignore.append(r"re:.*lm_head$")

    # qwen4_exp (Qwen3.8-Flash-Next): the gated-residual hyper-connections and the
    # per-layer n-gram embedding block are Linears that stay bf16.
    if merged.get("hc_count"):
        ignore.append("re:.*hyper_connection.*")
    if merged.get("ple_layer_ids"):
        ignore.append(r"re:.*\.ple\..*")

    # The multi-token-prediction head is never compressed by our pipeline -- its experts
    # ship bf16 -- so it must not be claimed by `targets: ["Linear"]`. ISTA-DASLab's own
    # published MoE quant configs carry the same exclusion.
    if any(k.startswith("mtp") for k in (model_config.get("architectures") or [])) or \
       merged.get("num_nextn_predict_layers") or merged.get("mtp_num_hidden_layers") or \
       merged.get("mtp_config") is not None:
        ignore.append("re:.*mtp.*")

    first_k_dense = merged.get("first_k_dense_replace", 0)
    if first_k_dense > 0:
        dense_indices = "|".join(str(i) for i in range(first_k_dense))
        ignore.append(f"re:.*layers\\.({dense_indices})\\.mlp\\.(gate_proj|up_proj|down_proj|gate_up_proj).*")

    return ignore


def inject_compression_config(
    out_dir,
    groupsize,
    compression_formats,
    sparse_dense_modules=None,
    wbits=4,
    fake_quantize_activations=False,
    paired48_sparse_storage=False,
):
    """Add compressed-tensors config metadata to config.json.

    This enables vLLM and HuggingFace transformers to auto-detect the
    compressed format when loading the assembled model. With
    ``paired48_sparse_storage`` the NVFP4 config also carries the marker that tells
    the patched vLLM to load the experts from paired48 sparse storage.
    """
    config_path = os.path.join(out_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    ignore = _build_ignore_list(config)
    sparse_dense_modules = sorted(sparse_dense_modules or [])
    quantization_config = None

    if "humming-uint2" in compression_formats:
        # Schema copied from ISTA-DASLab/Kimi-K2.5-2Bit-GSQ's config.json, which is
        # what the `humming` vLLM quant_method reads.
        # NOTE the key is weight_scale_group_size, NOT group_size.
        quantization_config = {
            "quant_method": "humming",
            "b_dtype": f"uint{wbits}",
            "weight_scale_group_size": groupsize,
            "weight_scale_type": "group",
            "has_zero_point": False,
            "ignore": list(ignore),
        }
    elif "nvfp4-pack-quantized" in compression_formats:
        quant_ignore = list(ignore)
        quant_ignore.extend(sparse_dense_modules)
        input_activations = None
        if fake_quantize_activations:
            input_activations = {
                "num_bits": 4,
                "type": "float",
                "strategy": "tensor_group",
                "group_size": groupsize,
                "symmetric": True,
                "dynamic": "local",
                "observer": "static_minmax",
                "observer_kwargs": {},
                "scale_dtype": "torch.float8_e4m3fn",
                "zp_dtype": "torch.float8_e4m3fn",
            }
        quantization_config = {
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {
                    "input_activations": input_activations,
                    "output_activations": None,
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "strategy": "tensor_group",
                        "group_size": groupsize,
                        "symmetric": True,
                        "type": "float",
                        "dynamic": False,
                        "observer": "memoryless_minmax",
                        "observer_kwargs": {},
                        "scale_dtype": "torch.float8_e4m3fn",
                    },
                }
            },
            "format": "nvfp4-pack-quantized",
            "ignore": quant_ignore,
            "quantization_status": "compressed",
        }
        if paired48_sparse_storage:
            quantization_config[paired48_sparse.CONFIG_KEY] = paired48_sparse.make_marker()
    elif "pack-quantized" in compression_formats:
        quant_ignore = list(ignore)
        quant_ignore.extend(sparse_dense_modules)
        quantization_config = {
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {
                    "input_activations": None,
                    "output_activations": None,
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": wbits,
                        "strategy": "group",
                        "group_size": groupsize,
                        "symmetric": True,
                        "type": "int",
                    },
                }
            },
            "format": "pack-quantized",
            "ignore": quant_ignore,
            "quantization_status": "compressed",
        }

    if quantization_config is not None:
        config["quantization_config"] = quantization_config
    else:
        config.pop("quantization_config", None)

    config.pop("sparsity_config", None)

    if "text_config" in config and "quantization_config" in config["text_config"]:
        del config["text_config"]["quantization_config"]
    if "text_config" in config and "sparsity_config" in config["text_config"]:
        del config["text_config"]["sparsity_config"]

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Assemble quantized expert weights into a HF-compatible model checkpoint")
    parser.add_argument("--config", type=str, default="configs/p48_nvfp4_refine_actq.yaml",
                        help="Path to the training config YAML")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run ID to export. Defaults to the latest completed run.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory. Defaults to <checkpoint_dir>/<run_id>/assembled")
    parser.add_argument("--decompress-device", type=str, default="auto",
                        help="Device used for sparse decompression: auto, cpu, cuda, cuda:N")
    parser.add_argument("--sparse-storage", action=argparse.BooleanOptionalAction, default=None,
                        help="Write paired-4:8 NVFP4 experts in paired48 sparse storage "
                             "(weight_sparse_packed + weight_sparse_mask). On by default for "
                             "paired-4:8 NVFP4 runs; --no-sparse-storage writes dense weight_packed.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # paired48 sparse storage encodes "2 kept bytes of every 4" in packed NVFP4, i.e.
    # paired 4:8 (the only 4:8 the config accepts) with the group-32 dense scale span.
    sparse_storage_ok = (
        cfg.compression.quant_type == "nvfp4"
        and (cfg.compression.prunen, cfg.compression.prunem) == (4, 8)
        and dense_scale_groupsize(4, 8, cfg.compression.groupsize) == 32
    )
    if args.sparse_storage and not sparse_storage_ok:
        raise ValueError(
            "--sparse-storage needs a paired-4:8 NVFP4 run with group_size 32, got "
            f"quant_type={cfg.compression.quant_type!r} "
            f"prunen:prunem={cfg.compression.prunen}:{cfg.compression.prunem} "
            f"groupsize={cfg.compression.groupsize}")
    sparse_storage = sparse_storage_ok if args.sparse_storage is None else args.sparse_storage

    model_name = cfg.model.name
    checkpoint_dir = cfg.training.checkpoint_dir

    if args.run_id is not None:
        run_id = args.run_id
    else:
        run_id = find_latest_run(checkpoint_dir)
        if run_id is None:
            raise RuntimeError(
                f"No completed runs found in '{checkpoint_dir}'. "
                "Pass --run-id explicitly or complete a training run first.")

    run_dir = os.path.join(checkpoint_dir, run_id)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    out_dir = args.out_dir or os.path.join(run_dir, "assembled")
    decompress_device = resolve_decompress_device(args.decompress_device)

    print(f"Model:        {model_name}")
    print(f"Run ID:       {run_id}")
    print(f"Run dir:      {run_dir}")
    print(f"Output dir:   {out_dir}")
    print(f"Experts:      {'paired48 sparse storage' if sparse_storage else 'dense weight_packed'}")
    print(f"Decomp dev:   {decompress_device}")
    print()

    base_model_dir = resolve_model_path(model_name)
    cfg_hf = AutoConfig.from_pretrained(base_model_dir, trust_remote_code=True)
    num_layers = getattr(cfg_hf, "num_hidden_layers", None)
    if num_layers is None and hasattr(cfg_hf, "text_config"):
        num_layers = cfg_hf.text_config.num_hidden_layers
    if num_layers is None:
        raise RuntimeError(
            f"Cannot determine num_hidden_layers from {base_model_dir}/config.json. "
            "Check model config structure.")
    num_shards = num_layers + 1

    print(f"Base model:   {base_model_dir}")
    print(f"Layers:       {num_layers}")
    print(f"Shards:       {num_shards}")
    print()

    print("Collecting base model tensors...")
    base_entries = collect_base_model_entries(base_model_dir)
    print(f"  Found {len(base_entries)} tensors")

    print("Collecting quantized tensors...")
    quantized_entries, compression_formats, compressed_weight_keys, sparse_dense_modules = collect_quantized_entries(
        run_dir, nvfp4=(cfg.compression.quant_type == "nvfp4")
    )
    print(f"  Found {len(quantized_entries)} tensors")

    if not quantized_entries:
        raise RuntimeError(
            f"No quantized shard files found in {run_dir}. "
            "Make sure the training run completed at least one layer.")

    replaced_dense = set(base_entries.keys()) & compressed_weight_keys
    print(f"  Replacing {len(replaced_dense)} dense weight tensors with exported versions")
    if compression_formats:
        print(f"  Detected compression formats: {', '.join(sorted(compression_formats))}")
    if sparse_dense_modules:
        print(f"  Decompressing {len(sparse_dense_modules)} sparse modules to dense weights")

    quantized_entries, compressed_weight_keys, superseded_fused = align_expert_namespace(
        base_entries, quantized_entries, compressed_weight_keys
    )

    filtered_base_entries = {
        k: v for k, v in base_entries.items()
        if k not in compressed_weight_keys and k not in superseded_fused
    }
    merged = {**filtered_base_entries, **quantized_entries}
    all_entries = list(merged.values())

    for e in all_entries:
        if e["layer_idx"] is None:
            e["shard_idx"] = 0
        else:
            e["shard_idx"] = e["layer_idx"] + 1

    shard_sizes = [0] * num_shards
    for e in all_entries:
        shard_sizes[e["shard_idx"]] += e["nbytes"]

    print("\nShard sizes:")
    for idx, size in enumerate(shard_sizes):
        if size > 0:
            print(f"  shard {idx:03d}: {size / (1024**3):.3f} GB")

    os.makedirs(out_dir, exist_ok=True)

    def make_shard_name(idx):
        return f"model-{idx+1:05d}-of-{num_shards:05d}.safetensors"

    entries_by_file = defaultdict(list)
    for e in all_entries:
        entries_by_file[e["file"]].append(e)

    shard_tensors = [dict() for _ in range(num_shards)]
    weight_map = {}
    total_size = 0

    print("\nLoading tensors...")
    for src_file, file_entries in entries_by_file.items():
        with safe_open(src_file, framework="pt", device="cpu") as f:
            for e in file_entries:
                if e.get("source_kind") == "sparse_dense":
                    t = decompress_sparse_to_dense(
                        compressed=f.get_tensor(e["compressed_key"]),
                        bitmask=f.get_tensor(e["bitmask_key"]),
                        shape=f.get_tensor(e["shape_key"]),
                        device=decompress_device,
                    ).cpu()
                else:
                    t = f.get_tensor(e.get("src_key", e["key"]))
                shard_idx = e["shard_idx"]
                shard_name = make_shard_name(shard_idx)
                shard_tensors[shard_idx][e["key"]] = t
                weight_map[e["key"]] = shard_name
                total_size += e["nbytes"]

    if cfg.compression.fake_quantize_activations:
        # Pure-dynamic NVFP4 activation quant: at inference, vLLM's `dynamic:
        # "local"` path computes per-group fp8 microscales as
        # `fp8_project(amax_g · input_global_scale / FP4_MAX)`. With
        # input_global_scale = 1.0 this matches the training-time formula in
        # `fake_quantize_activation_nvfp4`.
        nvfp4_modules = set()
        for key in quantized_entries:
            if key.endswith(".weight_global_scale"):
                nvfp4_modules.add(key[: -len(".weight_global_scale")])
        print(f"  Injecting input_global_scale=1.0 for {len(nvfp4_modules)} NVFP4 Linears (fake_quantize_activations=True)")
        for module in nvfp4_modules:
            scale_key = f"{module}.input_global_scale"
            ref_key = f"{module}.weight_global_scale"
            ref_entry = quantized_entries[ref_key]
            shard_idx = ref_entry["layer_idx"] + 1 if ref_entry["layer_idx"] is not None else 0
            shard_name = make_shard_name(shard_idx)
            t = torch.tensor(1.0, dtype=torch.float32)
            shard_tensors[shard_idx][scale_key] = t
            weight_map[scale_key] = shard_name
            total_size += t.numel() * t.element_size()

    print(f"\nWriting {num_shards} shards to {out_dir}...")
    n_sparse = 0
    for shard_idx in range(num_shards):
        if not shard_tensors[shard_idx]:
            continue
        shard_name = make_shard_name(shard_idx)
        shard_path = os.path.join(out_dir, shard_name)
        if sparse_storage:
            dense = shard_tensors[shard_idx]
            sparse, n = to_paired48_sparse_storage(dense)
            for key in dense.keys() - sparse.keys():
                del weight_map[key]
                total_size -= dense[key].numel() * dense[key].element_size()
            for key in sparse.keys() - dense.keys():
                weight_map[key] = shard_name
                total_size += sparse[key].numel() * sparse[key].element_size()
            shard_tensors[shard_idx] = sparse
            n_sparse += n
        print(f"  {shard_name}: {len(shard_tensors[shard_idx])} tensors")
        save_file(shard_tensors[shard_idx], shard_path)
    if sparse_storage:
        if n_sparse == 0:
            raise RuntimeError(
                "sparse storage requested but no *.experts.<e>.<proj>.weight_packed "
                "tensors were found; pass --no-sparse-storage for this run")
        print(f"  {n_sparse} expert projections written in paired48 sparse storage")

    index = {
        "metadata": {"total_size": int(total_size)},
        "weight_map": weight_map,
    }
    index_path = os.path.join(out_dir, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print("\nCopying config & tokenizer files...")
    copied = copy_non_weight_files(base_model_dir, out_dir)
    for fname in copied:
        print(f"  {fname}")

    groupsize = cfg.compression.groupsize
    if cfg.compression.quant_type == "nvfp4":
        # Scales are indexed by dense columns (sparse storage only re-encodes the
        # packed values): the per-group scale spans compression.groupsize *
        # prunem/prunen dense columns, so the exported group_size must match that
        # dense span rather than the per-16-compressed kernel block.
        groupsize = dense_scale_groupsize(
            cfg.compression.prunen, cfg.compression.prunem, cfg.compression.groupsize
        )
    print(f"\nInjecting compressed-tensors config into config.json...")
    inject_compression_config(
        out_dir,
        groupsize,
        compression_formats,
        sparse_dense_modules=sparse_dense_modules,
        wbits=cfg.init.wbits,
        fake_quantize_activations=cfg.compression.fake_quantize_activations,
        paired48_sparse_storage=sparse_storage,
    )

    print(f"\nDone. Total model size: ~{total_size / (1024**3):.2f} GB")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
