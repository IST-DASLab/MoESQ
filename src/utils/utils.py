import json
import os
import time
import uuid
from copy import deepcopy

import torch

PROGRESS_FILENAME = "progress.json"


# ─────────────────────────────────────────────────────────────────────────────
# Run-id / progress.json checkpointing
# ─────────────────────────────────────────────────────────────────────────────

def generate_run_id():
    ts = time.strftime("%Y%m%d-%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"{ts}_{short}"


def run_dir(checkpoint_dir, run_id):
    return os.path.join(checkpoint_dir, run_id)


def progress_path(run_dir_path):
    return os.path.join(run_dir_path, PROGRESS_FILENAME)


def find_latest_run(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        return None
    candidates = []
    for name in os.listdir(checkpoint_dir):
        run_path = os.path.join(checkpoint_dir, name)
        prog_path = os.path.join(run_path, PROGRESS_FILENAME)
        if os.path.isdir(run_path) and os.path.exists(prog_path):
            mtime = os.path.getmtime(prog_path)
            candidates.append((mtime, name))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def load_progress(run_dir_path):
    path = progress_path(run_dir_path)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def save_progress(run_dir_path, layer_idx, run_id=None, wandb_run_id=None):
    os.makedirs(run_dir_path, exist_ok=True)
    progress = {
        "run_id": run_id,
        "last_completed_layer": layer_idx,
        "wandb_run_id": wandb_run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = progress_path(run_dir_path)
    tmp_path = path + ".tmp"
    with open(tmp_path, 'w') as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp_path, path)


# ─────────────────────────────────────────────────────────────────────────────
# Model wrapper dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def _checkpoint_model_type(model_name):
    """Architecture id from the checkpoint's own config, lowercased ('' if unknown).

    Name-substring dispatch alone is fragile: a repo whose name contains
    "qwen3" but not "qwen3.5" can still be a qwen3_5_moe_text checkpoint and
    would silently select Qwen3MoeWrapper. Keying on model_type first picks
    the wrapper from the architecture, whatever the repo is called.
    Returns "" on any failure so the name-based branches still apply.
    """
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        mt = getattr(cfg, "model_type", None)
        if not mt:
            tc = getattr(cfg, "text_config", None)
            mt = getattr(tc, "model_type", None) if tc is not None else None
        return (mt or "").lower()
    except Exception:
        return ""


def get_model_wrapper(model_name, tokenizer, batch_size, seqlen, device, dtype, world_size, dummy=False):
    name_lower = model_name.lower()
    model_type = _checkpoint_model_type(model_name)
    is_kimi_k25_arch = any(token in name_lower for token in ("k2.5", "k2_5", "k2.6", "k2_6"))
    if 'opt' in name_lower:
        from src.models.opt import OPTWrapper
        return OPTWrapper(model_name, tokenizer, batch_size, seqlen, device, dtype)
    elif 'llama' in name_lower:
        from src.models.llama import LLaMAWrapper
        return LLaMAWrapper(model_name, tokenizer, batch_size, seqlen, device, dtype)
    elif model_type.startswith('qwen4_exp'):
        # Qwen3.8-Flash-Next. Keyed on model_type only: the repo name says "Qwen3.8",
        # which would otherwise fall through to the Qwen3-MoE branch below.
        from src.models.qwen4_exp import Qwen4ExpWrapper
        return Qwen4ExpWrapper(model_name, tokenizer, batch_size, seqlen, device, dtype)
    elif model_type.startswith('qwen3_5_moe') or 'qwen3.5' in name_lower or 'qwen3_5' in name_lower:
        from src.models.qwen35_moe import Qwen35MoeWrapper
        return Qwen35MoeWrapper(model_name, tokenizer, batch_size, seqlen, device, dtype)
    elif model_type.startswith('qwen3_moe') or 'qwen3' in name_lower:
        # transformers >= 5 fuses Qwen3-MoE experts into 3D params; 4.x keeps a
        # per-expert ModuleList. The layout, not the version string, picks the wrapper.
        from src.models.qwen3_moe_fused import uses_fused_experts
        if uses_fused_experts():
            from src.models.qwen3_moe_fused import Qwen3MoeFusedWrapper
            return Qwen3MoeFusedWrapper(model_name, tokenizer, batch_size, seqlen, device, dtype)
        from src.models.qwen3_moe import Qwen3MoeWrapper
        return Qwen3MoeWrapper(model_name, tokenizer, batch_size, seqlen, device, dtype)
    elif model_type == 'kimi_k3':
        from src.models.kimi_k3 import KimiK3Wrapper
        return KimiK3Wrapper(model_name, tokenizer, batch_size, seqlen, device, dtype, dummy=dummy)
    elif is_kimi_k25_arch:
        # transformers >= 5.14 ships native Kimi_K25 classes (fused experts) and cannot
        # import the checkpoint's remote code; older versions have only the remote code.
        from src.models.kimi_k25_fused import native_kimi_k25_available
        if native_kimi_k25_available():
            from src.models.kimi_k25_fused import KimiK25FusedWrapper
            return KimiK25FusedWrapper(model_name, tokenizer, batch_size, seqlen, device, dtype, dummy=dummy)
        from src.models.kimi_k25 import KimiK25Wrapper
        return KimiK25Wrapper(model_name, tokenizer, batch_size, seqlen, device, dtype, dummy=dummy)
    elif 'kimi' in name_lower:
        from src.models.kimi_k2 import KimiK2Wrapper
        return KimiK2Wrapper(model_name, tokenizer, batch_size, seqlen, device, dtype)
    else:
        raise ValueError(f"Unsupported model family: {model_name}")


# ─────────────────────────────────────────────────────────────────────────────
# Activation-cache tensor / mmap utilities
# ─────────────────────────────────────────────────────────────────────────────

def create_act_cache(num_samples, seqlen, hidden_size, dtype, mmap_dir=None, mmap_threshold_bytes=2 * 1024 ** 3,
                     tag="act"):
    """``{'input': [num_samples, seqlen, hidden_size]}``, memory-mapped when large.

    ``tag`` names the mmap file, so several caches of one shape can coexist.
    """
    nbytes = num_samples * seqlen * hidden_size * 2  # 2 bytes for any 16-bit dtype
    if mmap_dir is not None and nbytes > mmap_threshold_bytes:
        import numpy as np
        os.makedirs(mmap_dir, exist_ok=True)
        path = os.path.join(mmap_dir, f"{tag}_{os.getpid()}_{num_samples}x{seqlen}x{hidden_size}.bin")
        arr = np.memmap(path, dtype='uint16', mode='w+', shape=(num_samples, seqlen, hidden_size))
        inps = torch.from_numpy(arr).view(dtype)
        return {'input': inps, '_mmap_path': path}
    inps = torch.zeros((num_samples, seqlen, hidden_size), dtype=dtype, device='cpu')
    return {'input': inps}


def cleanup_act_cache_mmap(*dicts):
    """Remove a cache's mmap file, plus any side caches a wrapper registered in it."""
    for d in dicts:
        paths = [d.pop('_mmap_path', None)] + d.pop('_mmap_paths', [])
        for path in paths:
            if path is not None and os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# Distributed env helpers
# ─────────────────────────────────────────────────────────────────────────────

def local_rank_from_env():
    return int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))


def local_world_size_from_env():
    if "LOCAL_WORLD_SIZE" in os.environ:
        return int(os.environ["LOCAL_WORLD_SIZE"])
    if "SLURM_GPUS_ON_NODE" in os.environ and os.environ["SLURM_GPUS_ON_NODE"].isdigit():
        return int(os.environ["SLURM_GPUS_ON_NODE"])
    if "SLURM_NTASKS_PER_NODE" in os.environ:
        raw = os.environ["SLURM_NTASKS_PER_NODE"].split("(")[0]
        if raw.isdigit():
            return int(raw)
    return max(1, torch.cuda.device_count() if torch.cuda.is_available() else 1)


def resolve_device():
    if not torch.cuda.is_available():
        return "cpu"
    device_count = torch.cuda.device_count()
    if device_count <= 0:
        return "cpu"
    device_id = min(local_rank_from_env(), device_count - 1)
    return f"cuda:{device_id}"


# ─────────────────────────────────────────────────────────────────────────────
# Config merging / wandb override helpers
# ─────────────────────────────────────────────────────────────────────────────

def to_plain_data(obj):
    if hasattr(obj, "as_dict") and callable(obj.as_dict):
        return to_plain_data(obj.as_dict())
    if isinstance(obj, dict):
        return {k: to_plain_data(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain_data(v) for v in obj]
    return obj


def config_to_plain_dict(obj):
    if isinstance(obj, dict):
        return {k: config_to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [config_to_plain_dict(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return {
            k: config_to_plain_dict(v)
            for k, v in vars(obj).items()
            if not k.startswith("_")
        }
    return obj


def normalize_raw_config_dict(raw):
    raw = to_plain_data(raw)
    if not isinstance(raw, dict):
        return raw
    raw = deepcopy(raw)
    if "wandb" in raw and not isinstance(raw["wandb"], dict):
        raw["wandb"] = {"enabled": bool(raw["wandb"])}
    return raw


def deep_merge_dicts(base, override):
    if not isinstance(base, dict) or not isinstance(override, dict):
        return to_plain_data(override)

    merged = deepcopy(base)
    for key, value in override.items():
        value = to_plain_data(value)
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _set_nested_attr(obj, path, value):
    cur = obj
    for key in path[:-1]:
        if not hasattr(cur, key):
            raise AttributeError(f"Unknown config override path: {'.'.join(path)}")
        cur = getattr(cur, key)
    leaf = path[-1]
    if not hasattr(cur, leaf):
        raise AttributeError(f"Unknown config override path: {'.'.join(path)}")
    setattr(cur, leaf, value)


def is_known_config_key(config_obj, key):
    if hasattr(config_obj, key):
        return True
    if "." in key:
        return hasattr(config_obj, key.split(".", 1)[0])
    return False


def apply_config_dict(config_obj, values, prefix="config"):
    for key, value in values.items():
        if "." in key and not hasattr(config_obj, key):
            _set_nested_attr(config_obj, key.split("."), to_plain_data(value))
            continue

        if not hasattr(config_obj, key):
            raise AttributeError(f"Unknown config override: {prefix}.{key}")

        current = getattr(config_obj, key)
        if isinstance(value, bool) and hasattr(current, "enabled"):
            setattr(current, "enabled", value)
            continue
        if isinstance(value, dict) and hasattr(current, "__dict__"):
            apply_config_dict(current, value, prefix=f"{prefix}.{key}")
        else:
            setattr(config_obj, key, to_plain_data(value))


def build_run_name(config):
    model_short = config.model.name.split("/")[-1]
    temp = config.refine.temperature
    scale = config.refine.scale
    quant_type = config.compression.quant_type if config.compression.quant_type else "sparseonly"
    sparsity = f"{config.compression.prunen}-{config.compression.prunem}"
    init_tag = config.init.method
    refine_tag = "refine" if config.refine.enabled else "norefine"
    weight_tag = "learnw" if config.compression.learn_weight_values else "fixedw"
    # Emitted only when the support is FROZEN. A learned support is what every recipe
    # before the support/value ablation did, so tagging it would rename every existing
    # run -- including one being resumed with wandb resume="must" -- for no information:
    # the frozen-support arm is the only one that needs distinguishing, and the frozen-
    # VALUE arm is already covered by fixedw.
    mask_tag = "" if config.refine.learn_masks else "fixedm_"
    actq_tag = "actq" if config.compression.fake_quantize_activations else "noactq"
    return (
        f"{model_short}_"
        f"{init_tag}+{refine_tag}_"
        f"qt={quant_type}_"
        f"sp={sparsity}_"
        f"{mask_tag}"
        f"{weight_tag}_"
        f"{actq_tag}_"
        f"gwp={config.refine.gate_weight_exponent}_"
        f"gsz={config.compression.groupsize}_"
        f"lg_dtype={config.refine.logits_dtype}_"
        f"ep={config.refine.num_epochs}_"
        f"tmp={temp[0]}-{temp[1]}_scl={scale[0]}-{scale[1]}_"
        f"str={config.refine.strength}_"
        f"ncal={config.data.num_samples}_"
        f"bsz={config.data.batch_size}_"
        f"lr={config.refine.masks_lr}_"
        f"wlr={config.refine.weights_lr}"
    )


def print_effective_config(config_dict, source_label):
    import yaml
    print(f"Config source: {source_label}", flush=True)
    print("Effective config:", flush=True)
    print(
        yaml.safe_dump(
            config_to_plain_dict(config_dict),
            sort_keys=False,
            default_flow_style=False,
        ).rstrip(),
        flush=True,
    )
