import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from abc import ABC, abstractmethod
import json
from pathlib import Path
from types import SimpleNamespace
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from transformers import AutoConfig, AutoModelForCausalLM
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file
from src.compression.ct_compat import (
    FP8_E4M3_DATA,
    BitmaskCompressor,
    BitmaskConfig,
    NVFP4PackedCompressor,
    PackedQuantizationCompressor,
    QuantizationStrategy,
    QuantizationType,
    Sparse24BitMaskCompressor,
    Sparse24BitMaskConfig,
    pack_bitmasks,
)
from src.compression.quant.gsq import dequantize_gsq_packed
from contextlib import ExitStack
from src.compression.initialization.gptq import *
from src.compression.initialization import make_initializer
from src.compression.quant.nvfp4 import dense_scale_groupsize
from src.evaluation.wiki_eval import *
from src.utils.progress_reporter import report_gptq_calib, report_gptq_linear

def token_clip_weights(out_fp, k):
    """Sink-aware per-row loss weights for the MoE reconstruction loss (refine.token_weight_clip_k).

    out_fp: dense expert outputs, one row per (token, expert) pair, [rows, H].
    Returns [rows, 1] float weights w = min(1, (k * median ||y||)^2 / ||y||^2), or None when
    k <= 0 (caller then keeps the exact old loss). The median is over the rows on this rank,
    which is enough: the rows this exists for (attention-sink tokens) are 10^2-10^3x the
    median, so any sane estimate of it puts them far past the clip and leaves every
    ordinary row at exactly 1. Detached: the weights are data, not something to optimise.
    """
    if not k or k <= 0 or out_fp is None or out_fp.shape[0] == 0:
        return None
    with torch.no_grad():
        ref2 = out_fp.float().pow(2).sum(dim=-1)
        med2 = ref2.median()
        w = (float(k) * float(k) * med2 / ref2.clamp_min(1e-12)).clamp(max=1.0)
    return w.unsqueeze(1)


class BaseModelWrapper(ABC):
    def __init__(self, model_name, tokenizer, batch_size, seqlen, device, dtype, dummy=False):
        self.ckpt_path = self.resolve_model_path(model_name)
        self.dtype = self.normalize_dtype(dtype)
        self.device = device
        self.save_dir = None

        cfg = AutoConfig.from_pretrained(self.ckpt_path, trust_remote_code=True)
        # HF only sets _attn_implementation on the top-level config; propagate to
        # sub-configs so decoder layers pick up DeepseekV3FlashAttention2.
        if hasattr(cfg, 'text_config') and cfg.text_config is not None:
            cfg.text_config._attn_implementation = "flash_attention_2"
        with init_empty_weights():
            empty_model = self._build_empty_model(cfg)

        self.model = empty_model

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.model.seqlen = seqlen
        self.model.config.use_cache = False

        self.current_layer_idx = 0
        self.loss_fn = torch.nn.MSELoss()

        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        self.dummy = dummy
        self.kwargs = {}
        self._attention_mask_1 = None  # single-item causal mask, expanded on demand
        self.meta_init_std = 0.02
        self.calib_report_divisor = 10
        self.batch_report_divisor = 5

        if dummy:
            self._name_to_shard = None
            self._single_shard = None
        else:
            self._name_to_shard, self._single_shard = self._load_safetensor_index(self.ckpt_path)
        self.compressor = PackedQuantizationCompressor() 

        self.quantization_config = self.dict_to_ns({
            "config_groups": {
              "group_0": {
                "input_activations": None,
                "output_activations": None,
                "targets": [
                  "Linear"
                ],
                "weights": {
                  "actorder": None,
                  "block_structure": None,
                  "dynamic": False,
                  "group_size": 32,
                  "num_bits": None,
                  "observer": "minmax",
                  "observer_kwargs": {},
                  "strategy": "group",
                  "symmetric": True,
                  "type": "int"
                }
              }
            },
            "format": "pack-quantized",
            "ignore": [
              "lm_head",
              "re:.*self_attn.*"
            ],
            "kv_cache_scheme": None,
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed"
          })
        self.temp_weights = {}
        self.is_moe = False
        self.is_nvfp4 = False
        self.groupsize = self.quantization_config.config_groups.group_0.weights.group_size
        self.prunen = 0
        self.prunem = 0

    def _build_empty_model(self, cfg):
        return AutoModelForCausalLM.from_config(
            cfg, attn_implementation="flash_attention_2", trust_remote_code=True
        ).eval()

    def resolve_model_path(self, model_name):
        p = Path(model_name)

        if p.exists():
            if p.is_file():
                return str(p.parent.resolve())
            return str(p.resolve())

        try:
            from huggingface_hub import snapshot_download

            local_dir = snapshot_download(repo_id=model_name, ignore_patterns=["*.bin", "*.pth"])
            return str(Path(local_dir).resolve())
        except Exception as e:
            raise FileNotFoundError(
                f"Could not resolve '{model_name}' as a local path or download it from the Hugging Face Hub. "
                f"Original error: {e}"
            )
        
    def dict_to_ns(self, d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: self.dict_to_ns(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [self.dict_to_ns(x) for x in d]
        else:
            return d

    def configure_quantization_from_config(self, config=None):
        # When called with a full config, latch the compression+init settings.
        # When called with config=None (e.g. from _compress_and_store_weight),
        # reuse whatever was latched on the prior full-config call.
        if config is not None:
            self.is_nvfp4 = (config.compression.quant_type == "nvfp4")
            self.is_gsq = (config.compression.quant_type == "gsq")
            self.prunen = config.compression.prunen
            self.prunem = config.compression.prunem
            # NVFP4 scales are per *dense* group = compression.groupsize * prunem/prunen,
            # so a 2:N-compressed block lands on an exact NVFP4 (per-16-compressed) block.
            if self.is_nvfp4:
                self.groupsize = dense_scale_groupsize(
                    self.prunen, self.prunem, config.compression.groupsize
                )
            else:
                self.groupsize = config.compression.groupsize
            self._init_wbits = config.init.wbits
            self.gate_weight_exponent = getattr(config.refine, "gate_weight_exponent", 0.0)
            self.token_weight_clip_k = getattr(config.refine, "token_weight_clip_k", 0.0)

        weights = self.quantization_config.config_groups.group_0.weights
        weights.group_size = self.groupsize
        # default 16 bit (no quant) if no full-config call has run yet
        weights.num_bits = getattr(self, "_init_wbits", 16)
        if self.is_nvfp4:
            weights.num_bits = 4
            weights.type = QuantizationType.FLOAT
            weights.symmetric = True
            weights.strategy = QuantizationStrategy.TENSOR_GROUP
            weights.scale_dtype = FP8_E4M3_DATA.dtype
            weights.zp_dtype = FP8_E4M3_DATA.dtype
            weights.observer = "memoryless_minmax"
        return weights

    def _compress_and_store_weight(self, to_save, base, weight, scale, global_scale=None):
        quantization_args = self.configure_quantization_from_config()
        if getattr(self, "is_gsq", False):
            self._store_gsq_weight(to_save, base, weight, scale)
            return
        if self.is_nvfp4 and global_scale is None and not self._scale_is_sparse_only(scale):
            scale, global_scale = self._project_nvfp4_scales(scale)
        if global_scale is not None:
            global_scale = global_scale.to(torch.float32).reshape(1)
            compressed = NVFP4PackedCompressor().compress_weight(
                weight=weight,
                scale=scale,
                global_scale=global_scale,
                quantization_args=quantization_args,
            )
            to_save[base + ".weight_packed"] = compressed["weight_packed"]
            to_save[base + ".weight_scale"] = compressed["weight_scale"]
            to_save[base + ".weight_global_scale"] = global_scale
            return
        compressed = self.compressor.compress_weight(
            weight=weight,
            scale=scale,
            quantization_args=quantization_args,
        )
        to_save[base + ".weight_packed"] = compressed["weight_packed"]
        to_save[base + ".weight_scale"] = scale
        to_save[base + ".weight_shape"] = compressed["weight_shape"]

    def _store_gsq_weight(self, to_save, base, weight, scale):
        """Write a GSQ layer in Humming's uint2 format.

        NOT the NVFP4 path: that packs FP4 with an FP8 group scale, and its scale
        convention here (minimum nonzero |w| per group) clips 93.75% of weights when
        applied at 2 bits -- measured. GSQ is unsigned 2-bit grid indices into
        {-2,-1,0,+1} with one bf16 scale per 128 columns, which is what
        ISTA-DASLab/Kimi-K2.5-2Bit-GSQ declares and what `humming-kernels` reads.

        The module holds s*q after get_hard_weights(), and `scale` arrives alongside
        it, so the codes are recovered exactly rather than re-derived heuristically.
        """
        from src.compression.quant.gsq import (
            GSQ_GRIDS, codes_from_dequantized, pack_uint2_to_int32,
        )
        wbits = getattr(self, "_init_wbits", 2)
        grid = torch.tensor(GSQ_GRIDS[wbits], dtype=torch.float32)
        w = weight.detach().to("cpu", torch.float32)
        s = scale.detach().to("cpu", torch.float32)
        groupsize = w.shape[1] // s.shape[1]
        codes = codes_from_dequantized(w, s, groupsize, grid)
        to_save[base + ".weight"] = pack_uint2_to_int32(codes)
        to_save[base + ".weight_scale"] = s.to(torch.bfloat16)

    @staticmethod
    def _scale_is_sparse_only(scale):
        return scale is None or torch.count_nonzero(scale).item() == 0

    @staticmethod
    def _project_nvfp4_scales(scale):
        scale = scale.to(torch.float32)
        if not torch.isfinite(scale).all() or torch.any(scale < 0):
            raise ValueError("NVFP4 scales must be finite and non-negative.")

        scale_eps = torch.tensor(1e-12, device=scale.device, dtype=torch.float32)
        scale_min = torch.tensor(
            torch.finfo(torch.float8_e4m3fn).tiny / 8.0,
            device=scale.device,
            dtype=torch.float32,
        )
        scale_max = torch.tensor(
            torch.finfo(torch.float8_e4m3fn).max,
            device=scale.device,
            dtype=torch.float32,
        )

        scale = torch.clamp(scale, min=scale_eps)
        max_scale = scale.amax().to(torch.float32)
        if max_scale <= 0:
            global_scale = torch.ones(1, device=scale.device, dtype=torch.float32)
        else:
            global_scale = (scale_max / max_scale).reshape(1).to(torch.float32)

        projected = torch.clamp(scale * global_scale, min=scale_min, max=scale_max)
        projected = projected.to(torch.float8_e4m3fn).to(torch.float32)
        if not torch.isfinite(projected).all() or torch.any(projected <= 0):
            raise ValueError("Projected NVFP4 scales must be finite and positive.")
        return projected, global_scale

    def _validate_configured_sparse_weight(self, base, weight, label):
        if weight.ndim != 2:
            raise ValueError(
                f"{base} has {label} weights, but expected a 2D tensor and got {tuple(weight.shape)}."
            )
        if weight.shape[1] % self.prunem != 0:
            raise ValueError(
                f"{base} has {label} weights, but its shape {tuple(weight.shape)} "
                f"is incompatible with {self.prunen}:{self.prunem} serialization."
            )

        nonzero_mask = weight.ne(0)
        block_mask = nonzero_mask.reshape(weight.shape[0], -1, self.prunem)
        expected_nnz = self.prunem - self.prunen
        block_nnz = block_mask.sum(dim=-1)
        if not torch.all(block_nnz == expected_nnz):
            raise ValueError(
                f"{base} has {label} weights, but the tensor is not valid "
                f"{expected_nnz}:{self.prunem} sparsity."
            )

        if self.prunen == 4 and self.prunem == 8:
            pair_view = block_mask.reshape(weight.shape[0], -1, 4, 2)
            pair_nnz = pair_view.sum(dim=-1)
            valid_pairs = (pair_nnz == 0) | (pair_nnz == 2)
            kept_pairs = (pair_nnz == 2).sum(dim=-1)
            if not torch.all(valid_pairs) or not torch.all(kept_pairs == 2):
                raise ValueError(
                    f"{base} has {label} weights, but the tensor is not valid paired 4:8 sparsity."
                )

    def _validate_sparse_only_weight(self, base, weight):
        if self.prunen <= 0 or self.prunem <= 0:
            raise ValueError(
                f"{base} has sparse-only weights, but gptq.prunen/prunem are not configured."
            )
        self._validate_configured_sparse_weight(base, weight, "sparse-only")

    def _compress_and_store_sparse_weight(self, to_save, base, weight, mask=None):
        weight = weight.detach()
        mask = mask.detach().bool() if mask is not None else None
        mask_for_pack = mask.cpu() if mask is not None else None

        if mask is not None and tuple(mask.shape) != tuple(weight.shape):
            raise ValueError(
                f"{base} explicit sparse mask shape {tuple(mask.shape)} does not match "
                f"weight shape {tuple(weight.shape)}."
            )

        if mask is not None and self.prunen == 2 and self.prunem == 4:
            rows, cols = weight.shape
            if cols % 4 != 0:
                raise ValueError(f"{base} columns must be divisible by 4 for 2:4 sparse compression.")
            block_nnz = mask.reshape(rows, -1, 4).sum(dim=-1)
            if not torch.all(block_nnz == 2):
                raise ValueError(f"{base} explicit mask is not valid 2:4 sparsity.")
            name_prefix = base[: -len(".weight")] if base.endswith(".weight") else base
            to_save[f"{name_prefix}.shape"] = torch.tensor(list(weight.shape)).reshape(-1, 1)
            to_save[f"{name_prefix}.compressed"] = weight[mask].reshape(rows, cols // 2)
            to_save[f"{name_prefix}.bitmask"] = pack_bitmasks(mask_for_pack)
            return

        if mask is not None:
            row_counts = mask_for_pack.sum(dim=-1)
            row_offsets = torch.cumsum(row_counts, 0) - row_counts
            to_save[f"{base}.shape"] = torch.tensor(list(weight.shape))
            to_save[f"{base}.compressed"] = weight[mask]
            to_save[f"{base}.bitmask"] = pack_bitmasks(mask_for_pack)
            to_save[f"{base}.row_offsets"] = row_offsets
            return

        if self.prunen == 2 and self.prunem == 4:
            compressor = Sparse24BitMaskCompressor(Sparse24BitMaskConfig())
        else:
            sparsity_structure = f"{self.prunem - self.prunen}:{self.prunem}"
            compressor = BitmaskCompressor(BitmaskConfig(sparsity_structure=sparsity_structure))

        to_save.update(compressor.compress_weight(base, weight))
        
    @staticmethod
    def _load_safetensor_index(ckpt_path):
        index_json = os.path.join(ckpt_path, "model.safetensors.index.json")
        if os.path.exists(index_json):
            with open(index_json, "r") as f:
                idx = json.load(f)
            return idx["weight_map"], None

        single = os.path.join(ckpt_path, "model.safetensors")
        if os.path.exists(single):
            return None, single

        raise FileNotFoundError(f"No safetensors found under {ckpt_path}")

    def _materialize_to_device(self, name_shard_pairs):
        params = dict(self.model.named_parameters())
        for n, _ in name_shard_pairs:
            if n not in params:
                continue
            if params[n].device.type == "meta":
                t = torch.randn(params[n].shape, dtype=self.dtype, device=self.device) * self.meta_init_std
                set_module_tensor_to_device(self.model, n, self.device, value=t, dtype=self.dtype)

    def _names_from_ckpt(self, prefixes):
        if isinstance(prefixes, str):
            prefixes = [prefixes]

        def wanted(name):
            for p in prefixes:
                if name == p or name.startswith(p + "."):
                    return True
            return False

        if self.dummy:
            return [(n, None) for n, _ in self.model.named_parameters() if wanted(n)]

        pairs = []
        if self._name_to_shard is None:
            names = safe_load_file(self._single_shard, device="cpu").keys()
            for n in names:
                if wanted(n):
                    pairs.append((n, self._single_shard))
        else:
            for n, shard in self._name_to_shard.items():
                if wanted(n):
                    pairs.append((n, os.path.join(self.ckpt_path, shard)))
        return pairs

    def _set_tensors(self, name_shard_pairs):
        if self.dummy:
            self._materialize_to_device(name_shard_pairs)
            return

        by_shard = {}
        for n, s in name_shard_pairs:
            by_shard.setdefault(s, []).append(n)

        for shard, names in by_shard.items():
            tensors = safe_load_file(shard, device=self.device)
            for n in names:
                if n.endswith("inv_freq"):
                    continue
                t = tensors[n].to(dtype=self.dtype, copy=False)
                set_module_tensor_to_device(self.model, n, self.device, value=t, dtype=self.dtype)

    def move_layer_to_gpu(self, layer_name):
        prefixes = self._layer_prefixes(layer_name)
        pairs = self._names_from_ckpt(prefixes["non_mlp"] + prefixes["mlp"])
        self._set_tensors(pairs)

    def _offload_names_to_meta(self, name_shard_pairs):
        names = [n if isinstance(n, str) else n[0] for n in name_shard_pairs]
        for n in names:
            if n.endswith("inv_freq"):
                continue
            set_module_tensor_to_device(self.model, n, "meta")
        torch.cuda.empty_cache()

    def offload_to_meta(self, layer_name):
        prefixes = self._layer_prefixes(layer_name)
        pairs = self._names_from_ckpt(prefixes["non_mlp"] + prefixes["mlp"])
        self._offload_names_to_meta(pairs)

    @torch.no_grad()
    def get_inputs(self, data_dict, data_loader):
        current_layer = self.get_layer_module(self.current_layer_idx)
        cache = {'index': 0}
        def store_input_hook(_, args, kwargs):
            start = cache['index'] * self.batch_size
            end = min(start + self.batch_size, data_dict['input'].shape[0])
            if isinstance(args, tuple):
                args = args[0]
            data_dict['input'][start:end] = args
            cache['index'] += 1
            for k, v in kwargs.items():
                if k == "attention_mask":
                    if v is not None:
                        self._attention_mask_1 = v[:1].detach().cpu()
                elif k not in ("hidden_states", "past_key_values", "past_key_value"):
                    self.kwargs[k] = v
            raise ValueError
            
        total_batches = len(data_loader)
        handle = current_layer.register_forward_pre_hook(store_input_hook, with_kwargs=True)
        for batch_idx, batch in enumerate(data_loader):
            try:
                self.model(batch.to(self.device))
            except ValueError:
                pass
            if self.rank == 0 and (batch_idx + 1) % max(1, total_batches // self.batch_report_divisor) == 0:
                print(f"  get_inputs: {batch_idx + 1}/{total_batches} batches", flush=True)
        handle.remove()

    def _build_layer_inputs(self, batch_size):
        """Build kwargs dict for a layer forward, expanding attention_mask to batch_size."""
        inputs = dict(self.kwargs)
        if self._attention_mask_1 is not None:
            inputs["attention_mask"] = self._attention_mask_1.to(self.device).expand(batch_size, -1, -1, -1)
        else:
            inputs["attention_mask"] = None
        return inputs

    @torch.no_grad()
    def get_layer_activations(self, data_all):
        current_layer = self.get_layer_module(self.current_layer_idx)
        for batch_idx in range((data_all['input'].size(0) + self.batch_size - 1) // self.batch_size):
            start_idx, end_idx = batch_idx * self.batch_size, min((batch_idx + 1) * self.batch_size, data_all['input'].shape[0])
            x = data_all['input'][start_idx:end_idx].to(self.device, non_blocking=True)

            additional_layer_inputs = self._build_layer_inputs(x.shape[0])
            out = current_layer(x, **additional_layer_inputs)
            if isinstance(out, tuple):
                out = out[0]
            data_all['input'][start_idx:end_idx] = out.detach().cpu()

    @torch.no_grad()
    def get_mlp_input_all(self, data_all):
        num_samples = data_all['input'].shape[0]
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        for batch_idx in range(num_batches):
            start_idx, end_idx = batch_idx * self.batch_size, min((batch_idx + 1) * self.batch_size, num_samples)
            data_all['input'][start_idx:end_idx] = self.get_mlp_input(data_all['input'][start_idx:end_idx].to(self.device)).detach().cpu()
    
    @torch.no_grad()
    def get_mlp_output_all(self, data_all):
        num_samples = data_all['input'].shape[0]
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        for batch_idx in range(num_batches):
            start_idx, end_idx = batch_idx * self.batch_size, min((batch_idx + 1) * self.batch_size, num_samples)
            data_all['input'][start_idx:end_idx] = self.get_mlp_output(data_all['input'][start_idx:end_idx].to(self.device)).detach().cpu()
        
    @torch.no_grad()
    def get_loss(self, layer_inputs, layer_outputs):
        new_outputs = self.get_mlp_output(layer_inputs)            
        val_loss = self.loss_fn(layer_outputs, new_outputs)
        return layer_inputs.detach().cpu(), new_outputs.detach().cpu(), val_loss.item()

    def get_layer_initialization(self, trainer, gpt_all, config, logging, is_attn=False):
        if logging is not None:
            logging = logging.logger
        current_layer = self.get_layer_module(self.current_layer_idx)
        self.configure_quantization_from_config(config)
        refine_enabled = config.refine.enabled
        subset = {}
        
        if is_attn:
            self_attn = current_layer.self_attn
            base_prefix = f"{self.get_current_layer()}.self_attn"
            for name, module in self_attn.named_modules():
                if isinstance(module, torch.nn.Linear):
                    subset[f"{base_prefix}.{name}"] = module
        else:
            mlp = current_layer.mlp
            base_prefix = f"{self.get_current_layer()}.mlp"
            for name, module in mlp.named_modules():
                if isinstance(module, torch.nn.Linear):
                    subset[f"{base_prefix}.{name}"] = module

        init_method = config.init.method

        if init_method in ("rtn", "random"):
            quantize_fn = random_quantize if init_method == "random" else rtn_quantize
            for name in subset:
                Q, scales = quantize_fn(subset[name], config, self.device, self.dtype)
                if "q_proj" in name or "k_proj" in name or not refine_enabled:
                    self.update_compressed_weights(name, (Q, scales))
                    if self.world_size > 1:
                        layer = self._get_layer_by_name(name)
                        dist.broadcast(layer.weight.data, src=0)
                else:
                    trainer.setup_layer_training(name, Q, scales)
            return

        if init_method not in ("gptq", "obr"):
            raise ValueError(
                f"Unknown init_method={init_method!r}. Supported: 'gptq', 'obr', 'rtn', 'random'"
            )

        gpts = {}
        for name in subset:
            gpts[name] = make_initializer(config, subset[name], name, self.device, self.dtype)
            if config.compression.quant_type == "nvfp4" or config.init.wbits < 16:
                gpts[name].quantizer = make_quantizer(config)

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        
        calib_total = gpt_all['input'].shape[0]
        calib_report_interval = max(1, calib_total // self.calib_report_divisor)
        calib_start = time.time()
        for j in range(calib_total):
            inp = gpt_all['input'][j].unsqueeze(0).to(self.device)
            additional_layer_inputs = self._build_layer_inputs(1)
            _ = current_layer(inp, **additional_layer_inputs)
            if self.rank == 0 and (j + 1) % calib_report_interval == 0:
                report_gptq_calib(j + 1, calib_total, time.time() - calib_start)
        for h in handles:
            h.remove()

        gptq_losses = []
        linear_names = list(gpts.keys())
        linear_start = time.time()
        for li, name in enumerate(linear_names):
            if self.rank == 0:
                report_gptq_linear(li + 1, len(linear_names), name,
                                   time.time() - linear_start)
            if self.world_size > 1:
                gpts[name].sync_H(self.world_size)
            Q, scales = gpts[name].fasterquant(
                logging, percdamp=config.init.percdamp, blocksize=config.init.blocksize, groupsize=config.compression.groupsize, static_groups=config.init.static_groups, prunen=config.compression.prunen, prunem=config.compression.prunem
            ) ##TODO:check scales for 2:4 and 2:4+nvfp4, group_scales would be torch.zeros.
            if hasattr(gpts[name], 'last_gptq_loss'):
                gptq_losses.append(gpts[name].last_gptq_loss)
            if "q_proj" in name or "k_proj" in name or not refine_enabled:
                self.update_compressed_weights(name, (Q, scales))
                if self.world_size > 1:
                    layer = self._get_layer_by_name(name)
                    dist.broadcast(layer.weight.data, src=0)
            else:
                trainer.setup_layer_training(
                    name,
                    Q,
                    scales,
                    init_dense_weight=gpts[name].last_init_dense_weight,
                    init_support_mask=gpts[name].last_init_support_mask,
                )
            gpts[name].free()

        if gptq_losses:
            trainer.gptq_avg_loss = sum(gptq_losses) / len(gptq_losses)

    def _expert_recon_loss(self, out_q, out_fp, gate_w=None):
        """MoE reconstruction loss over (token, expert) rows of expert outputs.

        With refine.gate_weight_exponent p > 0 each row is weighted by its router weight
        g^p (``gate_w`` [rows, 1]): p=1 is a Jensen upper bound on the block-output error
        ||sum_e g_e d_e||^2, p=2 its exact diagonal. The weighted mean keeps the loss scale
        comparable across p, and refine.token_weight_clip_k caps sink-token rows. With
        neither set this is the plain MSE.

        The squared error is summed per row in fp32 straight from the model-dtype
        difference: an fp32 copy of the whole [rows, hidden] difference (and its
        gradient) is what the unweighted MSE never needed, and at Kimi-K2.5 scale it
        runs a 180 GB GPU out of memory on layers with a hot expert.
        """
        gate_p = getattr(self, "gate_weight_exponent", 0.0)
        clip = token_clip_weights(out_fp, getattr(self, "token_weight_clip_k", 0.0))
        if gate_p <= 0 and clip is None:
            return self.loss_fn(out_q, out_fp)
        if gate_p > 0:
            wp = gate_w.float().pow(gate_p)
        else:
            wp = torch.ones(out_fp.shape[0], 1, device=out_fp.device, dtype=torch.float)
        if clip is not None:
            wp = wp * clip
        row_se = (out_q - out_fp).pow(2).sum(dim=-1, keepdim=True, dtype=torch.float32)
        return (wp * row_se).sum() / (wp.sum().clamp_min(1e-8) * out_fp.shape[-1])

    def calculate_mse(self, batch, compressed_weights, self_attn, validation=False, accumulation_steps=1, fake_act_quant=False):
        with torch.no_grad():
            out_fp = self.forward_with_quantized(batch, None, self_attn)
        out_q = self.forward_with_quantized(batch, compressed_weights, self_attn)

        mse = self.loss_fn(out_q, out_fp)
        if not validation:
            (mse / accumulation_steps).backward()

        return mse.item()

    def forward_with_quantized(self, batch, compressed_weights, self_attn):
        class LinearWeightHook:
            def __init__(self, module, compressed_weight):
                self.module = module
                self.compressed_weight = compressed_weight
                self.original_forward = module.forward

            def __enter__(self):
                def new_forward(module_self, x):
                    return torch.nn.functional.linear(x, self.compressed_weight, self.module.bias)
                self.module.forward = new_forward.__get__(self.module, torch.nn.Linear)

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.module.forward = self.original_forward

        current_layer = self.get_layer_module(self.current_layer_idx)
        hooks = []
        try:
            if compressed_weights is not None:
                if self_attn:
                    layer = current_layer.self_attn
                else:
                    layer = current_layer.mlp

                for name, module in layer.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        if self_attn:
                            key = f"{self.get_current_layer()}.self_attn.{name}"
                        else: 
                            key = f"{self.get_current_layer()}.mlp.{name}"
                        if key in compressed_weights:
                            hooks.append(LinearWeightHook(module, compressed_weights[key]))
            else:
                for name, module in current_layer.self_attn.named_modules():
                    key = f"{self.get_current_layer()}.self_attn.{name}"
                    if key in self.temp_weights:
                        hooks.append(LinearWeightHook(module, self.temp_weights[key]))

            with ExitStack() as stack:
                for h in hooks: stack.enter_context(h)
                if self_attn:
                    output = self.get_mlp_input(batch)
                else:
                    additional_layer_inputs = self._build_layer_inputs(batch.shape[0])
                    output = current_layer(batch, **additional_layer_inputs)
            return output
        finally:
            hooks.clear()

    def find_layers(self, module, layers=[nn.Linear], name=''):
        if type(module) in layers:
            return {name: module}
        res = {}
        for name1, child in module.named_children():
            res.update(self.find_layers(
                child, layers=layers, name=name + '.' + name1 if name != '' else name1
            ))
        return res
        
    def _get_layer_by_name(self, layer_name):
        return self.model.get_submodule(layer_name)

    @staticmethod
    def _split_weight_payload(weight_data):
        if not isinstance(weight_data, tuple):
            return weight_data, None, None, None
        if len(weight_data) == 2:
            weight, aux = weight_data
            if isinstance(aux, torch.Tensor) and aux.dtype == torch.bool:
                return weight, None, None, aux
            return weight, aux, None, None
        if len(weight_data) == 3:
            weight, scale, third = weight_data
            if isinstance(third, torch.Tensor) and third.dtype == torch.bool:
                return weight, scale, None, third
            return weight, scale, third, None
        if len(weight_data) == 4:
            return weight_data
        raise ValueError(f"Unexpected quantized weight tuple length: {len(weight_data)}")

    @staticmethod
    def _split_weight_aux(weight_data):
        weight, scale, _, mask = BaseModelWrapper._split_weight_payload(weight_data)
        return weight, scale, mask
          
    def update_compressed_weights(self, layer_name, compressed_weights):
        weight, scale, global_scale, mask = self._split_weight_payload(compressed_weights)
        layer = self._get_layer_by_name(layer_name)
        self.temp_weights[layer_name] = layer.weight.data
        self.temp_weights[f"{layer_name}.scale"] = scale ## this will be zeros/None for sparse-only gptq
        if global_scale is not None:
            self.temp_weights[f"{layer_name}.global_scale"] = global_scale
        if mask is not None:
            self.temp_weights[f"{layer_name}.mask"] = mask
        with torch.no_grad():
            layer.weight.data = weight.to(self.device).to(layer.weight.data.dtype)
        
    def get_current_layer(self):
        return f"{self.layer_prefix}.{self.current_layer_idx}"
        
    def move_to_next_layer(self):
        self.current_layer_idx += 1
        if self.current_layer_idx < self.num_layers:
            return self.get_current_layer()
        return None
    
    def normalize_dtype(self, x):
        if isinstance(x, torch.dtype):
            return x
        if isinstance(x, str):
            try:
                return getattr(torch, x)
            except AttributeError:
                raise ValueError(f"Unknown dtype string: {x}")
        raise TypeError(f"Expected torch.dtype or str, got {type(x)}")
    
    def save_to_disc(self, pfx, pairs):
        os.makedirs(self.save_dir, exist_ok=True)
        to_save = {}
        for name, pair in pairs.items():
            base = f'{pfx}.{name}'
            weight, scale, global_scale, mask = self._split_weight_payload(pair)
            if self._scale_is_sparse_only(scale):
                self._compress_and_store_sparse_weight(to_save, base, weight, mask)
            else:
                self._compress_and_store_weight(to_save, base, weight, scale, global_scale)
        safe = pfx.replace('.', '_')
        path = os.path.join(self.save_dir, f"{safe}.safetensors")
        safe_save_file(to_save, path)

    def save_attention_to_disc(self, pfx):
        os.makedirs(self.save_dir, exist_ok=True)
        to_save = {}
        self_attn_layers = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        for name in self_attn_layers:
            base = f"{pfx}.{name}"
            module = self.model.get_submodule(base)
            self._compress_and_store_weight(
                to_save,
                base,
                module.weight.data,
                self.temp_weights[f"{base}.scale"],
                self.temp_weights.get(f"{base}.global_scale"),
            )
        safe = pfx.replace('.', '_')
        path = os.path.join(self.save_dir, f"{safe}.safetensors")
        safe_save_file(to_save, path)

    def save_mlp_to_disc(self, pfx):
        os.makedirs(self.save_dir, exist_ok=True)
        mlp_module = self.model.get_submodule(pfx)
        linear_names = [
            n for n, m in mlp_module.named_modules()
            if isinstance(m, torch.nn.Linear)
        ]
        to_save = {}
        for name in linear_names:
            base = f"{pfx}.{name}"
            module = self.model.get_submodule(base)
            scale = self.temp_weights[f"{base}.scale"]
            global_scale = self.temp_weights.get(f"{base}.global_scale")
            mask = self.temp_weights.get(f"{base}.mask")
            weight = module.weight.data
            if self._scale_is_sparse_only(scale): ## sparse weights give torch.zeros_like(w) for scales.
                self._compress_and_store_sparse_weight(to_save, base, weight, mask)
            else:
                self._compress_and_store_weight(to_save, base, weight, scale, global_scale)
        safe = pfx.replace('.', '_')
        path = os.path.join(self.save_dir, f"{safe}.safetensors")
        safe_save_file(to_save, path)

    def save_moe_experts_to_disc(self):
        os.makedirs(self.save_dir, exist_ok=True)
        scale_keys = sorted(k for k in self.temp_weights if k.endswith(".scale"))
        experts = {}
        for sk in scale_keys:
            full_name = sk[: -len(".scale")]
            if ".gate_proj" in full_name:
                base = full_name[: -len(".gate_proj")]
                experts.setdefault(base, {})["gate_proj"] = full_name
            elif ".up_proj" in full_name:
                base = full_name[: -len(".up_proj")]
                experts.setdefault(base, {})["up_proj"] = full_name
            elif ".down_proj" in full_name:
                base = full_name[: -len(".down_proj")]
                experts.setdefault(base, {})["down_proj"] = full_name
        for base, proj_names in experts.items():
            pairs = {}
            for proj, full_name in proj_names.items():
                q_key = f"{full_name}.Q"
                if q_key in self.temp_weights:
                    Q = self.temp_weights[q_key]
                else:
                    Q = self.model.get_submodule(full_name).weight.data
                scale = self.temp_weights[f"{full_name}.scale"]
                global_scale = self.temp_weights.get(f"{full_name}.global_scale")
                mask = self.temp_weights.get(f"{full_name}.mask")
                if global_scale is not None:
                    pairs[proj] = (Q, scale, global_scale, mask)
                else:
                    pairs[proj] = (Q, scale, mask) if mask is not None else (Q, scale)
            self.save_to_disc(base, pairs)

    def save_prefixes_to_disc(self, prefixes, exclude=[]):
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        if not prefixes:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        prefixes = [p for p in prefixes if not any(x in p for x in exclude)]
        for pfx in prefixes:
            module = self.model.get_submodule(pfx)
            sd = module.state_dict(keep_vars=True)
            to_save = {}
            for local_name, tensor in sd.items():
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                    to_save[f"{pfx}.{local_name}"] = tensor.detach().cpu()
            safe = pfx.replace('.', '_')
            path = os.path.join(self.save_dir, f"{safe}.safetensors")
            safe_save_file(to_save, path)

    def load_from_disc(self, layer_name):
        quantization_args = self.configure_quantization_from_config()
        prefixes = self._layer_prefixes(layer_name)
        if isinstance(prefixes, str):
            prefixes = [prefixes]

        files = {}
        for item in prefixes:
            for p in prefixes[item]:
                base = p.replace('.', '_')
                st = os.path.join(self.save_dir, f"{base}.safetensors")
                files[p] = st

        for p, path in files.items():
            tensors = safe_load_file(path, device="cpu")
            tensor_names = set(tensors.keys())
            for name in tensors.keys():
                if (
                    name.endswith(".weight_shape")
                    or name.endswith(".weight_scale")
                    or name.endswith(".weight_global_scale")
                    or name.endswith(".bitmask")
                    or name.endswith(".row_offsets")
                    or name.endswith(".shape")
                    or name.endswith("inv_freq")
                ):
                    continue
                if name.endswith(".weight_packed"):
                    base = name[: -len(".weight_packed")]
                    if f"{base}.weight_global_scale" in tensor_names:
                        compressed_data = {
                            "weight_packed": tensors[f"{base}.weight_packed"],
                            "weight_scale": tensors[f"{base}.weight_scale"],
                            "weight_global_scale": tensors[f"{base}.weight_global_scale"],
                        }
                        W_deq = NVFP4PackedCompressor().decompress_weight(
                            compressed_data, quantization_args
                        )
                    else:
                        compressed_data = {
                            "weight_packed": tensors[f"{base}.weight_packed"], 
                            "weight_scale": tensors[f"{base}.weight_scale"],
                            "weight_shape": tensors[f"{base}.weight_shape"]
                        }
                        W_deq = self.compressor.decompress_weight(compressed_data, quantization_args)

                    set_module_tensor_to_device(self.model, f"{base}.weight", self.device, value=W_deq, dtype=self.dtype)
                    continue
                if name.endswith(".compressed"):
                    base = name[: -len(".compressed")]
                    if f"{base}.row_offsets" in tensor_names:
                        compressed_data = {
                            "compressed": tensors[f"{base}.compressed"],
                            "bitmask": tensors[f"{base}.bitmask"],
                            "row_offsets": tensors[f"{base}.row_offsets"],
                            "shape": tensors[f"{base}.shape"],
                        }
                        W_deq = BitmaskCompressor(BitmaskConfig()).decompress_weight(compressed_data)
                    else:
                        compressed_data = {
                            "compressed": tensors[f"{base}.compressed"], 
                            "bitmask": tensors[f"{base}.bitmask"],
                            "shape": tensors[f"{base}.shape"]
                        }
                        W_deq = Sparse24BitMaskCompressor(Sparse24BitMaskConfig()).decompress_weight(compressed_data)
                    set_module_tensor_to_device(self.model, f"{base}.weight", self.device, value=W_deq, dtype=self.dtype)
                    continue    
                if (
                    getattr(self, "is_gsq", False)
                    and name.endswith(".weight")
                    and tensors[name].dtype in (torch.int32, torch.int64)
                    and f"{name}_scale" in tensor_names
                ):
                    # GSQ reuses the plain `.weight` key for Humming's packed uint2
                    # codes, so it reaches neither the .weight_packed nor the
                    # .compressed branch above; the generic setter would try to put a
                    # [out, in/16] int32 tensor into the [out, in] dense parameter.
                    # Dequantize it here instead.
                    W_deq = dequantize_gsq_packed(
                        tensors[name], tensors[f"{name}_scale"], getattr(self, "_init_wbits", 2)
                    )
                    set_module_tensor_to_device(self.model, name, self.device, value=W_deq, dtype=self.dtype)
                    continue
                set_module_tensor_to_device(self.model, name, self.device, value=tensors[name],
                                            dtype=self._reload_dtype(name))

    def _reload_dtype(self, name):
        """dtype a plain (uncompressed) tensor is restored in by ``load_from_disc``.

        None keeps the parameter's own dtype, for models that hold some parameters in
        fp32 (e.g. linear-attention decay terms) alongside a bf16 model dtype.
        """
        return self.dtype

    @abstractmethod
    def get_mlp_input(self, layer_input):
        pass

    @abstractmethod
    def get_mlp_output(self, mlp_input_batch):
        pass
    
    @abstractmethod
    def get_layer_module(self, idx):
        pass

    @abstractmethod
    def move_embed_to(self, device):
        pass

    @abstractmethod
    def move_output_heads_to(self, device):
        pass

    @abstractmethod
    def _layer_prefixes(self, layer_name):
        pass

    @abstractmethod
    def ppl_evaluation(self, read_from_disk=-1):
        pass
