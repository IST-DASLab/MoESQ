import gc
import importlib.util
import math
import os
import re
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate.utils import set_module_tensor_to_device
from src.compression.ct_compat import (
    BitmaskCompressor,
    BitmaskConfig,
    NVFP4PackedCompressor,
    Sparse24BitMaskCompressor,
    Sparse24BitMaskConfig,
)
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file

from .base import BaseModelWrapper
from src.compression.activation_quant import fake_quantize_activation_nvfp4
from src.compression.initialization import make_initializer
from src.compression.initialization.gptq import make_quantizer, random_quantize, rtn_quantize
from src.compression.quant.gsq import dequantize_gsq_packed
from src.evaluation.wiki_eval import get_dataset, prepare_test_dataloader
from src.moe.autograd_ops import AllToAllTokens
from src.moe.placement import ExpertSharder
from src.utils.progress_reporter import report_gptq_calib, report_ppl_layer


def native_kimi_k25_available():
    """True when the installed transformers ships the native ``Kimi_K25`` classes (>= 5.14)."""
    try:
        from transformers import Kimi_K25Config, Kimi_K25ForConditionalGeneration  # noqa: F401
    except ImportError:
        return False
    return True


class KimiK25FusedWrapper(BaseModelWrapper):
    """Expert-parallel Kimi-K2.5 on the native transformers ``Kimi_K25`` classes.

    The text model is transformers' DeepseekV3: routed experts live in one
    ``DeepseekV3Experts`` per layer as two fused 3D parameters,
    ``gate_up_proj [E, 2*I, H]`` (gate in the first half of dim 1) and
    ``down_proj [E, H, I]``, and the router is ``DeepseekV3TopkRouter``.

    Two name spaces are in play:

    * **checkpoint names** -- ``language_model.model.layers.N...`` and
      ``language_model.lm_head`` -- are what the on-disk checkpoint, this wrapper's
      ``layer_prefix``, the trainer's tensor keys, the per-layer shards and the
      exported model all use. They are identical to ``KimiK25Wrapper``'s, so shards
      and exports are interchangeable between the two wrappers.
    * **model names** -- ``model.language_model.layers.N...`` and ``lm_head`` -- are
      the native module paths. They are only ever produced by
      ``_ckpt_to_model_name`` right before a module or tensor is looked up.

    Expert storage is rank-sharded: a layer's fused params hold only this rank's
    experts, in ascending global-eid order, so every row index must go through
    ``_local_expert_index``. transformers' own ``DeepseekV3Experts.forward`` indexes
    by global eid and must never be called on them; all expert compute goes through
    ``_batched_expert_forward``.
    """

    _CKPT_TO_MODEL_PREFIXES = (
        ("language_model.model.", "model.language_model."),
        ("language_model.lm_head.", "lm_head."),
    )

    # Per-expert checkpoint weight (after `.weight_packed` -> `.weight`), written into
    # its slice of the fused params by _write_expert_weight.
    _PER_EXPERT_WEIGHT_RE = re.compile(
        r"^(?P<experts>.+\.mlp\.experts)\.(?P<eid>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
    )
    # Checkpoints saved from the native classes already hold the fused 3D tensors.
    _FUSED_EXPERT_WEIGHT_RE = re.compile(
        r"^(?P<experts>.+\.mlp\.experts)\.(?P<pname>gate_up_proj|down_proj)$"
    )
    # Trainer / GPTQ tensor key of one per-expert linear.
    _EXPERT_KEY_RE = re.compile(r"\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$")

    def __init__(self, model_name, tokenizer, batch_size, seqlen, device, dtype, dummy=False):
        super().__init__(model_name, tokenizer, batch_size, seqlen, device, dtype, dummy=dummy)
        text_cfg = self.model.config.text_config
        self.text_config = text_cfg
        # Expert-pruned bf16 checkpoints carry no quantization_config; the base default is
        # then inert because only the `.weight_packed` branch of _set_tensors reads it.
        qc = getattr(text_cfg, "quantization_config", None)
        if qc is not None:
            if not isinstance(qc, dict):
                qc = qc.to_dict() if hasattr(qc, "to_dict") else vars(qc)
            self.quantization_config = self.dict_to_ns(qc)

        self.layer_prefix = "language_model.model.layers"
        self.num_layers = len(self._text_model().layers)
        self.num_experts = text_cfg.n_routed_experts
        self.first_k_dense_replace = text_cfg.first_k_dense_replace
        self.hidden_size = text_cfg.hidden_size
        self.moe_intermediate_size = text_cfg.moe_intermediate_size
        self.is_moe = True

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.sharder = ExpertSharder(num_experts=self.num_experts, world_size=self.world_size)
        self.groupsize = 32

        self._owner_lut = torch.tensor(
            [self.sharder.owner(e) for e in range(self.num_experts)],
            dtype=torch.long
        )
        self._local_eids = sorted(self.sharder.local_experts(self.rank))
        self._eid_to_local = {int(e): i for i, e in enumerate(self._local_eids)}
        self.num_local_experts = len(self._local_eids)

    # ------------------------------------------------------------------ model build

    @staticmethod
    def _attn_implementation():
        """flash_attention_2 when flash-attn is installed and CUDA is visible, else sdpa."""
        if importlib.util.find_spec("flash_attn") is not None and torch.cuda.is_available():
            return "flash_attention_2"
        return "sdpa"

    def _build_empty_model(self, cfg):
        """Build the native ``Kimi_K25ForConditionalGeneration`` on meta.

        ``cfg`` (the checkpoint's remote-code config) is not used: ``config.json``
        carries an ``auto_map``, so the Auto* factories would pick the remote classes,
        which do not import on transformers 5.x. The native config is loaded from the
        same ``config.json`` instead.
        """
        from transformers import Kimi_K25Config, Kimi_K25ForConditionalGeneration

        native_cfg = Kimi_K25Config.from_pretrained(self.ckpt_path)
        attn = self._attn_implementation()
        native_cfg._attn_implementation = attn
        native_cfg.text_config._attn_implementation = attn
        native_cfg.text_config.use_cache = False
        return Kimi_K25ForConditionalGeneration._from_config(
            native_cfg, attn_implementation=attn, dtype=self.dtype
        ).eval()

    def _text_model(self):
        return self.model.model.language_model

    def get_layer_module(self, idx):
        return self._text_model().layers[idx]

    def _is_moe_layer_idx(self, layer_idx):
        return layer_idx >= self.first_k_dense_replace

    # ------------------------------------------------------------------ names

    @classmethod
    def _ckpt_to_model_name(cls, name):
        for ckpt_pfx, model_pfx in cls._CKPT_TO_MODEL_PREFIXES:
            if name.startswith(ckpt_pfx):
                return model_pfx + name[len(ckpt_pfx):]
        return name

    @classmethod
    def _model_to_ckpt_name(cls, name):
        for ckpt_pfx, model_pfx in cls._CKPT_TO_MODEL_PREFIXES:
            if name.startswith(model_pfx):
                return ckpt_pfx + name[len(model_pfx):]
        return name

    def _module(self, ckpt_name):
        return self.model.get_submodule(self._ckpt_to_model_name(ckpt_name))

    def _get_layer_by_name(self, layer_name):
        return self._module(layer_name)

    def _names_from_ckpt(self, prefixes):
        if not self.dummy:
            return super()._names_from_ckpt(prefixes)
        # Dummy mode has no checkpoint: enumerate the model's own tensors, in checkpoint
        # names so every caller keeps working in one name space.
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        pairs = []
        for n, _ in list(self.model.named_parameters()) + list(self.model.named_buffers()):
            c = self._model_to_ckpt_name(n)
            if any(c == p or c.startswith(p + ".") for p in prefixes):
                pairs.append((c, None))
        return pairs

    def _layer_prefixes(self, layer_name):
        layer_idx = int(layer_name.split('.')[-1])
        base = f"{self.layer_prefix}.{layer_idx}"
        if not self._is_moe_layer_idx(layer_idx):
            non_mlp = [
                f"{base}.input_layernorm",
                f"{base}.self_attn",
                f"{base}.post_attention_layernorm"
            ]
            mlp = [f"{base}.mlp"]
        else:
            non_mlp = [
                f"{base}.input_layernorm",
                f"{base}.self_attn",
                f"{base}.mlp.gate",
                f"{base}.mlp.shared_experts",
                f"{base}.post_attention_layernorm"
            ]
            mlp = [f"{base}.mlp.experts.{e}" for e in self._local_eids]
            # A fused checkpoint stores every expert in these two tensors; they match
            # nothing in a per-expert checkpoint, so listing them is safe for both.
            mlp += [f"{base}.mlp.experts.gate_up_proj", f"{base}.mlp.experts.down_proj"]
        return {"non_mlp": non_mlp, "mlp": mlp}

    # ------------------------------------------------------------------ fused expert storage

    def _local_expert_index(self, eid):
        """Row of global expert ``eid`` in this rank's sharded fused params.

        Raises for an eid this rank does not own: a stray global index below
        ``num_local_experts`` would otherwise read another expert's row silently.
        """
        idx = self._eid_to_local.get(int(eid), -1)
        if idx < 0:
            raise KeyError(
                f"expert {int(eid)} is not owned by rank {self.rank} "
                f"(owns {self.num_local_experts} of {self.num_experts})"
            )
        return idx

    def _fused_expert_shapes(self):
        n, inter, hid = self.num_local_experts, self.moe_intermediate_size, self.hidden_size
        return {"gate_up_proj": (n, 2 * inter, hid), "down_proj": (n, hid, inter)}

    def _ensure_fused_expert_params(self, experts_prefix):
        """Materialise a layer's sharded fused expert params on device, once.

        Rows are filled by the loaders; in dummy mode they are random instead.
        """
        mod = self._module(experts_prefix)
        for pname, shape in self._fused_expert_shapes().items():
            p = getattr(mod, pname)
            if p.device.type != "meta":
                continue
            if self.dummy:
                t = torch.randn(shape, dtype=self.dtype, device=self.device) * self.meta_init_std
            else:
                t = torch.zeros(shape, dtype=self.dtype, device=self.device)
            setattr(mod, pname, torch.nn.Parameter(t, requires_grad=False))

    def _write_expert_weight(self, experts_prefix, eid, proj, weight):
        """Write one per-expert 2D weight into its slice of the fused params.

        Layout matches transformers' converter for this architecture
        (``MergeModulelist(dim=0), Concatenate(dim=1)`` over gate, up): experts on
        dim 0 and gate in the FIRST half of ``gate_up_proj``'s dim 1. Swapping the
        halves swaps SiLU's gate and value paths without raising.
        """
        self._ensure_fused_expert_params(experts_prefix)
        mod = self._module(experts_prefix)
        row = self._local_expert_index(eid)
        inter = self.moe_intermediate_size
        w = weight.to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            if proj == "gate_proj":
                mod.gate_up_proj[row, :inter].copy_(w)
            elif proj == "up_proj":
                mod.gate_up_proj[row, inter:].copy_(w)
            else:
                mod.down_proj[row].copy_(w)

    def _write_fused_expert_rows(self, match, weight):
        """Gather this rank's rows out of an already-fused ``[num_experts, ...]`` tensor.

        ``_local_eids`` is ascending and ``_eid_to_local`` is its inverse, so
        ``index_select(0, _local_eids)`` puts each owned eid on the row
        ``_local_expert_index`` reads.
        """
        self._ensure_fused_expert_params(match.group("experts"))
        param = getattr(self._module(match.group("experts")), match.group("pname"))
        if weight.shape[0] != self.num_experts:
            raise ValueError(
                f"{match.group(0)}: fused expert tensor has {weight.shape[0]} rows but the "
                f"config declares n_routed_experts={self.num_experts}"
            )
        if tuple(weight.shape[1:]) != tuple(param.shape[1:]):
            raise ValueError(
                f"{match.group(0)}: checkpoint trailing dims {tuple(weight.shape[1:])} do not "
                f"match the model's {tuple(param.shape[1:])}"
            )
        idx = torch.as_tensor(self._local_eids, dtype=torch.long, device=weight.device)
        with torch.no_grad():
            param.copy_(weight.index_select(0, idx).to(device=self.device, dtype=self.dtype))

    def _write_weight(self, ckpt_name, value):
        """Place a dense tensor, given by checkpoint name, into the model.

        Floating tensors are cast to ``self.dtype``. That includes the router's fp32
        ``e_score_correction_bias``, which ``KimiK25Wrapper`` also holds in the model
        dtype; keeping the same rounding keeps expert selection identical between
        the two wrappers.
        """
        m = self._PER_EXPERT_WEIGHT_RE.match(ckpt_name)
        if m is not None:
            self._write_expert_weight(m.group("experts"), m.group("eid"), m.group("proj"), value)
            return
        mf = self._FUSED_EXPERT_WEIGHT_RE.match(ckpt_name)
        if mf is not None:
            self._write_fused_expert_rows(mf, value)
            return
        set_module_tensor_to_device(
            self.model, self._ckpt_to_model_name(ckpt_name), self.device,
            value=value, dtype=self.dtype,
        )

    # ------------------------------------------------------------------ checkpoint I/O

    def _materialize_to_device(self, name_shard_pairs):
        params = dict(self.model.named_parameters())
        buffers = dict(self.model.named_buffers())
        for n, _ in name_shard_pairs:
            mf = self._FUSED_EXPERT_WEIGHT_RE.match(n)
            if mf is not None:
                self._ensure_fused_expert_params(mf.group("experts"))
                continue
            model_name = self._ckpt_to_model_name(n)
            p = params.get(model_name)
            if p is not None and p.device.type == "meta":
                t = torch.randn(p.shape, dtype=self.dtype, device=self.device) * self.meta_init_std
                set_module_tensor_to_device(self.model, model_name, self.device, value=t, dtype=self.dtype)
            elif model_name in buffers:
                # Buffers (the router bias) are built on CPU with their init values.
                b = buffers[model_name]
                value = torch.zeros(b.shape, dtype=self.dtype) if b.device.type == "meta" else b
                set_module_tensor_to_device(self.model, model_name, self.device, value=value, dtype=self.dtype)

    def _set_tensors(self, name_shard_pairs):
        if self.dummy:
            self._materialize_to_device(name_shard_pairs)
            return
        by_shard = {}
        for n, s in name_shard_pairs:
            by_shard.setdefault(s, []).append(n)
        weight_args = self.quantization_config.config_groups.group_0.weights
        for shard, names in by_shard.items():
            tensors = safe_load_file(shard, device=self.device)
            for n in names:
                # weight_shape / weight_scale are consumed with their weight_packed.
                if n.endswith(".weight_shape") or n.endswith(".weight_scale") or n.endswith("inv_freq"):
                    continue
                if n.endswith(".weight_packed"):
                    base = n[: -len(".weight_packed")]
                    w = self.compressor.decompress_weight({
                        "weight_packed": tensors[f"{base}.weight_packed"],
                        "weight_scale": tensors[f"{base}.weight_scale"],
                        "weight_shape": tensors[f"{base}.weight_shape"],
                    }, weight_args)
                    self._write_weight(f"{base}.weight", w)
                    continue
                self._write_weight(n, tensors[n])
            del tensors
        gc.collect()

    def _offload_names_to_meta(self, name_shard_pairs):
        names = [n if isinstance(n, str) else n[0] for n in name_shard_pairs]
        fused = set()
        for n in names:
            if n.endswith(".weight_shape") or n.endswith(".weight_scale") or n.endswith("inv_freq"):
                continue
            if n.endswith(".weight_packed"):
                n = n[: -len(".weight_packed")] + ".weight"
            m = self._PER_EXPERT_WEIGHT_RE.match(n) or self._FUSED_EXPERT_WEIGHT_RE.match(n)
            if m is not None:
                fused.add(m.group("experts"))
                continue
            set_module_tensor_to_device(self.model, self._ckpt_to_model_name(n), "meta")
        for experts_prefix in fused:
            mod = self._module(experts_prefix)
            for pname, shape in self._fused_expert_shapes().items():
                setattr(mod, pname, torch.nn.Parameter(
                    torch.empty(shape, device="meta", dtype=self.dtype), requires_grad=False,
                ))
        torch.cuda.empty_cache()

    def _rotary_emb(self):
        return self._text_model().rotary_emb

    def move_embed_to(self, device):
        names = self._names_from_ckpt(["language_model.model.embed_tokens"])
        on_gpu = str(device).startswith("cuda")
        if on_gpu:
            self._set_tensors(names)
        else:
            self._offload_names_to_meta(names)
        # rotary_emb's inv_freq is computed at build time (on CPU), not stored in the
        # checkpoint; keep it next to the embedding the forward starts from.
        self._rotary_emb().to(self.device if on_gpu else "cpu")

    def move_output_heads_to(self, device):
        names = []
        names += self._names_from_ckpt("language_model.model.norm")
        names += self._names_from_ckpt("language_model.lm_head")
        if str(device).startswith("cuda"):
            self._set_tensors(names)
        else:
            self._offload_names_to_meta(names)

    def save_prefixes_to_disc(self, prefixes, exclude=()):
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        prefixes = [p for p in prefixes if not any(x in p for x in exclude)]
        if not prefixes:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        for pfx in prefixes:
            sd = self._module(pfx).state_dict(keep_vars=True)
            to_save = {
                f"{pfx}.{local_name}": tensor.detach().cpu()
                for local_name, tensor in sd.items()
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda
            }
            path = os.path.join(self.save_dir, f"{pfx.replace('.', '_')}.safetensors")
            safe_save_file(to_save, path)

    def load_from_disc(self, layer_name):
        """Load a finished layer from its shards (the formats ``BaseModelWrapper`` writes).

        Shard keys are checkpoint names; per-expert weights are decompressed and
        written into the fused params through ``_write_weight``.
        """
        quantization_args = self.configure_quantization_from_config()
        prefixes = self._layer_prefixes(layer_name)
        paths = [
            os.path.join(self.save_dir, f"{p.replace('.', '_')}.safetensors")
            for p in prefixes["non_mlp"] + prefixes["mlp"]
            if self._FUSED_EXPERT_WEIGHT_RE.match(p) is None
        ]
        skip_suffixes = (".weight_shape", ".weight_scale", ".weight_global_scale",
                         ".bitmask", ".row_offsets", ".shape", "inv_freq")
        for path in paths:
            tensors = safe_load_file(path, device="cpu")
            for name in tensors:
                if name.endswith(skip_suffixes):
                    continue
                if name.endswith(".weight_packed"):
                    base = name[: -len(".weight_packed")]
                    if f"{base}.weight_global_scale" in tensors:
                        w = NVFP4PackedCompressor().decompress_weight({
                            "weight_packed": tensors[f"{base}.weight_packed"],
                            "weight_scale": tensors[f"{base}.weight_scale"],
                            "weight_global_scale": tensors[f"{base}.weight_global_scale"],
                        }, quantization_args)
                    else:
                        w = self.compressor.decompress_weight({
                            "weight_packed": tensors[f"{base}.weight_packed"],
                            "weight_scale": tensors[f"{base}.weight_scale"],
                            "weight_shape": tensors[f"{base}.weight_shape"],
                        }, quantization_args)
                    self._write_weight(f"{base}.weight", w)
                    continue
                if name.endswith(".compressed"):
                    base = name[: -len(".compressed")]
                    if f"{base}.row_offsets" in tensors:
                        w = BitmaskCompressor(BitmaskConfig()).decompress_weight({
                            "compressed": tensors[f"{base}.compressed"],
                            "bitmask": tensors[f"{base}.bitmask"],
                            "row_offsets": tensors[f"{base}.row_offsets"],
                            "shape": tensors[f"{base}.shape"],
                        })
                    else:
                        w = Sparse24BitMaskCompressor(Sparse24BitMaskConfig()).decompress_weight({
                            "compressed": tensors[f"{base}.compressed"],
                            "bitmask": tensors[f"{base}.bitmask"],
                            "shape": tensors[f"{base}.shape"],
                        })
                    self._write_weight(f"{base}.weight", w)
                    continue
                if (
                    getattr(self, "is_gsq", False)
                    and name.endswith(".weight")
                    and tensors[name].dtype in (torch.int32, torch.int64)
                    and f"{name}_scale" in tensors
                ):
                    # GSQ stores Humming's packed uint2 codes under the plain `.weight` key.
                    w = dequantize_gsq_packed(
                        tensors[name], tensors[f"{name}_scale"], getattr(self, "_init_wbits", 2)
                    )
                    self._write_weight(name, w)
                    continue
                self._write_weight(name, tensors[name])

    def _load_layer_for_eval(self, layer_idx, read_from_disk):
        layer_name = f"{self.layer_prefix}.{layer_idx}"
        if layer_idx <= read_from_disk and self._is_moe_layer_idx(layer_idx):
            self.load_from_disc(layer_name)
        else:
            self.move_layer_to_gpu(layer_name)

    # ------------------------------------------------------------------ forward pieces

    def _build_layer_inputs(self, batch_size):
        # Calibration sequences are unpadded, so the causal mask is implicit for both
        # flash_attention_2 and sdpa. The captured rope (cos, sin) has batch dim 1 and
        # broadcasts over any batch size.
        inputs = dict(self.kwargs)
        inputs["attention_mask"] = None
        return inputs

    def _run_attention(self, layer, hidden_states, layer_inputs):
        out = layer.self_attn(hidden_states, **layer_inputs)
        return out[0] if isinstance(out, tuple) else out

    def _route(self, layer, hidden):
        """DeepSeek-V3 ``noaux_tc`` routing: ``(topk_idx [N, k], topk_weight [N, k] fp32)``.

        Selection uses sigmoid scores plus ``e_score_correction_bias`` with grouped
        top-k (``n_group`` / ``topk_group``); the weights are the un-biased sigmoid
        scores of the selected experts, normalised when ``norm_topk_prob`` and scaled
        by ``routed_scaling_factor``. The native router computes exactly the
        remote-code ``MoEGate``; this returns its outputs in ``MoEGate``'s
        ``(topk_idx, topk_weight)`` order.
        """
        _, topk_weight, topk_idx = layer.mlp.gate(hidden.reshape(-1, self.hidden_size))
        return topk_idx, topk_weight

    def _dense_forward(self, layer, mlp_input):
        return layer.mlp(layer.post_attention_layernorm(mlp_input)) + mlp_input

    @torch.no_grad()
    def get_layer_activations(self, data_all):
        current_layer = self.get_layer_module(self.current_layer_idx)
        num_samples = data_all['input'].shape[0]
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, num_samples)
            x = data_all['input'][start_idx:end_idx].to(self.device, non_blocking=True)
            layer_inputs = self._build_layer_inputs(x.shape[0])
            mlp_input = x + self._run_attention(current_layer, current_layer.input_layernorm(x), layer_inputs)
            if self._is_moe_layer_idx(self.current_layer_idx):
                out = self.run_expert_parallel(mlp_input)
            else:
                out = self._dense_forward(current_layer, mlp_input)
            data_all['input'][start_idx:end_idx] = out.detach().cpu()

    def get_mlp_input(self, batch):
        current_layer = self.get_layer_module(self.current_layer_idx)
        layer_inputs = self._build_layer_inputs(batch.shape[0])
        return self._run_attention(current_layer, current_layer.input_layernorm(batch), layer_inputs) + batch

    @torch.no_grad()
    def get_mlp_output(self, mlp_input_batch):
        return self.run_expert_parallel(mlp_input_batch)

    def _dispatch_tokens(self, mlp_input_batch, with_weights=True):
        """Route tokens to their experts' owner ranks via all-to-all.

        Returns ``(x_flat, hidden, send_idx_flat, in_split_sizes, out_split_sizes,
        xin, win, eids, B, T, H)``; ``win`` is None when ``with_weights`` is False.
        """
        layer = self.get_layer_module(self.current_layer_idx)
        device = self.device
        pg = dist.group.WORLD

        B, T, H = mlp_input_batch.shape
        hidden = layer.post_attention_layernorm(mlp_input_batch)
        x_flat = hidden.reshape(B * T, H)

        topi, topw = self._route(layer, hidden)
        top_k = topi.shape[-1]

        tok_idx_flat = torch.arange(B * T, device=device, dtype=torch.long).repeat_interleave(top_k)
        eid_flat = topi.reshape(-1).to(torch.long)

        owners_flat = self._owner_lut.to(device)[eid_flat]
        perm = torch.argsort(owners_flat, stable=True)
        owners_flat = owners_flat.index_select(0, perm)
        send_idx_flat = tok_idx_flat.index_select(0, perm)
        send_eid_flat = eid_flat.index_select(0, perm)
        send_x_flat = x_flat.index_select(0, send_idx_flat)

        in_sizes_tensor = torch.bincount(owners_flat, minlength=self.world_size).to(torch.long)
        all_sizes = [torch.empty_like(in_sizes_tensor) for _ in range(self.world_size)]
        dist.all_gather(all_sizes, in_sizes_tensor, group=pg)
        out_split_sizes = torch.stack(all_sizes)[:, self.rank].tolist()
        in_split_sizes = in_sizes_tensor.tolist()

        xin = AllToAllTokens.apply(send_x_flat, out_split_sizes, in_split_sizes, pg)
        win = None
        if with_weights:
            send_w_flat = topw.reshape(-1).to(self.dtype).index_select(0, perm)
            win = AllToAllTokens.apply(send_w_flat.unsqueeze(1), out_split_sizes, in_split_sizes, pg).to(self.dtype)
        eids = AllToAllTokens.apply(send_eid_flat.unsqueeze(1), out_split_sizes, in_split_sizes, pg).squeeze(1)

        return x_flat, hidden, send_idx_flat, in_split_sizes, out_split_sizes, xin, win, eids, B, T, H

    def _expert_weights(self, experts, eid):
        """(gate, up, down) views of one owned expert's rows in the fused params."""
        row = self._local_expert_index(eid)
        inter = self.moe_intermediate_size
        gate_up = experts.gate_up_proj[row]
        return gate_up[:inter], gate_up[inter:], experts.down_proj[row]

    def _batched_expert_forward(self, xin, eids, compressed_weights=None, fake_act_quant=False):
        """Run every received token through its (owned) expert, grouped by expert id.

        Each expert is ``down(silu(gate(x)) * up(x))`` with separate gate and up GEMMs.
        With ``compressed_weights`` the trainer's per-expert matrices replace the
        fused params.
        """
        layer = self.get_layer_module(self.current_layer_idx)
        layer_key = self.get_current_layer()
        experts = layer.mlp.experts

        eids_long = eids.to(torch.long)
        unique_eids, inverse, counts = torch.unique(eids_long, sorted=True, return_inverse=True, return_counts=True)
        sort_idx = torch.argsort(inverse, stable=True)
        sorted_x = xin.index_select(0, sort_idx)
        out_buf = torch.empty_like(sorted_x)

        def _q(x):
            # Activation scale block must align with the weight's dense scale group
            # (= self.groupsize) so the sparse NVFP4 microscale matmul factors per block.
            return fake_quantize_activation_nvfp4(x, groupsize=self.groupsize) if fake_act_quant else x

        def _w(qw, proj):
            w = qw[proj]
            return w[0] if isinstance(w, tuple) else w

        offset = 0
        for i, eid_val in enumerate(unique_eids.tolist()):
            n = counts[i].item()
            inp_e = sorted_x[offset:offset + n]
            if compressed_weights is not None:
                qw = compressed_weights[f"{layer_key}.mlp.experts.{eid_val}"]
                gate_w, up_w, down_w = _w(qw, "gate_proj"), _w(qw, "up_proj"), _w(qw, "down_proj")
                hidden = F.silu(F.linear(_q(inp_e), gate_w)) * F.linear(_q(inp_e), up_w)
                out_e = F.linear(_q(hidden), down_w)
            else:
                gate_w, up_w, down_w = self._expert_weights(experts, eid_val)
                out_e = F.linear(F.silu(F.linear(inp_e, gate_w)) * F.linear(inp_e, up_w), down_w)
            out_buf[offset:offset + n] = out_e
            offset += n

        return out_buf.index_select(0, torch.argsort(sort_idx))

    def run_expert_parallel(self, mlp_input_batch, compressed_weights=None):
        pg = dist.group.WORLD
        x_flat, hidden, send_idx_flat, in_split_sizes, out_split_sizes, xin, win, eids, B, T, H = \
            self._dispatch_tokens(mlp_input_batch)

        out_local = self._batched_expert_forward(xin, eids, compressed_weights)
        returned = AllToAllTokens.apply(out_local * win, in_split_sizes, out_split_sizes, pg)

        y_flat = x_flat.new_zeros(x_flat.shape)
        y_flat.index_add_(0, send_idx_flat, returned)
        y = y_flat.view(B, T, H)

        layer = self.get_layer_module(self.current_layer_idx)
        y = y + layer.mlp.shared_experts(hidden)
        return y + mlp_input_batch

    def calculate_mse(self, mlp_input_batch, compressed_weights, self_attn=False, validation=False,
                      accumulation_steps=1, fake_act_quant=False):
        gate_weighted = getattr(self, "gate_weight_exponent", 0.0) > 0
        with torch.no_grad():
            _, _, _, _, _, xin, win, eids, _, _, _ = self._dispatch_tokens(mlp_input_batch, with_weights=gate_weighted)
            out_fp = self._batched_expert_forward(xin, eids, compressed_weights=None)
        out_q = self._batched_expert_forward(xin, eids, compressed_weights=compressed_weights,
                                             fake_act_quant=fake_act_quant)

        # win: each received (token, expert) row's router weight, for the gate-weighted loss
        total_mse = self._expert_recon_loss(out_q, out_fp, win)
        if not validation:
            (total_mse / accumulation_steps).backward()
        return total_mse.item()

    # ------------------------------------------------------------------ initialization

    def _expert_linear_views(self, experts, eid):
        """nn.Linear handles whose weights are views into one expert's fused rows."""
        mods = {}
        for proj, w in zip(("gate_proj", "up_proj", "down_proj"), self._expert_weights(experts, eid)):
            lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=False, device="meta", dtype=self.dtype)
            lin.weight = torch.nn.Parameter(w, requires_grad=False)
            mods[proj] = lin
        return mods

    def get_layer_initialization(self, trainer, gpt_all, config, logging):
        if logging is not None:
            logging = logging.logger
        layer_idx = self.current_layer_idx
        layer = self.get_layer_module(layer_idx)
        rank = self.rank
        self.configure_quantization_from_config(config)
        experts = layer.mlp.experts
        layer_key = self.get_current_layer()

        subset = {}
        for e in self._local_eids:
            for proj, lin in self._expert_linear_views(experts, e).items():
                subset[f"{layer_key}.mlp.experts.{e}.{proj}"] = lin

        init_method = config.init.method
        refine_enabled = config.refine.enabled

        if init_method in ("rtn", "random"):
            quantize_fn = random_quantize if init_method == "random" else rtn_quantize
            for name in subset:
                Q, scales = quantize_fn(subset[name], config, self.device, self.dtype)
                if refine_enabled:
                    trainer.setup_layer_training(name, Q, scales)
                else:
                    self.update_compressed_weights(name, (Q, scales))
            dist.barrier()
            return

        if init_method not in ("gptq", "obr", "jsq"):
            raise ValueError(
                f"Unknown init_method={init_method!r}. Supported: 'gptq', 'obr', 'jsq', 'rtn', 'random'"
            )

        gpts = {}
        for name in subset:
            gpts[name] = make_initializer(config, subset[name], name, self.device, self.dtype)
            if config.compression.quant_type == "nvfp4" or config.init.wbits < 16:
                gpts[name].quantizer = make_quantizer(config)

        # One calibration sample per step, one add_batch per (sample, expert that
        # received tokens). add_batch counts CALLS in nsamples, so this call pattern
        # -- the one KimiK25Wrapper's per-expert forward hooks produce -- fixes the
        # Hessian's normalisation; pooling an expert's tokens into one call would
        # rescale H and every GPTQ loss with it.
        n_hessian = config.init.nsamples // self.world_size
        chunk_tokens = config.init.hessian_chunk_tokens
        if logging is not None and rank == 0:
            logging.info(f"GPTQ Hessian accumulation: {n_hessian} samples, {self.num_local_experts} local experts")
        calib_start = time.time()
        calib_report_interval = max(1, n_hessian // self.calib_report_divisor)
        with torch.no_grad():
            for j in range(n_hessian):
                x = gpt_all['input'][j].unsqueeze(0).to(self.device, non_blocking=True)
                layer_inputs = self._build_layer_inputs(x.shape[0])
                x = x + self._run_attention(layer, layer.input_layernorm(x), layer_inputs)

                _, _, _, _, _, xin, _, eids, _, _, _ = self._dispatch_tokens(x, with_weights=False)
                eids = eids.to(torch.long)
                for eid_val in torch.unique(eids, sorted=True).tolist():
                    inp_e = xin[eids == eid_val]
                    base_prefix = f"{layer_key}.mlp.experts.{eid_val}"
                    gate_w, up_w, _ = self._expert_weights(experts, eid_val)
                    gpts[f"{base_prefix}.gate_proj"].add_batch(inp_e, None, chunk_tokens=chunk_tokens)
                    if init_method == "jsq":
                        # JSQ's range term depends on up's own output; range only
                        # (h=False) because up inherits gate's H below.
                        gpts[f"{base_prefix}.up_proj"].add_batch(inp_e, None, chunk_tokens=chunk_tokens, h=False)
                    inter = F.silu(F.linear(inp_e, gate_w)) * F.linear(inp_e, up_w)
                    gpts[f"{base_prefix}.down_proj"].add_batch(inter, None, chunk_tokens=chunk_tokens)
                    del inp_e, inter
                if rank == 0 and (j + 1) % calib_report_interval == 0:
                    report_gptq_calib(j + 1, n_hessian, time.time() - calib_start)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # A handle that received nothing is an expert with ZERO routed tokens; say so
        # explicitly instead of leaving fasterquant to infer it from a missing H. jsq's
        # up_proj keeps amax with nsamples 0 (h=False), which the amax test excludes.
        # Experts fed all-zero activations have nsamples > 0 and are handled inside
        # the initializers.
        for _mh in gpts.values():
            if (getattr(_mh, "nsamples", 0) == 0 and getattr(_mh, "amax", None) is None
                    and hasattr(_mh, "mark_no_tokens")):
                _mh.mark_no_tokens()

        def _quantize(name, **kw):
            Q, scales = gpts[name].fasterquant(
                logging,
                percdamp=config.init.percdamp,
                blocksize=config.init.blocksize,
                groupsize=config.compression.groupsize,
                static_groups=config.init.static_groups,
                prunen=config.compression.prunen,
                prunem=config.compression.prunem,
                **kw,
            )
            if hasattr(gpts[name], 'last_gptq_loss'):
                gptq_losses.append(gpts[name].last_gptq_loss)
            if scales is not None and refine_enabled:
                trainer.setup_layer_training(
                    name,
                    Q,
                    scales,
                    init_dense_weight=gpts[name].last_init_dense_weight,
                    init_support_mask=gpts[name].last_init_support_mask,
                )
            else:
                self.update_compressed_weights(name, (Q, scales) if scales is not None else Q)

        gptq_losses = []
        for name in gpts:
            if "up_proj" in name:
                continue
            _quantize(name)
            if "gate_proj" in name:
                # up_proj shares gate_proj's input and therefore its Hessian.
                up_name = f"{name[: -len('.gate_proj')]}.up_proj"
                gpts[up_name].H = gpts[name].H
                gpts[up_name].dead = gpts[name].dead
                _quantize(up_name, calculate_cholesky=False)
                gpts[name].free()
                gpts[up_name].free()
            else:
                gpts[name].free()

        if gptq_losses:
            trainer.gptq_avg_loss = sum(gptq_losses) / len(gptq_losses)

        del subset
        torch.cuda.empty_cache()
        dist.barrier()

    def update_compressed_weights(self, layer_name, compressed_weights):
        m = self._EXPERT_KEY_RE.search(layer_name)
        if m is None:
            layer = self._get_layer_by_name(layer_name)
            if isinstance(compressed_weights, tuple):
                Q, scales, global_scale, mask = self._split_weight_payload(compressed_weights)
                self.temp_weights[layer_name] = layer.weight.data
                self.temp_weights[f"{layer_name}.scale"] = scales
                if global_scale is not None:
                    self.temp_weights[f"{layer_name}.global_scale"] = global_scale
                if mask is not None:
                    self.temp_weights[f"{layer_name}.mask"] = mask
            else:
                Q = compressed_weights
            with torch.no_grad():
                layer.weight.data = Q.to(layer.weight.device).to(layer.weight.dtype)
            return

        # A fused expert has no per-expert module to hold Q, so the quantized matrix is
        # kept as `<name>.Q` for save_moe_experts_to_disc and also written into the
        # fused rows for the forward.
        if isinstance(compressed_weights, tuple):
            qw, scales, global_scale, mask = self._split_weight_payload(compressed_weights)
            # Held in the model dtype, as a per-expert nn.Linear would hold it.
            qw = qw.to(self.dtype)
            self.temp_weights[f"{layer_name}.Q"] = qw
            self.temp_weights[f"{layer_name}.scale"] = scales
            if global_scale is not None:
                self.temp_weights[f"{layer_name}.global_scale"] = global_scale
            if mask is not None:
                self.temp_weights[f"{layer_name}.mask"] = mask
        else:
            qw = compressed_weights
        self._write_expert_weight(layer_name[:m.start()] + ".mlp.experts", m.group(1), m.group(2), qw)

    # ------------------------------------------------------------------ evaluation

    def _pad_token_id(self):
        pad = getattr(self.model.config, "pad_token_id", None)
        return pad if pad is not None else getattr(self.text_config, "pad_token_id", None)

    @torch.no_grad()
    def ppl_evaluation(self, read_from_disk=-1):
        dataset = get_dataset("open_thoughts", self.tokenizer)
        testloader = prepare_test_dataloader(
            dataset=dataset["test"],
            tokenizer=self.tokenizer,
            seqlen=self.model.seqlen,
            batch_size=4,
            world_size=self.world_size,
            rank=self.rank
        )

        pad_token_id = self._pad_token_id()
        if pad_token_id is not None:
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)
        else:
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

        self.model.eval()
        self.move_embed_to(self.device)
        self.move_output_heads_to(self.device)
        text_model = self._text_model()

        input_ids_cpu_list = []
        activations_cpu_list = []
        for batch in testloader:
            input_ids_cpu_list.append(batch["input_ids"].to("cpu", non_blocking=True).pin_memory())
            activations_cpu_list.append(None)
            del batch
        num_batches = len(input_ids_cpu_list)

        idx_copy = self.current_layer_idx
        transfer_stream = torch.cuda.Stream(device=self.device)

        self.current_layer_idx = 0
        self._load_layer_for_eval(0, read_from_disk)
        ppl_start = time.time()

        for i in range(self.num_layers):
            if self.rank == 0:
                report_ppl_layer(i, self.num_layers, elapsed=time.time() - ppl_start)
            self.current_layer_idx = i
            layer_name = f"{self.layer_prefix}.{i}"
            layer = self.get_layer_module(i)

            for b in range(num_batches):
                x_cpu = activations_cpu_list[b]
                if x_cpu is None:
                    x = text_model.embed_tokens(input_ids_cpu_list[b].to(self.device, non_blocking=True))
                else:
                    x = x_cpu.to(self.device, non_blocking=True)

                layer_inputs = self._build_layer_inputs(x.shape[0])
                x = x + self._run_attention(layer, layer.input_layernorm(x), layer_inputs)
                if self._is_moe_layer_idx(i):
                    x = self.run_expert_parallel(x)
                else:
                    x = self._dense_forward(layer, x)
                activations_cpu_list[b] = x.to("cpu", non_blocking=True).pin_memory()

            self.offload_to_meta(layer_name)
            torch.cuda.synchronize(self.device)

            if i + 1 < self.num_layers:
                with torch.cuda.stream(transfer_stream):
                    self._load_layer_for_eval(i + 1, read_from_disk)
                transfer_stream.synchronize()

            torch.cuda.empty_cache()

        local_nll_sum = torch.tensor(0.0, device=self.device)
        local_tok_cnt = torch.tensor(0.0, device=self.device)

        for b in range(num_batches):
            input_ids = input_ids_cpu_list[b].to(self.device, non_blocking=True)
            x = activations_cpu_list[b].to(self.device, non_blocking=True)

            logits = self.model.lm_head(text_model.norm(x))[:, :-1, :]
            shift_labels = input_ids[:, 1:]

            nll = loss_fn(logits.permute(0, 2, 1), shift_labels).float()
            mask = shift_labels != loss_fn.ignore_index
            local_nll_sum += (nll * mask).sum()
            local_tok_cnt += mask.sum()

            del input_ids, x, logits, shift_labels, nll, mask

        self.move_embed_to("meta")
        self.move_output_heads_to("meta")
        self.current_layer_idx = idx_copy

        dist.all_reduce(local_nll_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_tok_cnt, op=dist.ReduceOp.SUM)
        ppl = math.exp((local_nll_sum / local_tok_cnt).item())

        torch.cuda.synchronize(self.device)
        gc.collect()
        torch.cuda.empty_cache()
        return ppl
