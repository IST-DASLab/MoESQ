"""Qwen3-MoE on transformers 5.x, whose native Qwen3MoeExperts stores experts FUSED.

transformers >= 5 replaces the per-expert ``mlp.experts[e].{gate,up,down}_proj``
ModuleList with one ``Qwen3MoeExperts`` holding two 3D parameters,

    gate_up_proj  [num_experts, 2 * moe_intermediate, hidden]   (gate rows first)
    down_proj     [num_experts, hidden, moe_intermediate]

and turns ``mlp.gate`` into a ``Qwen3MoeTopKRouter`` (a bare ``weight`` Parameter
whose forward returns ``(logits, scores, indices)``). The Qwen/Qwen3-30B-A3B checkpoint
still stores experts per expert on disk. Storage handling mirrors Qwen35MoeWrapper
(qwen35_moe.py): each rank holds only the experts it owns, and checkpoint tensors are
written into -- or gathered out of -- the fused params by hand.

All routing, loss, validation and GPTQ logic is inherited from Qwen3MoeWrapper
through its three access points (_router_logits, _router_config, _expert_module), so
the math is the 4.x wrapper's, op for op. Shard names on disk stay per expert
(``...mlp.experts.<eid>.<proj>.weight_packed`` etc.), identical to the 4.x wrapper, so
save_model.py and the vLLM loader see exactly the same export.
"""
import gc
import os
import re

import torch
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
from transformers import AutoModelForCausalLM

from src.compression.quant.gsq import dequantize_gsq_packed
from .qwen3_moe import Qwen3MoeWrapper


def uses_fused_experts():
    """True when the installed transformers builds Qwen3-MoE with fused experts."""
    try:
        from transformers.models.qwen3_moe import modeling_qwen3_moe as m
    except ImportError:
        return False
    return hasattr(m, "Qwen3MoeExperts")


def _linear_view(weight):
    """nn.Linear whose weight SHARES storage with `weight` (a slice of a fused param).

    Built on meta so no storage is allocated before the view replaces it. An in-place
    write to the fused param is visible through the view and vice versa; rebinding
    ``.data`` on it would NOT reach the fused param, so writers go through
    Qwen3MoeFusedWrapper._write_expert_weight instead.
    """
    out_f, in_f = weight.shape
    lin = torch.nn.Linear(in_f, out_f, bias=False, device="meta")
    lin.weight = torch.nn.Parameter(weight, requires_grad=False)
    return lin


class _FusedExpertView(torch.nn.Module):
    """One expert of a fused Qwen3MoeExperts, laid out as the 4.x Qwen3MoeMLP.

    Children are declared gate, up, down -- the 4.x declaration order, which fixes
    the iteration order of named_modules() and therefore of GPTQ's per-linear loop.
    gate and up are two separate matmuls, as in 4.x, rather than one over the
    concatenated gate_up rows: a single wider GEMM may pick a different kernel and
    reduction order, and the port is meant to reproduce the 4.x numerics.
    """

    def __init__(self, gate_up_row, down_row, intermediate):
        super().__init__()
        self.gate_proj = _linear_view(gate_up_row[:intermediate])
        self.up_proj = _linear_view(gate_up_row[intermediate:])
        self.down_proj = _linear_view(down_row)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3MoeFusedWrapper(Qwen3MoeWrapper):
    """Qwen3MoeWrapper for transformers 5.x fused Qwen3MoeExperts."""

    # ``model.layers.N.mlp.experts.<eid>.<proj>.weight`` -- per-expert checkpoint key.
    _EXPERT_WEIGHT_RE = re.compile(
        r"^(?P<experts>.*\.mlp\.experts)\.(?P<eid>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
    )
    # ``model.layers.N.mlp.experts.{gate_up_proj,down_proj}`` -- already-fused checkpoint key.
    _FUSED_EXPERT_WEIGHT_RE = re.compile(
        r"^(?P<experts>.*\.mlp\.experts)\.(?P<pname>gate_up_proj|down_proj)$"
    )
    # ``...mlp.experts.<eid>.<proj>`` -- a per-expert linear, as named by GPTQ, the
    # trainer and the saved shards.
    _EXPERT_LINEAR_RE = re.compile(
        r"^(?P<layer>.*)\.mlp\.experts\.(?P<eid>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)$"
    )

    def __init__(self, model_name, tokenizer, batch_size, seqlen, device, dtype):
        super().__init__(model_name, tokenizer, batch_size, seqlen, device, dtype)
        cfg = self.model.config
        self.hidden_size = cfg.hidden_size
        self.moe_intermediate_size = cfg.moe_intermediate_size

        # Storage sharding: this rank's fused expert params hold only the experts it
        # owns, in ascending global-eid order. Every index into gate_up_proj /
        # down_proj must therefore go through _local_expert_index().
        self._local_eids = sorted(self.sharder.local_experts(self.rank))
        self._eid_to_local = {int(e): i for i, e in enumerate(self._local_eids)}
        self.num_local_experts = len(self._local_eids)

        # id(experts module) -> {eid: (view, gate_up_param, down_param)}. A view is
        # valid only while the fused params it was cut from are still installed.
        self._expert_views = {}

    def _build_empty_model(self, cfg):
        # flash-attn has no wheel for every torch/CUDA/Python combination transformers
        # 5.x runs on; sdpa is the numerically closest built-in fallback.
        from transformers.utils import is_flash_attn_2_available
        attn = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
        return AutoModelForCausalLM.from_config(
            cfg, attn_implementation=attn, trust_remote_code=True
        ).eval()

    # ── router / expert access points (see Qwen3MoeWrapper) ─────────────────────

    def _router_logits(self, layer, hidden):
        # Qwen3MoeTopKRouter.forward also does softmax/top-k, but the inherited
        # routing does that itself; only the logits (a bias-free linear) are needed.
        return F.linear(hidden, layer.mlp.gate.weight)

    def _router_config(self, layer):
        return layer.mlp.gate.top_k, layer.mlp.gate.norm_topk_prob

    def _expert_module(self, layer, eid):
        experts = layer.mlp.experts
        views = self._expert_views.setdefault(id(experts), {})
        hit = views.get(int(eid))
        if hit is not None and hit[1] is experts.gate_up_proj and hit[2] is experts.down_proj:
            return hit[0]
        row = self._local_expert_index(eid)
        view = _FusedExpertView(experts.gate_up_proj[row], experts.down_proj[row],
                                self.moe_intermediate_size)
        views[int(eid)] = (view, experts.gate_up_proj, experts.down_proj)
        return view

    def _get_layer_by_name(self, layer_name):
        m = self._EXPERT_LINEAR_RE.match(layer_name)
        if m is None:
            return super()._get_layer_by_name(layer_name)
        layer = self.model.get_submodule(m.group("layer"))
        return getattr(self._expert_module(layer, int(m.group("eid"))), m.group("proj"))

    # ── fused storage ───────────────────────────────────────────────────────────

    def _module_by_name(self, name):
        mod = self.model
        for part in name.split("."):
            mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
        return mod

    def _local_expert_index(self, eid):
        """Row of global expert `eid` inside this rank's sharded fused params.

        Raises for an eid this rank does not own: with sharded storage a stray global
        index below num_local_experts would otherwise read another expert's row.
        """
        idx = self._eid_to_local.get(int(eid), -1)
        if idx < 0:
            raise KeyError(
                f"expert {int(eid)} is not owned by rank {self.rank} "
                f"(owns {self.num_local_experts} of {self.num_experts})"
            )
        return idx

    def _fused_expert_shapes(self):
        """Shapes of this rank's sharded fused params: num_local_experts rows only.

        HF's own Qwen3MoeExperts.forward indexes by global eid and would break on
        these; nothing calls it -- all expert compute goes through _expert_module.
        """
        n, inter, hid = self.num_local_experts, self.moe_intermediate_size, self.hidden_size
        return {"gate_up_proj": (n, 2 * inter, hid), "down_proj": (n, hid, inter)}

    def _drop_expert_views(self, experts_module):
        # Views pin the storage of the params they were cut from.
        self._expert_views.pop(id(experts_module), None)

    def _ensure_fused_expert_params(self, experts_prefix):
        """Allocate a layer's sharded fused params on device (zeroed), once."""
        mod = self._module_by_name(experts_prefix)
        for pname, shape in self._fused_expert_shapes().items():
            p = getattr(mod, pname, None)
            if p is not None and p.device.type == "meta":
                self._drop_expert_views(mod)
                setattr(mod, pname, torch.nn.Parameter(
                    torch.zeros(shape, dtype=self.dtype, device=self.device),
                    requires_grad=False,
                ))

    def _write_expert_weight(self, layer_prefix, eid, proj, weight):
        """Write one per-expert 2D weight into its slice of the fused params.

        Layout follows transformers' own converter for this architecture
        (``MergeModulelist(dim=0), Concatenate(dim=1)`` over gate_proj, up_proj ->
        gate_up_proj): experts on dim 0, gate in the FIRST half of dim 1, up in the
        second. Swapping the halves swaps SiLU's gate and value paths silently.
        """
        experts_prefix = f"{layer_prefix}.mlp.experts"
        self._ensure_fused_expert_params(experts_prefix)
        experts = self._module_by_name(experts_prefix)
        row = self._local_expert_index(eid)
        inter = self.moe_intermediate_size
        with torch.no_grad():
            w = weight.to(experts.gate_up_proj.device).to(self.dtype)
            if proj == "gate_proj":
                experts.gate_up_proj[row, :inter].copy_(w)
            elif proj == "up_proj":
                experts.gate_up_proj[row, inter:].copy_(w)
            else:
                experts.down_proj[row].copy_(w)

    def _write_fused_expert_rows(self, match, weight):
        """Gather this rank's owned rows out of an ALREADY-fused [num_experts, ...] tensor.

        That tensor is the converter's output, already in the layout
        _write_expert_weight builds, so this is a pure row gather. `_local_eids` is
        ascending, so index_select lands global eid e on row _local_expert_index(e).
        """
        mod = self._module_by_name(match.group("experts"))
        param = getattr(mod, match.group("pname"))
        w = weight.to(dtype=self.dtype)
        if w.shape[0] != self.num_experts:
            raise ValueError(
                f"{match.group(0)}: fused expert tensor has {w.shape[0]} rows but the "
                f"config declares num_experts={self.num_experts}"
            )
        if tuple(w.shape[1:]) != tuple(param.shape[1:]):
            raise ValueError(
                f"{match.group(0)}: checkpoint trailing dims {tuple(w.shape[1:])} do not "
                f"match the model's {tuple(param.shape[1:])}"
            )
        idx = torch.as_tensor(self._local_eids, dtype=torch.long, device=w.device)
        with torch.no_grad():
            param.copy_(w.index_select(0, idx))

    def _set_tensors(self, name_shard_pairs):
        by_shard = {}
        for n, s in name_shard_pairs:
            by_shard.setdefault(s, []).append(n)
        for shard, names in by_shard.items():
            tensors = safe_load_file(shard, device=self.device)
            for n in names:
                if n.endswith("inv_freq"):
                    continue
                t = tensors[n]
                mf = self._FUSED_EXPERT_WEIGHT_RE.match(n)
                if mf is not None:
                    self._ensure_fused_expert_params(mf.group("experts"))
                    self._write_fused_expert_rows(mf, t)
                    continue
                m = self._EXPERT_WEIGHT_RE.match(n)
                if m is not None:
                    self._write_expert_weight(
                        m.group("experts")[: -len(".mlp.experts")],
                        int(m.group("eid")), m.group("proj"), t,
                    )
                    continue
                t = t.to(dtype=self.dtype, copy=False)
                set_module_tensor_to_device(self.model, n, self.device, value=t, dtype=self.dtype)
            del tensors
        gc.collect()

    def _offload_names_to_meta(self, name_shard_pairs):
        names = [n if isinstance(n, str) else n[0] for n in name_shard_pairs]
        fused = set()
        for n in names:
            if n.endswith("inv_freq"):
                continue
            m = self._FUSED_EXPERT_WEIGHT_RE.match(n) or self._EXPERT_WEIGHT_RE.match(n)
            if m is not None:
                fused.add(m.group("experts"))
                continue
            set_module_tensor_to_device(self.model, n, "meta")
        shapes = self._fused_expert_shapes()
        for experts_prefix in fused:
            mod = self._module_by_name(experts_prefix)
            self._drop_expert_views(mod)
            for pname, shape in shapes.items():
                setattr(mod, pname, torch.nn.Parameter(
                    torch.empty(shape, device="meta", dtype=self.dtype),
                    requires_grad=False,
                ))
        torch.cuda.empty_cache()

    def _layer_prefixes(self, layer_name):
        prefixes = super()._layer_prefixes(layer_name)
        layer_idx = int(layer_name.split('.')[-1])
        if self._is_moe_layer(layer_idx):
            # An already-fused checkpoint keeps every expert in these two tensors
            # (_write_fused_expert_rows keeps the owned rows). On a per-expert
            # checkpoint they match no key, and load_from_disc skips their absent shards.
            base = f"{self.layer_prefix}.{layer_idx}.mlp.experts"
            prefixes["mlp"] = prefixes["mlp"] + [f"{base}.gate_up_proj", f"{base}.down_proj"]
        return prefixes

    # ── compressed weights ──────────────────────────────────────────────────────

    def update_compressed_weights(self, layer_name, compressed_weights):
        m = self._EXPERT_LINEAR_RE.match(layer_name)
        if m is None:
            return super().update_compressed_weights(layer_name, compressed_weights)
        if isinstance(compressed_weights, tuple):
            Q, scales, global_scale, mask = self._split_weight_payload(compressed_weights)
            # The 4.x wrapper rebinds the expert Linear's weight to Q (cast to the
            # model dtype) and save_moe_experts_to_disc reads it back from there. A
            # fused slice has no Linear of its own, so Q is kept under `.Q`, which
            # save_moe_experts_to_disc prefers -- same bf16 values, same export.
            Q = Q.to(self.device).to(self.dtype)
            self.temp_weights[f"{layer_name}.Q"] = Q
            self.temp_weights[f"{layer_name}.scale"] = scales
            if global_scale is not None:
                self.temp_weights[f"{layer_name}.global_scale"] = global_scale
            if mask is not None:
                self.temp_weights[f"{layer_name}.mask"] = mask
        else:
            Q = compressed_weights
        self._write_expert_weight(m.group("layer"), int(m.group("eid")), m.group("proj"), Q)

    def _write_weight(self, name, value):
        """set_module_tensor_to_device(model, name, ...) that also reaches fused experts."""
        if name.endswith(".weight"):
            m = self._EXPERT_LINEAR_RE.match(name[: -len(".weight")])
            if m is not None:
                self._write_expert_weight(m.group("layer"), int(m.group("eid")),
                                          m.group("proj"), value)
                return
        set_module_tensor_to_device(self.model, name, self.device, value=value, dtype=self.dtype)

    def load_from_disc(self, layer_name):
        """BaseModelWrapper.load_from_disc, with expert weights written into fused slices.

        Same shard set, same formats (NVFP4 / pack-quantized weight_packed, sparse
        bitmask, GSQ uint2, plain) and same decompression; only the destination of a
        per-expert weight differs. The two fused-name shards from _layer_prefixes do
        not exist for runs of this pipeline and are skipped; every other shard must
        exist, as in the base implementation.
        """
        quantization_args = self.configure_quantization_from_config()
        prefixes = self._layer_prefixes(layer_name)

        files = {}
        for item in prefixes:
            for p in prefixes[item]:
                files[p] = os.path.join(self.save_dir, f"{p.replace('.', '_')}.safetensors")

        for p, path in files.items():
            if self._FUSED_EXPERT_WEIGHT_RE.match(p) and not os.path.isfile(path):
                continue
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
                            "weight_shape": tensors[f"{base}.weight_shape"],
                        }
                        W_deq = self.compressor.decompress_weight(compressed_data, quantization_args)
                    self._write_weight(f"{base}.weight", W_deq)
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
                            "shape": tensors[f"{base}.shape"],
                        }
                        W_deq = Sparse24BitMaskCompressor(Sparse24BitMaskConfig()).decompress_weight(
                            compressed_data
                        )
                    self._write_weight(f"{base}.weight", W_deq)
                    continue
                if (
                    getattr(self, "is_gsq", False)
                    and name.endswith(".weight")
                    and tensors[name].dtype in (torch.int32, torch.int64)
                    and f"{name}_scale" in tensor_names
                ):
                    W_deq = dequantize_gsq_packed(
                        tensors[name], tensors[f"{name}_scale"], getattr(self, "_init_wbits", 2)
                    )
                    self._write_weight(name, W_deq)
                    continue
                self._write_weight(name, tensors[name])
