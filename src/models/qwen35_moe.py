import re
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
import math
import os, gc
from accelerate.utils import set_module_tensor_to_device
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file
from transformers import AutoModelForCausalLM
from src.compression.ct_compat import NVFP4PackedCompressor
from .base import BaseModelWrapper
from src.moe.placement import ExpertSharder
from src.moe.autograd_ops import AllToAllTokens
from src.compression.activation_quant import fake_quantize_activation_nvfp4
from src.compression.quant.gsq import dequantize_gsq_packed
from src.compression.initialization.gptq import *
from src.compression.initialization import make_initializer
from src.evaluation.wiki_eval import *
from src.utils.progress_reporter import report_ppl_layer


class Qwen35MoeWrapper(BaseModelWrapper):
    """Distributed wrapper for Qwen3.5 MoE with expert parallelism.

    Handles 3D fused expert tensors, shared experts, and hybrid attention.
    Expert sharding follows the same all-to-all dispatch pattern as Kimi K2.
    """

    def __init__(self, model_name, tokenizer, batch_size, seqlen, device, dtype):
        super().__init__(model_name, tokenizer, batch_size, seqlen, device, dtype)

        self._text_model = self._resolve_text_model()
        cfg = self._text_model.config if hasattr(self._text_model, 'config') else self.model.config
        text_cfg = getattr(cfg, 'text_config', cfg)

        self.layer_prefix = self._resolve_layer_prefix()
        self.num_layers = len(self._get_layers())
        self.num_experts = text_cfg.num_experts
        self.num_experts_per_tok = text_cfg.num_experts_per_tok
        self.layer_types = getattr(text_cfg, 'layer_types', ["full_attention"] * self.num_layers)
        self.is_moe = True
        self.hidden_size = text_cfg.hidden_size
        self.moe_intermediate_size = text_cfg.moe_intermediate_size
        self._fp8_block = self._detect_fp8_block_quant(cfg, text_cfg)

        self._ckpt_prefix_remap = self._detect_ckpt_prefix_remap()

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.sharder = ExpertSharder(num_experts=self.num_experts, world_size=self.world_size)
        self.groupsize = 32

        self._owner_lut = torch.tensor(
            [self.sharder.owner(e) for e in range(self.num_experts)],
            dtype=torch.long
        )

        # Storage sharding: this rank's fused expert params hold only the experts
        # it owns, in ascending global-eid order. Every index into gate_up_proj /
        # down_proj must therefore go through _local_expert_index().
        self._local_eids = sorted(self.sharder.local_experts(self.rank))
        self._eid_to_local = {int(e): i for i, e in enumerate(self._local_eids)}
        self.num_local_experts = len(self._local_eids)

    def _build_empty_model(self, cfg):
        """Build the text-only CausalLM from text_config (matching the official
        ``Qwen3_5MoeForCausalLM.from_pretrained`` pathway).  The checkpoint name
        remapping (``model.language_model.`` → ``model.``) is handled by
        ``_detect_ckpt_prefix_remap``.
        """
        from_cfg = cfg
        if hasattr(cfg, 'text_config') and cfg.text_config is not None:
            from_cfg = cfg.text_config
            from_cfg._attn_implementation = "flash_attention_2"
        return AutoModelForCausalLM.from_config(
            from_cfg, attn_implementation="flash_attention_2", trust_remote_code=True
        ).eval()

    def _detect_ckpt_prefix_remap(self):
        """Detect if checkpoint names differ from model parameter names.

        Qwen3.5 checkpoints are saved as multimodal models with prefix
        ``model.language_model.``  but when loaded as CausalLM (text-only)
        the model parameters live directly under ``model.``.  Returns a
        (ckpt_prefix, model_prefix) tuple to translate, or None if no
        remapping is needed.
        """
        if self._name_to_shard is None:
            return None
        model_names = {n for n, _ in self.model.named_parameters()}
        ckpt_names = list(self._name_to_shard)

        # Score over MANY keys, not one. Sampling a single arbitrary key
        # (`next(iter(...))`) silently fails whenever that key happens to be one the
        # model does not hold under the same name -- and a per-expert expert weight is
        # exactly such a key, because the model stores experts FUSED. Checkpoints
        # that store experts per-expert (e.g. expert-pruned ones with fewer routed
        # experts than the base config) would otherwise miss the remap, and every
        # tensor, including the embedding, would stay on the meta device.
        def hits(transform, limit=4096):
            n = 0
            for name in ckpt_names[:limit]:
                if transform(name) in model_names:
                    n += 1
                    if n >= 8:
                        break
            return n

        if hits(lambda n: n) >= 8:
            return None
        for ckpt_pfx, model_pfx in [
            ("model.language_model.", "model."),
            ("model.text_model.", "model."),
        ]:
            if hits(lambda n, a=ckpt_pfx, b=model_pfx:
                    n.replace(a, b, 1) if n.startswith(a) else n) >= 8:
                return (ckpt_pfx, model_pfx)
        return None

    def _ckpt_to_model_name(self, ckpt_name):
        if self._ckpt_prefix_remap is not None:
            ckpt_pfx, model_pfx = self._ckpt_prefix_remap
            if ckpt_name.startswith(ckpt_pfx):
                return ckpt_name.replace(ckpt_pfx, model_pfx, 1)
        return ckpt_name

    def _model_to_ckpt_name(self, model_name):
        if self._ckpt_prefix_remap is not None:
            ckpt_pfx, model_pfx = self._ckpt_prefix_remap
            if model_name.startswith(model_pfx):
                return model_name.replace(model_pfx, ckpt_pfx, 1)
        return model_name

    def _names_from_ckpt(self, prefixes):
        if self._ckpt_prefix_remap is None:
            return super()._names_from_ckpt(prefixes)
        ckpt_pfx, model_pfx = self._ckpt_prefix_remap
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        ckpt_prefixes = [p.replace(model_pfx, ckpt_pfx, 1) if p.startswith(model_pfx) else p
                         for p in prefixes]
        return super()._names_from_ckpt(ckpt_prefixes)

    # ``model.layers.N.mlp.experts.<eid>.<proj>.weight`` -- the checkpoint keeps
    # experts split per-expert, while the model holds them fused (see
    # _write_fused_expert_slice).
    _EXPERT_WEIGHT_RE = re.compile(
        r"^(?P<experts>.*\.mlp\.experts)\.(?P<eid>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
    )

    # Some Qwen3.5 releases ship the expert weights ALREADY fused as 3D tensors --
    # `model.language_model.layers.N.mlp.experts.{gate_up_proj,down_proj}` -- with no
    # per-expert `.{eid}.gate_proj.weight` entries (only the `mtp.` head has them).
    # The per-expert regex above matches nothing there, so these must be loaded
    # through this pattern or the fused params stay on the meta device.
    _FUSED_EXPERT_WEIGHT_RE = re.compile(
        r"^(?P<experts>.*\.mlp\.experts)\.(?P<pname>gate_up_proj|down_proj)$"
    )

    @staticmethod
    def _detect_fp8_block_quant(cfg, text_cfg):
        """Block size of a DeepSeek/Qwen-style block-FP8 checkpoint, else None.

        None means "not block-FP8", and every weight then loads as a plain cast
        exactly as before -- BF16 checkpoints are unaffected by this path.
        """
        qc = getattr(cfg, "quantization_config", None)
        if qc is None:
            qc = getattr(text_cfg, "quantization_config", None)
        if qc is None:
            return None
        if not isinstance(qc, dict):
            qc = qc.to_dict() if hasattr(qc, "to_dict") else vars(qc)
        if str(qc.get("quant_method", "")).lower() != "fp8":
            return None
        block = tuple(qc.get("weight_block_size") or ())
        return block if len(block) == 2 else None

    @staticmethod
    def _dequant_fp8_block(w, scale_inv, block):
        """Block-wise FP8 -> float32: ``w_deq[i, j] = w[i, j] * scale_inv[i // b0, j // b1]``.

        Despite the name, ``weight_scale_inv`` is the multiplier applied at
        dequantization time (DeepSeek convention), not its reciprocal.
        """
        b0, b1 = block
        out_f, in_f = w.shape
        s = scale_inv.to(torch.float32)
        s = s.repeat_interleave(b0, dim=0)[:out_f, :]
        s = s.repeat_interleave(b1, dim=1)[:, :in_f]
        return w.to(torch.float32) * s

    def _module_by_name(self, name):
        mod = self.model
        for part in name.split("."):
            mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
        return mod

    def _local_expert_index(self, eid):
        """Row of global expert `eid` inside this rank's sharded fused params.

        Raises for an eid this rank does not own. That matters: with sharded
        storage a stray global index below `num_local_experts` would otherwise
        land on some *other* expert's row and read plausible-looking but wrong
        weights, and only eids >= num_local_experts would raise IndexError on
        their own. Failing here makes every miss loud.
        """
        idx = self._eid_to_local.get(int(eid), -1)
        if idx < 0:
            raise KeyError(
                f"expert {int(eid)} is not owned by rank {self.rank} "
                f"(owns {self.num_local_experts} of {self.num_experts})"
            )
        return idx

    def _fused_expert_shapes(self):
        """Shapes of this rank's *sharded* fused expert params.

        Only `num_local_experts` rows, not `num_experts` -- un-owned experts cost
        nothing. HF's own `Qwen3_5MoeExperts.forward` indexes these by global eid
        and would break on them, but nothing here calls it: all expert compute
        goes through `_batched_expert_forward`.
        """
        n, I, H = self.num_local_experts, self.moe_intermediate_size, self.hidden_size
        return {"gate_up_proj": (n, 2 * I, H), "down_proj": (n, H, I)}

    def _ensure_fused_expert_params(self, experts_prefix):
        """Materialise a layer's fused expert params on device, once.

        Loading and compute are both sharded (`_layer_prefixes` reads only owned
        experts; dispatch routes only owned eids) but *storage* deliberately is
        not: HF keeps all experts in one `[num_experts, ...]` Parameter, so a
        rank cannot leave un-owned slices on meta the way the Kimi ModuleList
        does. We allocate the whole tensor zeroed and fill just the owned
        experts. Un-owned slices stay zero and are never read.

        The alternative -- allocate `[n_local, ...]` and remap global->local --
        is ~9 mechanical index sites and would save 42 GiB/rank at ws=8. Not
        done: nothing calls HF's native `Qwen3_5MoeExperts.forward` (which does
        index by global eid), so it stays available if headroom ever binds.
        """
        mod = self._module_by_name(experts_prefix)
        for pname, shape in self._fused_expert_shapes().items():
            p = getattr(mod, pname, None)
            if p is not None and p.device.type == "meta":
                setattr(mod, pname, torch.nn.Parameter(
                    torch.zeros(shape, dtype=self.dtype, device=self.device),
                    requires_grad=False,
                ))

    def _write_fused_expert_slice(self, match, weight):
        """Place one per-expert 2D weight into the fused 3D expert parameter.

        Mirrors transformers' own converter for this architecture --
        ``operations=[MergeModulelist(dim=0), Concatenate(dim=1)]`` over
        ``["...gate_proj.weight", "...up_proj.weight"] -> "...gate_up_proj"`` --
        so experts stack on dim 0 and **gate takes the first half of dim 1**, up
        the second. Reversing that order silently swaps SiLU's gate and value
        paths, which trains and evaluates without ever raising.
        """
        mod = self._module_by_name(match.group("experts"))
        row = self._local_expert_index(match.group("eid"))
        proj = match.group("proj")
        w = weight.to(dtype=self.dtype)
        I = self.moe_intermediate_size
        with torch.no_grad():
            if proj == "gate_proj":
                mod.gate_up_proj[row, :I].copy_(w)
            elif proj == "up_proj":
                mod.gate_up_proj[row, I:].copy_(w)
            else:
                mod.down_proj[row].copy_(w)

    def _write_fused_expert_rows(self, match, weight):
        """Gather this rank's owned rows out of an ALREADY-fused 3D expert tensor.

        The checkpoint tensor is [num_experts, ...] in global-eid order and is
        already in the layout `_write_fused_expert_slice` builds by hand (experts on
        dim 0, gate in the first half of dim 1, up in the second) -- it IS the output
        of transformers' own converter. So there is nothing to re-order here: this is
        a pure row gather, which is why it cannot silently swap SiLU's gate and value
        paths the way a hand-rolled concat can.

        `_local_eids` is sorted ascending and `_eid_to_local` maps eid -> its index in
        that list, so `index_select(0, _local_eids)` lands global eid exactly on the
        row `_local_expert_index()` will later read it from.
        """
        mod = self._module_by_name(match.group("experts"))
        pname = match.group("pname")
        param = getattr(mod, pname)
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
        if self.dummy:
            self._materialize_to_device(name_shard_pairs)
            return
        by_shard = {}
        for n, s in name_shard_pairs:
            by_shard.setdefault(s, []).append(n)
        for shard, names in by_shard.items():
            tensors = safe_load_file(shard, device=self.device)
            for n in names:
                # weight_scale_inv is consumed alongside its weight, never on its own.
                if n.endswith("inv_freq") or n.endswith("weight_scale_inv"):
                    continue

                t = tensors[n]
                if self._fp8_block is not None:
                    s_inv = tensors.get(f"{n}_scale_inv")
                    if s_inv is not None:
                        t = self._dequant_fp8_block(t, s_inv, self._fp8_block)
                    elif t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                        # Casting FP8 straight to bf16 drops the block scale and is
                        # wrong by a per-block factor without ever raising. Refuse.
                        raise KeyError(
                            f"{n} is {t.dtype} but {n}_scale_inv is absent from "
                            f"{os.path.basename(shard)}; refusing to load it as a plain cast."
                        )

                model_name = self._ckpt_to_model_name(n)
                mf = self._FUSED_EXPERT_WEIGHT_RE.match(model_name)
                if mf is not None:
                    self._ensure_fused_expert_params(mf.group("experts"))
                    self._write_fused_expert_rows(mf, t)
                    continue
                m = self._EXPERT_WEIGHT_RE.match(model_name)
                if m is not None:
                    self._ensure_fused_expert_params(m.group("experts"))
                    self._write_fused_expert_slice(m, t)
                    continue

                # Integer buffers (hash multipliers, indices) keep their dtype.
                set_module_tensor_to_device(
                    self.model, model_name, self.device,
                    value=t.to(dtype=self.dtype) if t.is_floating_point() else t, dtype=self.dtype,
                )
            del tensors
        gc.collect()

    def _offload_names_to_meta(self, name_shard_pairs):
        names = [n if isinstance(n, str) else n[0] for n in name_shard_pairs]
        fused = set()
        for n in names:
            if n.endswith("inv_freq") or n.endswith("weight_scale_inv"):
                continue
            model_name = self._ckpt_to_model_name(n)
            mf = self._FUSED_EXPERT_WEIGHT_RE.match(model_name)
            if mf is not None:
                fused.add(mf.group("experts"))
                continue
            m = self._EXPERT_WEIGHT_RE.match(model_name)
            if m is not None:
                fused.add(m.group("experts"))
                continue
            set_module_tensor_to_device(self.model, model_name, "meta")
        shapes = self._fused_expert_shapes()
        for experts_prefix in fused:
            mod = self._module_by_name(experts_prefix)
            for pname, shape in shapes.items():
                if getattr(mod, pname, None) is not None:
                    setattr(mod, pname, torch.nn.Parameter(
                        torch.empty(shape, device="meta", dtype=self.dtype),
                        requires_grad=False,
                    ))
        torch.cuda.empty_cache()

    def _resolve_text_model(self):
        """Detect whether model is CausalLM (text-only) or ConditionalGeneration (multimodal)."""
        if hasattr(self.model, 'model'):
            inner = self.model.model
            if hasattr(inner, 'language_model'):
                return inner.language_model
            if hasattr(inner, 'text_model'):
                return inner.text_model
            return inner
        return self.model

    def _resolve_layer_prefix(self):
        """Determine the layer prefix based on model structure."""
        if hasattr(self.model.model, 'language_model'):
            return "model.language_model.layers"
        if hasattr(self.model.model, 'text_model'):
            return "model.text_model.layers"
        return "model.layers"

    def _get_layers(self):
        tm = self._text_model
        if hasattr(tm, 'layers'):
            return tm.layers
        if hasattr(tm, 'model') and hasattr(tm.model, 'layers'):
            return tm.model.layers
        raise AttributeError("Cannot locate decoder layers in the text model")

    def _is_linear_attention_layer(self, layer_idx):
        if layer_idx < len(self.layer_types):
            return self.layer_types[layer_idx] == "linear_attention"
        return False

    def _layer_prefixes(self, layer_name):
        layer_idx = int(layer_name.split('.')[-1])
        base = f"{self.layer_prefix}.{layer_idx}"

        attn_prefix = f"{base}.linear_attn" if self._is_linear_attention_layer(layer_idx) else f"{base}.self_attn"

        non_mlp = [
            f"{base}.input_layernorm",
            attn_prefix,
            f"{base}.mlp.gate",
            f"{base}.mlp.shared_expert",
            f"{base}.mlp.shared_expert_gate",
            f"{base}.post_attention_layernorm"
        ]
        mlp = [
            f"{base}.mlp.experts.{e}"
            for e in range(self.num_experts)
            if self.sharder.owner(e) == self.rank
        ]
        # Pre-fused checkpoints keep every expert in two tensors, so there is nothing
        # to shard at READ time -- ask for both and let _write_fused_expert_rows keep
        # only the owned rows. On a per-expert checkpoint these two names simply match
        # nothing, so adding them is safe for both layouts.
        mlp += [f"{base}.mlp.experts.gate_up_proj", f"{base}.mlp.experts.down_proj"]
        return {"non_mlp": non_mlp, "mlp": mlp}

    def _get_embed_tokens(self):
        tm = self._text_model
        if hasattr(tm, 'embed_tokens'):
            return tm.embed_tokens
        if hasattr(tm, 'model') and hasattr(tm.model, 'embed_tokens'):
            return tm.model.embed_tokens
        raise AttributeError("Cannot locate embed_tokens")

    def _get_norm(self):
        tm = self._text_model
        if hasattr(tm, 'norm'):
            return tm.norm
        if hasattr(tm, 'model') and hasattr(tm.model, 'norm'):
            return tm.model.norm
        raise AttributeError("Cannot locate norm layer")

    def _embed_prefix(self):
        if hasattr(self.model.model, 'language_model'):
            return "model.language_model.embed_tokens"
        if hasattr(self.model.model, 'text_model'):
            return "model.text_model.embed_tokens"
        return "model.embed_tokens"

    def _norm_prefix(self):
        if hasattr(self.model.model, 'language_model'):
            return "model.language_model.norm"
        if hasattr(self.model.model, 'text_model'):
            return "model.text_model.norm"
        return "model.norm"

    def move_embed_to(self, device):
        names = self._names_from_ckpt([self._embed_prefix()])
        if str(device).startswith("cuda"):
            self._set_tensors(names)
        else:
            self._offload_names_to_meta(names)
        # rotary_emb buffers (inv_freq, original_inv_freq) are computed at
        # init time, not stored in the checkpoint.  They must be co-located
        # with the tensors that flow through the forward pass.
        tm = self._text_model
        rope = getattr(tm, 'rotary_emb', None)
        if rope is None and hasattr(tm, 'model'):
            rope = getattr(tm.model, 'rotary_emb', None)
        if rope is not None:
            target = self.device if str(device).startswith("cuda") else "cpu"
            rope.to(target)

    def move_output_heads_to(self, device):
        names = []
        names += self._names_from_ckpt(self._norm_prefix())
        names += self._names_from_ckpt("lm_head")
        if str(device).startswith("cuda"):
            self._set_tensors(names)
        else:
            self._offload_names_to_meta(names)

    def _compute_position_embeddings(self, batch_size, seq_len, device):
        """Compute fresh (cos, sin) position embeddings for the given shape.

        Qwen3.5 uses MRoPE with 4 position-id axes (text, temporal, height, width).
        For text-only inference the last 3 are identical to the first, and the
        ``rotary_emb`` forward expects a ``(3, batch, seq)`` position_ids tensor
        (the text axis is split off earlier in the TextModel forward).
        """
        tm = self._text_model
        rope = getattr(tm, 'rotary_emb', None)
        if rope is None and hasattr(tm, 'model'):
            rope = getattr(tm.model, 'rotary_emb', None)
        if rope is None:
            return None
        ## rotary_emb keeps its inv_freq buffer wherever the model was built (CPU), and
        ## nothing else moves it. Move it to `device` so rope's own
        ## `inv_freq_expanded @ position_ids_expanded` does not straddle devices.
        inv_freq = getattr(rope, "inv_freq", None)
        if inv_freq is not None and inv_freq.device != torch.device(device):
            rope = rope.to(device)
        cache_pos = torch.arange(seq_len, device=device)
        # shape (3, batch, seq) — the 3 non-text MRoPE axes
        position_ids = cache_pos.view(1, 1, -1).expand(3, batch_size, -1)
        dummy = torch.empty(batch_size, seq_len, 1, device=device)
        return rope(dummy, position_ids)

    def _build_layer_inputs(self, batch_size):
        inputs = dict(self.kwargs)
        if self._attention_mask_1 is not None:
            inputs["attention_mask"] = self._attention_mask_1.to(self.device).expand(batch_size, -1, -1, -1)
        else:
            inputs["attention_mask"] = None
        cached_pe = inputs.get("position_embeddings")
        seq_len = cached_pe[0].shape[1] if cached_pe is not None else None
        if cached_pe is not None and cached_pe[0].shape[0] != batch_size:
            inputs["position_embeddings"] = self._compute_position_embeddings(
                batch_size, seq_len, self.device
            )
        ## position_ids is batch-shaped too ((3, batch, seq) for MRoPE), so a batch
        ## mismatch would reach flash-attention's packed-sequence check. Rebuild it the
        ## same way _compute_position_embeddings does rather than trusting the captured
        ## copy.
        cached_pids = inputs.get("position_ids")
        if cached_pids is not None and cached_pids.dim() >= 2 and cached_pids.shape[-2] != batch_size:
            n_axes = cached_pids.shape[0] if cached_pids.dim() == 3 else 1
            pids_seq = cached_pids.shape[-1] if seq_len is None else seq_len
            base = torch.arange(pids_seq, device=self.device).view(1, 1, -1)
            ## .contiguous(): expand() leaves stride-0 dims, and the attention kernels
            ## index position_ids directly rather than going through a materializing op.
            inputs["position_ids"] = base.expand(n_axes, batch_size, -1).contiguous()
        if os.environ.get("MOESQ_DUMP_LAYER_INPUTS") == "1" and self.rank == 0:
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    desc = f"tensor{tuple(v.shape)} {v.dtype} {v.device}"
                elif isinstance(v, (tuple, list)):
                    desc = "/".join(f"tensor{tuple(t.shape)}" if torch.is_tensor(t) else type(t).__name__ for t in v)
                else:
                    desc = repr(v)
                print(f"[layer-inputs b={batch_size}] {k}: {desc}", flush=True)
        return inputs

    def _run_attention(self, layer, layer_idx, hidden_states, additional_inputs):
        if self._is_linear_attention_layer(layer_idx):
            out = layer.linear_attn(hidden_states)
        else:
            out = layer.self_attn(hidden_states, **additional_inputs)
        if isinstance(out, tuple):
            return out[0]
        return out

    # The five hooks below are the only places that know how a decoder layer wires its
    # residual stream around attention and the MoE block. Subclasses for architectures
    # with a different residual (e.g. qwen4_exp's hyper-connections) override just these.

    def _attention_residual(self, layer, layer_idx, x, layer_inputs, input_ids=None):
        """Layer input -> the residual stream entering the MoE block (``mlp_input``).

        ``input_ids`` is the batch's token ids ([B, T], any device); only architectures
        with a per-layer token embedding read it.
        """
        hidden_states = layer.input_layernorm(x)
        return x + self._run_attention(layer, layer_idx, hidden_states, layer_inputs)

    def _moe_hidden(self, layer, mlp_input):
        """``mlp_input`` -> what the router and the routed and shared experts see."""
        return layer.post_attention_layernorm(mlp_input)

    def _moe_combine(self, layer, mlp_input, routed, hidden):
        """Layer output from the routed-expert sum ``routed`` [B, T, H] and ``hidden``."""
        B, T, H = routed.shape
        shared_out = layer.mlp.shared_expert(hidden.reshape(-1, H))
        shared_gate = F.sigmoid(layer.mlp.shared_expert_gate(hidden.reshape(-1, H)))
        shared_out = (shared_out * shared_gate).view(B, T, H)
        return routed + shared_out + mlp_input

    def _embed_input(self, input_ids):
        """Token ids -> the first layer's input."""
        return self._get_embed_tokens()(input_ids)

    def _output_hidden(self, x):
        """Last layer's output -> what ``lm_head`` sees."""
        return self._get_norm()(x)

    def _batch_input_ids(self, data_all, start, end):
        """Token ids of samples ``[start, end)`` of an activation cache, when needed."""
        return None

    def get_mlp_input(self, batch, input_ids=None):
        current_layer = self.get_layer_module(self.current_layer_idx)
        additional_layer_inputs = self._build_layer_inputs(batch.shape[0])
        return self._attention_residual(current_layer, self.current_layer_idx, batch,
                                        additional_layer_inputs, input_ids)

    @torch.no_grad()
    def get_layer_activations(self, data_all):
        current_layer = self.get_layer_module(self.current_layer_idx)
        num_samples = data_all['input'].shape[0]
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, num_samples)
            x = data_all['input'][start_idx:end_idx].to(self.device, non_blocking=True)

            ## Per batch, via _build_layer_inputs: self.kwargs caches
            ## position_embeddings at the batch size they were captured with, and the
            ## final batch here is short whenever num_samples % batch_size != 0. Only
            ## full_attention layers read them, so a mismatch is invisible until the
            ## first such layer and then fails inside apply_rotary_pos_emb.
            additional_layer_inputs = self._build_layer_inputs(x.shape[0])
            mlp_input = self._attention_residual(
                current_layer, self.current_layer_idx, x, additional_layer_inputs,
                self._batch_input_ids(data_all, start_idx, end_idx))

            out = self.run_expert_parallel(mlp_input)
            data_all['input'][start_idx:end_idx] = out.detach().cpu()

    @torch.no_grad()
    def get_mlp_output(self, mlp_input_batch):
        return self.run_expert_parallel(mlp_input_batch)

    def _dispatch_tokens(self, mlp_input_batch):
        """Route tokens to expert-owning ranks via all-to-all."""
        layer = self.get_layer_module(self.current_layer_idx)
        device = self.device
        pg = dist.group.WORLD

        hidden = self._moe_hidden(layer, mlp_input_batch)
        B, T, H = hidden.shape
        x_flat = hidden.reshape(B * T, H)

        _, routing_weights, selected_experts = layer.mlp.gate(hidden.reshape(-1, H))

        top_k = selected_experts.shape[-1]
        tok_idx_flat = torch.arange(B * T, device=device, dtype=torch.long).repeat_interleave(top_k)
        eid_flat = selected_experts.reshape(-1).to(torch.long)
        w_flat = routing_weights.reshape(-1).to(self.dtype)

        owner_lut = self._owner_lut.to(device)
        owners_flat = owner_lut[eid_flat]
        perm = torch.argsort(owners_flat, stable=True)
        owners_flat = owners_flat.index_select(0, perm)
        send_idx_flat = tok_idx_flat.index_select(0, perm)
        send_eid_flat = eid_flat.index_select(0, perm)
        send_w_flat = w_flat.index_select(0, perm)
        send_x_flat = x_flat.index_select(0, send_idx_flat)

        world_size = self.world_size
        in_sizes_tensor = torch.bincount(owners_flat, minlength=world_size).to(torch.long)
        all_sizes = [torch.empty_like(in_sizes_tensor) for _ in range(world_size)]
        dist.all_gather(all_sizes, in_sizes_tensor, group=pg)
        recv_sizes = torch.stack(all_sizes)[:, self.rank]
        out_split_sizes = recv_sizes.tolist()
        in_split_sizes = in_sizes_tensor.tolist()

        xin = AllToAllTokens.apply(send_x_flat, out_split_sizes, in_split_sizes, pg)
        win = AllToAllTokens.apply(send_w_flat.unsqueeze(1), out_split_sizes, in_split_sizes, pg).to(self.dtype)
        eids = AllToAllTokens.apply(send_eid_flat.unsqueeze(1), out_split_sizes, in_split_sizes, pg).squeeze(1)

        return x_flat, hidden, send_idx_flat, in_split_sizes, out_split_sizes, xin, win, eids, B, T, H

    def _batched_expert_forward(self, xin, eids, layer=None, compressed_weights=None, fake_act_quant=False):
        """Process local experts using 3D fused tensor indexing."""
        if layer is None:
            layer = self.get_layer_module(self.current_layer_idx)
        layer_key = self.get_current_layer()
        experts_module = layer.mlp.experts

        eids_long = eids.to(torch.long)
        unique_eids, inverse, counts = torch.unique(eids_long, sorted=True, return_inverse=True, return_counts=True)

        sort_idx = torch.argsort(inverse, stable=True)
        sorted_x = xin.index_select(0, sort_idx)

        out_buf = torch.empty_like(sorted_x)

        def _q(x):
            # Activation scale block must align with the weight's dense scale group
            # (= self.groupsize) so the sparse NVFP4 microscale matmul factors per block.
            return fake_quantize_activation_nvfp4(x, groupsize=self.groupsize) if fake_act_quant else x

        offset = 0
        for i, eid_val in enumerate(unique_eids.tolist()):
            n = counts[i].item()
            inp_e = sorted_x[offset:offset + n]

            if compressed_weights is not None:
                key = f"{layer_key}.mlp.experts.{eid_val}"
                qw = compressed_weights[key]
                gate_up_w = qw.get("gate_up_proj")
                down_w = qw.get("down_proj")
                if gate_up_w is None:
                    gate_w = qw["gate_proj"][0] if isinstance(qw["gate_proj"], tuple) else qw["gate_proj"]
                    up_w = qw["up_proj"][0] if isinstance(qw["up_proj"], tuple) else qw["up_proj"]
                    gate_out = F.linear(_q(inp_e), gate_w)
                    up_out = F.linear(_q(inp_e), up_w)
                else:
                    if isinstance(gate_up_w, tuple):
                        gate_up_w = gate_up_w[0]
                    if isinstance(down_w, tuple):
                        down_w = down_w[0]
                    gu = F.linear(_q(inp_e), gate_up_w)
                    gate_out, up_out = gu.chunk(2, dim=-1)
                hidden = F.silu(gate_out) * up_out
                if down_w is not None:
                    out_e = F.linear(_q(hidden), down_w if not isinstance(down_w, tuple) else down_w[0])
                else:
                    out_e = F.linear(_q(hidden), qw["down_proj"][0] if isinstance(qw["down_proj"], tuple) else qw["down_proj"])
            else:
                row = self._local_expert_index(eid_val)
                gate_up_w = experts_module.gate_up_proj[row]
                down_w = experts_module.down_proj[row]
                gu = F.linear(inp_e, gate_up_w)
                gate_out, up_out = gu.chunk(2, dim=-1)
                hidden = F.silu(gate_out) * up_out
                out_e = F.linear(hidden, down_w)

            out_buf[offset:offset + n] = out_e
            offset += n

        unsort_idx = torch.argsort(sort_idx)
        return out_buf.index_select(0, unsort_idx)

    def run_expert_parallel(self, mlp_input_batch, compressed_weights=None):
        pg = dist.group.WORLD

        x_flat, hidden, send_idx_flat, in_split_sizes, out_split_sizes, xin, win, eids, B, T, H = \
            self._dispatch_tokens(mlp_input_batch)

        out_local = self._batched_expert_forward(xin, eids, compressed_weights=compressed_weights)
        xin = out_local * win

        returned = AllToAllTokens.apply(xin, in_split_sizes, out_split_sizes, pg)

        y_flat = x_flat.new_zeros(x_flat.shape)
        y_flat.index_add_(0, send_idx_flat, returned)
        y = y_flat.view(B, T, H)

        layer = self.get_layer_module(self.current_layer_idx)
        return self._moe_combine(layer, mlp_input_batch, y, hidden)

    def calculate_mse(self, mlp_input_batch, compressed_weights, self_attn=False, validation=False, accumulation_steps=1, fake_act_quant=False):
        with torch.no_grad():
            _, _, _, _, _, xin, win, eids, _, _, _ = self._dispatch_tokens(mlp_input_batch)
            out_fp = self._batched_expert_forward(xin, eids, compressed_weights=None)
        out_q = self._batched_expert_forward(xin, eids, compressed_weights=compressed_weights, fake_act_quant=fake_act_quant)

        # win: each received (token, expert) row's router weight, for the gate-weighted loss
        total_mse = self._expert_recon_loss(out_q, out_fp, win)
        if not validation:
            (total_mse / accumulation_steps).backward()
        return total_mse.item()

    def _collect_expert_activations(self, gpt_all, config):
        """Collect per-expert input activations for GPTQ calibration.

        For 3D fused expert tensors we cannot register hooks on nn.Linear
        submodules (they don't exist). Instead, run the full expert-parallel
        forward and collect the input tensors that arrive at each owned expert
        after the all-to-all dispatch.
        """
        layer_idx = self.current_layer_idx
        layer = self.get_layer_module(layer_idx)

        expert_inputs = {e: [] for e in range(self.num_experts) if self.sharder.owner(e) == self.rank}

        ## Batch at self.batch_size -- the size self.kwargs (position_embeddings /
        ## position_ids) was captured at, as get_layer_activations does -- so the cached
        ## tensors are used verbatim instead of being reconstructed for another batch
        ## size. Same tokens either way, so H is unchanged.
        n_local = config.init.nsamples // self.world_size
        step = max(1, self.batch_size)

        with torch.no_grad():
            for start in range(0, n_local, step):
                x = gpt_all['input'][start:start + step].to(self.device, non_blocking=True)
                additional_layer_inputs = self._build_layer_inputs(x.shape[0])
                x = self._attention_residual(
                    layer, layer_idx, x, additional_layer_inputs,
                    self._batch_input_ids(gpt_all, start, start + x.shape[0]))

                hidden = self._moe_hidden(layer, x)
                B, T, H = hidden.shape
                x_flat = hidden.reshape(B * T, H)

                _, routing_weights, selected_experts = layer.mlp.gate(x_flat)
                top_k = selected_experts.shape[-1]
                pg = dist.group.WORLD

                tok_idx_flat = torch.arange(B * T, device=self.device, dtype=torch.long).repeat_interleave(top_k)
                eid_flat = selected_experts.reshape(-1).to(torch.long)

                owner_lut = self._owner_lut.to(self.device)
                owners_flat = owner_lut[eid_flat]
                perm = torch.argsort(owners_flat, stable=True)
                owners_flat = owners_flat.index_select(0, perm)
                send_idx_flat = tok_idx_flat.index_select(0, perm)
                send_eid_flat = eid_flat.index_select(0, perm)
                send_x_flat = x_flat.index_select(0, send_idx_flat)

                in_sizes_tensor = torch.bincount(owners_flat, minlength=self.world_size).to(torch.long)
                all_sizes = [torch.empty_like(in_sizes_tensor) for _ in range(self.world_size)]
                dist.all_gather(all_sizes, in_sizes_tensor, group=pg)
                recv_sizes = torch.stack(all_sizes)[:, self.rank]
                out_split_sizes = recv_sizes.tolist()
                in_split_sizes = in_sizes_tensor.tolist()

                xin = AllToAllTokens.apply(send_x_flat, out_split_sizes, in_split_sizes, pg)
                eids = AllToAllTokens.apply(send_eid_flat.unsqueeze(1), out_split_sizes, in_split_sizes, pg).squeeze(1)

                for eid_val in torch.unique(eids).tolist():
                    eid_val = int(eid_val)
                    if eid_val in expert_inputs:
                        mask = (eids == eid_val)
                        expert_inputs[eid_val].append(xin[mask].detach())

        ## Fill a preallocated buffer and release each chunk as it is copied. torch.cat
        ## would hold the whole chunk list AND its concatenation at the same time, which
        ## doubles the peak for exactly the hot experts that make this expensive.
        hidden_dim = layer.mlp.experts.gate_up_proj.shape[-1]
        for e in list(expert_inputs.keys()):
            chunks = expert_inputs[e]
            if not chunks:
                expert_inputs[e] = torch.empty(0, hidden_dim,
                                               device=self.device, dtype=self.dtype)
                continue
            total_tokens = sum(c.shape[0] for c in chunks)
            buf = torch.empty(total_tokens, hidden_dim,
                              device=self.device, dtype=chunks[0].dtype)
            offset = 0
            for i in range(len(chunks)):
                c = chunks[i]
                buf[offset:offset + c.shape[0]].copy_(c)
                offset += c.shape[0]
                chunks[i] = None
            expert_inputs[e] = buf

        return expert_inputs

    def get_layer_initialization(self, trainer, gpt_all, config, logging):
        if logging is not None:
            logging = logging.logger
        layer_idx = self.current_layer_idx
        layer = self.get_layer_module(layer_idx)
        rank = self.rank
        self.configure_quantization_from_config(config)

        experts_module = layer.mlp.experts
        owned_experts = [e for e in range(self.num_experts) if self.sharder.owner(e) == rank]

        expert_inputs = self._collect_expert_activations(gpt_all, config)

        subset = {}
        for e in owned_experts:
            row = self._local_expert_index(e)
            gate_up_w = experts_module.gate_up_proj[row]
            down_w = experts_module.down_proj[row]

            intermediate_size = gate_up_w.shape[0] // 2
            hidden_size = gate_up_w.shape[1]

            gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False, device=self.device, dtype=self.dtype)
            gate_proj.weight.data = gate_up_w[:intermediate_size]

            up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False, device=self.device, dtype=self.dtype)
            up_proj.weight.data = gate_up_w[intermediate_size:]

            down_proj_mod = torch.nn.Linear(intermediate_size, hidden_size, bias=False, device=self.device, dtype=self.dtype)
            down_proj_mod.weight.data = down_w

            base_prefix = f"{self.get_current_layer()}.mlp.experts.{e}"
            subset[f"{base_prefix}.gate_proj"] = gate_proj
            subset[f"{base_prefix}.up_proj"] = up_proj
            subset[f"{base_prefix}.down_proj"] = down_proj_mod

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
            del subset, expert_inputs
            torch.cuda.empty_cache()
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

        ## Release each expert's activations as it is consumed rather than at the end of
        ## the loop: expert_inputs holds every owned expert at once (num_experts/world_size
        ## of them), which dominates resident memory before a single Hessian is touched.
        chunk_tokens = config.init.hessian_chunk_tokens
        ## no_grad here is load-bearing, not hygiene: without it these forwards build an
        ## autograd graph that keeps every expert's activations reachable no matter what
        ## the dels below drop.
        with torch.no_grad():
            for e in owned_experts:
                inp_e = expert_inputs.pop(e)
                if inp_e.numel() == 0:
                    del inp_e
                    continue
                base_prefix = f"{self.get_current_layer()}.mlp.experts.{e}"
                gate_key = f"{base_prefix}.gate_proj"
                down_key = f"{base_prefix}.down_proj"

                gate_mod = subset[gate_key]
                up_mod = subset[f"{base_prefix}.up_proj"]

                gate_out = gate_mod(inp_e)
                gpts[gate_key].add_batch(inp_e, None, chunk_tokens=chunk_tokens)

                up_out = up_mod(inp_e)
                if init_method == "jsq":
                    ## JSQ's range term depends on up's OWN output, so up cannot inherit
                    ## gate's statistics; range only (h=False) -- up's H is overwritten
                    ## by gate's before its fasterquant in the loop below anyway.
                    gpts[f"{base_prefix}.up_proj"].add_batch(inp_e, None, chunk_tokens=chunk_tokens, h=False)
                intermediate = F.silu(gate_out) * up_out
                ## inp_e/gate_out/up_out are dead once intermediate exists; drop them before
                ## the down_proj pass instead of holding them to the end of the iteration.
                del gate_out, up_out, inp_e

                ## No down_mod(intermediate) here: add_batch ignores its `out` argument, so
                ## materializing down_out would cost a [tokens, hidden] tensor and a full
                ## matmul per expert for nothing.
                gpts[down_key].add_batch(intermediate, None, chunk_tokens=chunk_tokens)
                del intermediate

        del expert_inputs
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        ## Every owned expert was wired above (direct add_batch or a forward hook), so a
        ## handle that received nothing is an expert with ZERO routed tokens -- state that
        ## explicitly rather than leaving fasterquant to infer it from a missing H. jsq's
        ## range guard cannot otherwise tell an unrouted expert from an unwired dispatch,
        ## and would raise on the former. up_proj under jsq keeps amax with nsamples 0
        ## (h=False), so the amax check below is what stops it being marked.
        ##
        ## This marks ONLY experts that got no tokens. An expert fed identically-zero
        ## activations has nsamples > 0 and is deliberately not marked -- that case is
        ## handled inside the initializers (obr.py's dead-diagonal repair, and jsq's
        ## clamped range normalization), not here.
        for _mn, _mh in gpts.items():
            if (getattr(_mh, "nsamples", 0) == 0 and getattr(_mh, "amax", None) is None
                    and hasattr(_mh, "mark_no_tokens")):
                _mh.mark_no_tokens()

        gptq_losses = []
        for name in gpts:
            if "up_proj" in name:
                continue
            Q, scales = gpts[name].fasterquant(
                logging,
                percdamp=config.init.percdamp,
                blocksize=config.init.blocksize,
                groupsize=config.compression.groupsize,
                static_groups=config.init.static_groups,
                prunen=config.compression.prunen,
                prunem=config.compression.prunem
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
            if "gate_proj" in name:
                base = name[: -len(".gate_proj")]
                new_name = f"{base}.up_proj"
                gpts[new_name].H = gpts[name].H
                gpts[new_name].dead = gpts[name].dead
                Q, scales = gpts[new_name].fasterquant(
                    logging,
                    percdamp=config.init.percdamp,
                    blocksize=config.init.blocksize,
                    groupsize=config.compression.groupsize,
                    static_groups=config.init.static_groups,
                    calculate_cholesky=False,
                    prunen=config.compression.prunen,
                    prunem=config.compression.prunem
                )
                if hasattr(gpts[new_name], 'last_gptq_loss'):
                    gptq_losses.append(gpts[new_name].last_gptq_loss)
                if scales is not None and refine_enabled:
                    trainer.setup_layer_training(
                        new_name,
                        Q,
                        scales,
                        init_dense_weight=gpts[new_name].last_init_dense_weight,
                        init_support_mask=gpts[new_name].last_init_support_mask,
                    )
                else:
                    self.update_compressed_weights(new_name, (Q, scales) if scales is not None else Q)
                gpts[name].free()
                gpts[new_name].free()
            else:
                gpts[name].free()

        if gptq_losses:
            trainer.gptq_avg_loss = sum(gptq_losses) / len(gptq_losses)

        del subset
        torch.cuda.empty_cache()

        dist.barrier()

    def _load_layer_for_eval(self, layer_idx, read_from_disk):
        layer_name = f"{self.layer_prefix}.{layer_idx}"
        if layer_idx <= read_from_disk:
            self.load_from_disc(layer_name)
        else:
            self.move_layer_to_gpu(layer_name)

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

        pad_token_id = self.model.config.pad_token_id

        if pad_token_id is not None:
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)
        else:
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

        self.model.eval()

        self.move_embed_to(self.device)
        self.move_output_heads_to(self.device)

        input_ids_cpu_list = []
        activations_cpu_list = []

        for batch in testloader:
            ids_cpu = batch["input_ids"].to("cpu", non_blocking=True).pin_memory()
            input_ids_cpu_list.append(ids_cpu)
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
                ids_cpu = input_ids_cpu_list[b]

                x_cpu = activations_cpu_list[b]
                if x_cpu is None:
                    ids = ids_cpu.to(self.device, non_blocking=True)
                    x = self._embed_input(ids)
                else:
                    x = x_cpu.to(self.device, non_blocking=True)

                additional_layer_inputs = self._build_layer_inputs(x.shape[0])
                x = self._attention_residual(layer, i, x, additional_layer_inputs, ids_cpu)

                x = self.run_expert_parallel(x)

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
            ids_cpu = input_ids_cpu_list[b]
            x_cpu = activations_cpu_list[b]

            input_ids = ids_cpu.to(self.device, non_blocking=True)
            x = x_cpu.to(self.device, non_blocking=True)

            x = self._output_hidden(x)
            logits = self.model.lm_head(x)

            logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]

            nll = loss_fn(logits.permute(0, 2, 1), shift_labels).float()
            mask = shift_labels != loss_fn.ignore_index
            nll = (nll * mask).sum(dim=1)
            tok = mask.sum(dim=1)

            local_nll_sum += nll.sum()
            local_tok_cnt += tok.sum()

            del input_ids, x, x_cpu, logits, shift_labels, nll, tok

        self.move_embed_to("meta")
        self.move_output_heads_to("meta")

        self.current_layer_idx = idx_copy

        dist.all_reduce(local_nll_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_tok_cnt, op=dist.ReduceOp.SUM)

        mean_nll = (local_nll_sum / local_tok_cnt).item()
        ppl = math.exp(mean_nll)

        torch.cuda.synchronize(self.device)
        gc.collect()
        torch.cuda.empty_cache()
        return ppl

    def get_layer_module(self, idx):
        return self._get_layers()[idx]

    _EXPERT_KEY_RE = re.compile(
        r"\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$"
    )

    def update_compressed_weights(self, layer_name, compressed_weights):
        m = self._EXPERT_KEY_RE.search(layer_name)
        if m:
            expert_idx = int(m.group(1))
            proj = m.group(2)
            layer_prefix = layer_name[:m.start()]
            layer_mod = self.model.get_submodule(layer_prefix)
            experts = layer_mod.mlp.experts
            if isinstance(compressed_weights, tuple):
                qw, scales, global_scale, mask = self._split_weight_payload(compressed_weights)
                self.temp_weights[f"{layer_name}.Q"] = qw
                self.temp_weights[f"{layer_name}.scale"] = scales
                if global_scale is not None:
                    self.temp_weights[f"{layer_name}.global_scale"] = global_scale
                if mask is not None:
                    self.temp_weights[f"{layer_name}.mask"] = mask
            else:
                qw = compressed_weights
            with torch.no_grad():
                if proj == "gate_proj":
                    intermediate = experts.gate_up_proj.shape[1] // 2
                    experts.gate_up_proj.data[self._local_expert_index(expert_idx), :intermediate] = qw.to(experts.gate_up_proj.device)
                elif proj == "up_proj":
                    intermediate = experts.gate_up_proj.shape[1] // 2
                    experts.gate_up_proj.data[self._local_expert_index(expert_idx), intermediate:] = qw.to(experts.gate_up_proj.device)
                elif proj == "down_proj":
                    experts.down_proj.data[self._local_expert_index(expert_idx)] = qw.to(experts.down_proj.device)
        else:
            layer = self._get_layer_by_name(layer_name)
            if isinstance(compressed_weights, tuple):
                Q, scales, global_scale, mask = self._split_weight_payload(compressed_weights)
                self.temp_weights[layer_name] = layer.weight.data
                self.temp_weights[f"{layer_name}.scale"] = scales
                if global_scale is not None:
                    self.temp_weights[f"{layer_name}.global_scale"] = global_scale
                if mask is not None:
                    self.temp_weights[f"{layer_name}.mask"] = mask
                with torch.no_grad():
                    layer.weight.data = Q.to(layer.weight.device).to(layer.weight.dtype)
            else:
                with torch.no_grad():
                    layer.weight.data = compressed_weights.to(layer.weight.dtype).to(layer.weight.device)

    def save_prefixes_to_disc(self, prefixes, config=None):
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        if not prefixes:
            return

        os.makedirs(self.save_dir, exist_ok=True)

        for pfx in prefixes:
            module = self.model.get_submodule(pfx)

            sd = module.state_dict(keep_vars=True)
            to_save = {}

            for local_name, tensor in sd.items():
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                    if ".expert" in pfx or pfx.endswith(".experts"):
                        weight = tensor.detach().cpu()
                        quantization_args = self.configure_quantization_from_config()

                        flat = weight.flatten()
                        N = flat.numel()
                        num_blocks = N // self.groupsize
                        flat = flat[:num_blocks * self.groupsize]
                        blocks = flat.view(-1, self.groupsize)
                        abs_blocks = blocks.abs()
                        nonzero_mask = abs_blocks > 0

                        abs_blocks_masked = abs_blocks.clone()
                        abs_blocks_masked[~nonzero_mask] = float('inf')

                        scale, _ = abs_blocks_masked.min(dim=1)
                        scale[scale == float('inf')] = 0.0
                        scale = scale.view(weight.shape[0], -1)

                        compressed = self.compressor.compress_weight(
                            weight=weight,
                            scale=scale,
                            quantization_args=quantization_args
                        )
                        new_name = local_name[:-len(".weight")] if local_name.endswith(".weight") else local_name
                        base = f"{pfx}.{new_name}"
                        to_save[base + ".weight_packed"] = compressed["weight_packed"]
                        to_save[base + ".weight_scale"]  = scale
                        to_save[base + ".weight_shape"]  = compressed["weight_shape"]
                    else:
                        to_save[f"{pfx}.{local_name}"] = tensor.detach().cpu()

            safe = pfx.replace('.', '_')
            path = os.path.join(self.save_dir, f"{safe}.safetensors")
            safe_save_file(to_save, path)

    _EXPERT_PROJ_RE = re.compile(r"\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)")

    def _write_dequantized_expert_weight(self, base_key, W_deq):
        em = self._EXPERT_PROJ_RE.search(base_key)
        if em is None:
            set_module_tensor_to_device(
                self.model, f"{base_key}.weight",
                self.device, value=W_deq, dtype=self.dtype,
            )
            return
        expert_idx = int(em.group(1))
        proj = em.group(2)
        layer_prefix = base_key[:em.start()]
        experts = self.model.get_submodule(layer_prefix).mlp.experts
        # On --resume a finished layer is rebuilt from its shards alone -- _set_tensors
        # never ran for it -- so the fused params are still on meta and must be
        # allocated before slicing into them. No-op when the layer is already resident.
        self._ensure_fused_expert_params(f"{layer_prefix}.mlp.experts")
        row = self._local_expert_index(expert_idx)
        with torch.no_grad():
            W_deq = W_deq.to(self.device).to(self.dtype)
            if proj == "gate_proj":
                intermediate = experts.gate_up_proj.shape[1] // 2
                experts.gate_up_proj.data[row, :intermediate] = W_deq
            elif proj == "up_proj":
                intermediate = experts.gate_up_proj.shape[1] // 2
                experts.gate_up_proj.data[row, intermediate:] = W_deq
            else:
                experts.down_proj.data[row] = W_deq

    def load_from_disc(self, layer_name):
        """Load layer from disk, handling per-expert files for 3D fused tensors.

        The trainer saves per-expert files (model_layers_X_mlp_experts_Y.safetensors)
        with keys like ``...experts.Y.gate_proj.weight_packed`` (NVFP4 / pack-quantized)
        or ``...experts.Y.gate_proj.weight`` as int32 codes plus ``.weight_scale`` (GSQ,
        Humming uint2). These are decompressed and written back into the correct slices
        of the 3D fused ``gate_up_proj`` / ``down_proj`` parameters.
        """
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

        expert_re = self._EXPERT_PROJ_RE

        for p, path in files.items():
            if not os.path.isfile(path):
                continue
            tensors = safe_load_file(path, device="cpu")
            for name in tensors.keys():
                # weight_global_scale must be skipped too: without it the NVFP4
                # triple's third tensor falls through to the expert branch below
                # and gets written as if it were a weight.
                if (name.endswith(".weight_shape") or name.endswith(".weight_scale")
                        or name.endswith(".weight_global_scale") or name.endswith("inv_freq")):
                    continue
                if name.endswith(".weight_packed"):
                    base_key = name[: -len(".weight_packed")]
                    # Two save formats reach this path. The NVFP4 arm writes
                    # weight_packed/weight_scale/weight_global_scale (no shape);
                    # the int pack-quantized arm writes weight_shape instead.
                    # base.py already branches this way -- keep them in step.
                    if f"{base_key}.weight_global_scale" in tensors:
                        compressed_data = {
                            "weight_packed": tensors[f"{base_key}.weight_packed"],
                            "weight_scale": tensors[f"{base_key}.weight_scale"],
                            "weight_global_scale": tensors[f"{base_key}.weight_global_scale"],
                        }
                        W_deq = NVFP4PackedCompressor().decompress_weight(
                            compressed_data,
                            quantization_args,
                        )
                    else:
                        compressed_data = {
                            "weight_packed": tensors[f"{base_key}.weight_packed"],
                            "weight_scale": tensors[f"{base_key}.weight_scale"],
                            "weight_shape": tensors[f"{base_key}.weight_shape"],
                        }
                        W_deq = self.compressor.decompress_weight(
                            compressed_data,
                            quantization_args,
                        )

                    self._write_dequantized_expert_weight(base_key, W_deq)
                    continue

                if (
                    getattr(self, "is_gsq", False)
                    and name.endswith(".weight")
                    and tensors[name].dtype in (torch.int32, torch.int64)
                    and f"{name}_scale" in tensors
                ):
                    # GSQ shards reuse the plain `.weight` key for Humming's packed uint2
                    # codes. The per-expert fallthrough below never matches that key
                    # (its leaf test is "gate_up_proj"/"down_proj"), so it is handled here.
                    W_deq = dequantize_gsq_packed(
                        tensors[name], tensors[f"{name}_scale"], getattr(self, "_init_wbits", 2)
                    )
                    self._write_dequantized_expert_weight(name[: -len(".weight")], W_deq)
                    continue

                em = expert_re.search(name)
                if em:
                    expert_idx = int(em.group(1))
                    layer_prefix = name[:em.start()]
                    layer_mod = self.model.get_submodule(layer_prefix)
                    experts = layer_mod.mlp.experts
                    self._ensure_fused_expert_params(f"{layer_prefix}.mlp.experts")
                    local_key = name[em.start() + 1:]
                    param_parts = local_key.split(".")
                    param_name = param_parts[-1]
                    with torch.no_grad():
                        t = tensors[name].to(self.device).to(self.dtype)
                        if "gate_up_proj" in param_name:
                            experts.gate_up_proj.data[self._local_expert_index(expert_idx)] = t
                        elif "down_proj" in param_name:
                            experts.down_proj.data[self._local_expert_index(expert_idx)] = t
                else:
                    set_module_tensor_to_device(
                        self.model, name, self.device,
                        value=tensors[name], dtype=self.dtype,
                    )
