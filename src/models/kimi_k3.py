import gc
import inspect
import math
import re
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate.utils import set_module_tensor_to_device
from safetensors.torch import load_file as safe_load_file

from .base import BaseModelWrapper
from src.compression.activation_quant import fake_quantize_activation_nvfp4
from src.compression.initialization import make_initializer
from src.compression.initialization.gptq import make_quantizer, random_quantize, rtn_quantize
from src.evaluation.wiki_eval import get_dataset, prepare_test_dataloader
from src.moe.autograd_ops import AllToAllTokens
from src.moe.placement import ExpertSharder
from src.utils.progress_reporter import report_gptq_calib, report_ppl_layer
from src.utils.utils import create_act_cache

FP4_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                   -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def decode_mxfp4(packed, scale, dtype):
    """compressed-tensors ``mxfp4-pack-quantized`` -> dense ``[out, in]``.

    Two E2M1 codes per byte, the low nibble holding the even column, and one E8M0
    scale ``2 ** (s - 127)`` per group of columns. Matches compressed-tensors'
    ``MXFP4PackedCompressor.decompress`` bit for bit, without depending on its version.
    """
    lut = torch.tensor(FP4_E2M1_VALUES, dtype=torch.float32, device=packed.device)
    values = torch.stack([lut[(packed & 0xF).long()], lut[(packed >> 4).long()]], dim=-1).flatten(-2)
    group = values.shape[-1] // scale.shape[-1]
    return (values * torch.exp2(scale.float() - 127).repeat_interleave(group, dim=-1)).to(dtype)


def load_remote_k3_classes(ckpt_path):
    """``(KimiK3ForConditionalGeneration, modeling_kimi_linear module)`` from the checkpoint.

    The remote code targets transformers 4.56; three things changed in 5.x and are
    patched here, before any model is built:

    * ``OutputRecorder`` moved from ``transformers.utils.generic`` to ``output_capturing``;
    * flash-attention support is declared as ``_supports_flash_attn`` (the remote code
      sets the 4.x ``_supports_flash_attn_2``), without which FA2 is refused;
    * ``tie_weights`` now receives keyword arguments the remote override does not take;
    * ``create_causal_mask`` takes ``inputs_embeds`` (was ``input_embeds``) and no longer
      ``cache_position``.
    """
    import transformers.utils.generic as tf_generic
    if not hasattr(tf_generic, "OutputRecorder"):
        from transformers.utils.output_capturing import OutputRecorder
        tf_generic.OutputRecorder = OutputRecorder
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    top = get_class_from_dynamic_module("modeling_kimi_k3.KimiK3ForConditionalGeneration", ckpt_path)
    linear = sys.modules[sys.modules[top.__module__].KimiLinearForCausalLM.__module__]
    for name, module in list(sys.modules.items()):
        if not name.startswith("transformers_modules."):
            continue
        for obj in vars(module).values():
            if isinstance(obj, type) and getattr(obj, "_supports_flash_attn_2", False):
                obj._supports_flash_attn = True
    top.tie_weights = lambda self, *args, **kwargs: self.language_model.tie_weights(*args, **kwargs)

    mask_fn = getattr(linear.create_causal_mask, "__wrapped__", linear.create_causal_mask)
    accepted = set(inspect.signature(mask_fn).parameters)

    def create_causal_mask(**kwargs):
        if "input_embeds" in kwargs and "input_embeds" not in accepted:
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        return mask_fn(**{k: v for k, v in kwargs.items() if k in accepted})

    create_causal_mask.__wrapped__ = mask_fn
    linear.create_causal_mask = create_causal_mask
    return top, linear


class KimiK3Wrapper(BaseModelWrapper):
    """Expert-parallel Kimi-K3 on the checkpoint's remote code.

    Differences from Kimi-K2.5 that shape this wrapper:

    * **Attention Residuals.** A layer maps ``(prefix_sum, block_residual)`` to the same
      pair: attention and the MoE block read a softmax mix of ``prefix_sum`` and the
      block snapshots, and every ``attn_res_block_size``-th layer snapshots its input as
      a new block. ``data_all['input']`` holds ``prefix_sum``; snapshots are kept once per
      cache in ``data_all['attn_res_blocks']``. Before refinement ``get_mlp_input_all``
      replaces ``input`` with the MoE block's (normalised) input and parks the
      post-attention ``prefix_sum`` in ``data_all['residual']``, so ``calculate_mse`` needs
      nothing else.
    * **Latent MoE.** The router and shared experts see the hidden state; routed experts
      run at ``routed_expert_hidden_size`` between ``routed_expert_down_proj`` and
      ``routed_expert_norm`` + ``routed_expert_up_proj``. Tokens are dispatched in the
      latent space, and the per-expert linears compressed are ``w1``/``w3`` (gate/up) and
      ``w2`` (down), keyed ``gate_proj``/``up_proj``/``down_proj`` for the trainer.
    * **SiTU** expert activation: ``beta*tanh(g/beta)*sigmoid(g) * lb*tanh(u/lb)``.
    * **MXFP4** source experts, decoded to bf16 on load.

    Checkpoint and module names coincide (``language_model.model.layers.N...``).
    """

    _PROJ_TO_MODULE = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
    _EXPERT_KEY_RE = re.compile(r"\.block_sparse_moe\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$")

    def __init__(self, model_name, tokenizer, batch_size, seqlen, device, dtype, dummy=False):
        super().__init__(model_name, tokenizer, batch_size, seqlen, device, dtype, dummy=dummy)
        tc = self.model.config.text_config
        self.text_config = tc
        qc = getattr(tc, "quantization_config", None)
        if qc is not None:
            qc = qc if isinstance(qc, dict) else qc.to_dict()
            if qc.get("format") != "mxfp4-pack-quantized":
                raise NotImplementedError(f"Kimi-K3 source format {qc.get('format')!r} is not supported")
        if getattr(tc, "attn_res_block_size", None) is None:
            raise NotImplementedError("KimiK3Wrapper expects Attention Residuals (attn_res_block_size)")

        self.layer_prefix = "language_model.model.layers"
        self.num_layers = tc.num_hidden_layers
        self.num_experts = tc.num_experts
        self.first_k_dense_replace = tc.first_k_dense_replace
        self.hidden_size = tc.hidden_size
        self.attn_res_block_size = tc.attn_res_block_size
        self.kda_num_heads = tc.linear_attn_config["num_heads"]
        self.situ = ((tc.activation_situ_beta or 1.0, tc.activation_situ_linear_beta)
                     if tc.hidden_act == "situ" else None)
        self.is_moe = True
        self._act_cache_opts = (None, 2 * 1024 ** 3)

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.sharder = ExpertSharder(num_experts=self.num_experts, world_size=self.world_size)
        self.groupsize = 32
        self._owner_lut = torch.tensor([self.sharder.owner(e) for e in range(self.num_experts)],
                                       dtype=torch.long)
        self._local_eids = sorted(self.sharder.local_experts(self.rank))

    def _build_empty_model(self, cfg):
        top, self._k3_linear = load_remote_k3_classes(self.ckpt_path)
        # The remote text model forces flash_attention_2 whatever is requested.
        cfg._attn_implementation = "flash_attention_2"
        cfg.text_config._attn_implementation = "flash_attention_2"
        cfg.text_config.use_cache = False
        return top._from_config(cfg, attn_implementation="flash_attention_2", dtype=self.dtype).eval()

    def configure_quantization_from_config(self, config=None):
        if config is not None:
            self._act_cache_opts = (config.training.act_cache_dir or None,
                                    int(config.training.act_cache_mmap_threshold_gb * 1024 ** 3))
        return super().configure_quantization_from_config(config)

    def _reload_dtype(self, name):
        # KDA keeps A_log / dt_bias in fp32 next to bf16 weights; keep each tensor's dtype.
        return None

    # ------------------------------------------------------------------ structure / names

    def _text_model(self):
        return self.model.language_model.model

    def get_layer_module(self, idx):
        return self._text_model().layers[idx]

    def _is_moe_layer(self, layer_idx):
        return layer_idx >= self.first_k_dense_replace

    def _get_layer_by_name(self, layer_name):
        m = self._EXPERT_KEY_RE.search(layer_name)
        if m is not None:
            layer_name = layer_name[:m.start(2)] + self._PROJ_TO_MODULE[m.group(2)]
        return self.model.get_submodule(layer_name)

    def _layer_prefixes(self, layer_name):
        layer_idx = int(layer_name.split('.')[-1])
        base = f"{self.layer_prefix}.{layer_idx}"
        non_mlp = [
            f"{base}.input_layernorm",
            f"{base}.self_attn",
            f"{base}.post_attention_layernorm",
            f"{base}.self_attention_res_norm",
            f"{base}.self_attention_res_proj",
            f"{base}.mlp_res_norm",
            f"{base}.mlp_res_proj",
        ]
        if not self._is_moe_layer(layer_idx):
            return {"non_mlp": non_mlp, "mlp": [f"{base}.mlp"]}
        moe = f"{base}.block_sparse_moe"
        non_mlp += [f"{moe}.gate", f"{moe}.shared_experts", f"{moe}.routed_expert_down_proj",
                    f"{moe}.routed_expert_norm", f"{moe}.routed_expert_up_proj"]
        return {"non_mlp": non_mlp, "mlp": [f"{moe}.experts.{e}" for e in self._local_eids]}

    # ------------------------------------------------------------------ checkpoint I/O

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
                if n.endswith((".weight_scale", "inv_freq")):
                    continue  # scales are consumed with their weight_packed
                t = tensors[n]
                if n.endswith(".weight_packed"):
                    base = n[: -len(".weight_packed")]
                    w = decode_mxfp4(t, tensors[f"{base}.weight_scale"], self.dtype)
                    set_module_tensor_to_device(self.model, f"{base}.weight", self.device, value=w, dtype=self.dtype)
                    continue
                if n.endswith(".self_attn.A_log") and t.numel() > self.kda_num_heads:
                    # Stored zero-padded to 128 entries; the layer has num_heads (vLLM
                    # narrows the same way).
                    t = t[: self.kda_num_heads].contiguous()
                set_module_tensor_to_device(self.model, n, self.device, value=t)
            del tensors
        gc.collect()

    def _offload_names_to_meta(self, name_shard_pairs):
        names = [n if isinstance(n, str) else n[0] for n in name_shard_pairs]
        for n in names:
            if n.endswith((".weight_scale", "inv_freq")):
                continue
            if n.endswith(".weight_packed"):
                n = n[: -len(".weight_packed")] + ".weight"
            set_module_tensor_to_device(self.model, n, "meta")
        torch.cuda.empty_cache()

    def move_embed_to(self, device):
        names = self._names_from_ckpt(["language_model.model.embed_tokens"])
        if str(device).startswith("cuda"):
            self._set_tensors(names)
        else:
            self._offload_names_to_meta(names)

    def move_output_heads_to(self, device):
        names = self._names_from_ckpt([
            "language_model.model.output_attn_res_norm",
            "language_model.model.output_attn_res_proj",
            "language_model.model.norm",
            "language_model.lm_head",
        ])
        if str(device).startswith("cuda"):
            self._set_tensors(names)
        else:
            self._offload_names_to_meta(names)

    def _load_layer_for_eval(self, layer_idx, read_from_disk):
        layer_name = f"{self.layer_prefix}.{layer_idx}"
        if layer_idx <= read_from_disk and self._is_moe_layer(layer_idx):
            self.load_from_disc(layer_name)
        else:
            self.move_layer_to_gpu(layer_name)

    def save_to_disc(self, pfx, pairs):
        # Trainer keys say gate/up/down_proj; the checkpoint's expert linears are w1/w3/w2.
        super().save_to_disc(pfx, {self._PROJ_TO_MODULE.get(k, k): v for k, v in pairs.items()})

    def update_compressed_weights(self, layer_name, compressed_weights):
        layer = self._get_layer_by_name(layer_name)
        if isinstance(compressed_weights, tuple):
            qw, scales, global_scale, mask = self._split_weight_payload(compressed_weights)
            if self._EXPERT_KEY_RE.search(layer_name):
                # save_moe_experts_to_disc reads `<key>.Q`: the key is not a module path.
                self.temp_weights[f"{layer_name}.Q"] = qw.to(self.dtype)
            else:
                self.temp_weights[layer_name] = layer.weight.data
            self.temp_weights[f"{layer_name}.scale"] = scales
            if global_scale is not None:
                self.temp_weights[f"{layer_name}.global_scale"] = global_scale
            if mask is not None:
                self.temp_weights[f"{layer_name}.mask"] = mask
        else:
            qw = compressed_weights
        with torch.no_grad():
            layer.weight.data = qw.to(layer.weight.device, layer.weight.dtype)

    # ------------------------------------------------------------------ activation caches

    def _side_cache(self, data_all, key):
        """A tensor shaped like ``data_all['input']``, created on first use."""
        if key not in data_all:
            n, seqlen, hidden = data_all['input'].shape
            mmap_dir, threshold = self._act_cache_opts
            cache = create_act_cache(n, seqlen, hidden, data_all['input'].dtype, mmap_dir, threshold,
                                     tag=f"{key}_{id(data_all)}")
            data_all[key] = cache['input']
            if '_mmap_path' in cache:
                data_all.setdefault('_mmap_paths', []).append(cache['_mmap_path'])
        return data_all[key]

    def _snapshot_block(self, data_all, layer_idx):
        """At a block-start layer, keep its input as a new Attention Residual block (once)."""
        if layer_idx % self.attn_res_block_size:
            return
        layers = data_all.setdefault('attn_res_block_layers', [])
        if layer_idx in layers:
            return
        block = self._side_cache(data_all, f"attn_res_block_{layer_idx}")
        block.copy_(data_all['input'])
        data_all.setdefault('attn_res_blocks', []).append(block)
        layers.append(layer_idx)

    def _blocks_for(self, data_all, start, end, layer_idx):
        """``block_residual`` entering ``layer_idx`` for samples ``[start, end)``: [N, nb, H]."""
        blocks = [b for b, li in zip(data_all.get('attn_res_blocks', []),
                                     data_all.get('attn_res_block_layers', [])) if li < layer_idx]
        n, seqlen, hidden = end - start, data_all['input'].shape[1], self.hidden_size
        if not blocks:
            return torch.zeros(n * seqlen, 0, hidden, device=self.device, dtype=self.dtype)
        stacked = torch.stack([b[start:end].to(self.device, non_blocking=True) for b in blocks], dim=2)
        return stacked.reshape(n * seqlen, len(blocks), hidden)

    @torch.no_grad()
    def get_inputs(self, data_dict, data_loader):
        # The text model directly: the multimodal forward adds nothing for text ids.
        current_layer = self.get_layer_module(self.current_layer_idx)
        cache = {'index': 0}

        def store_input_hook(_, args, kwargs):
            start = cache['index'] * self.batch_size
            end = min(start + self.batch_size, data_dict['input'].shape[0])
            data_dict['input'][start:end] = args[0] if isinstance(args, tuple) else args
            cache['index'] += 1
            for k, v in kwargs.items():
                if k not in ("hidden_states", "attention_mask", "past_key_values", "block_residual"):
                    self.kwargs[k] = v
            raise ValueError

        total_batches = len(data_loader)
        handle = current_layer.register_forward_pre_hook(store_input_hook, with_kwargs=True)
        for batch_idx, batch in enumerate(data_loader):
            try:
                self._text_model()(input_ids=batch.to(self.device), use_cache=False)
            except ValueError:
                pass
            if self.rank == 0 and (batch_idx + 1) % max(1, total_batches // self.batch_report_divisor) == 0:
                print(f"  get_inputs: {batch_idx + 1}/{total_batches} batches", flush=True)
        handle.remove()

    def _build_layer_inputs(self, batch_size):
        # Unpadded calibration batches: flash-attention MLA is causal without a mask and
        # KDA needs none. cache_position is batch-independent.
        return {k: v for k, v in self.kwargs.items() if k not in ("attention_mask", "block_residual")}

    # ------------------------------------------------------------------ forward pieces

    def _attn_res(self, prefix, blocks, proj, norm):
        B, T, H = prefix.shape
        return self._k3_linear._apply_attn_res(prefix.reshape(-1, H), blocks, proj, norm).view(B, T, H)

    def _run_attention(self, layer, hidden_states, layer_inputs):
        if layer.is_linear_attn:
            return layer.self_attn(hidden_states=hidden_states, attention_mask=None, cache_params=None,
                                   use_cache=False, **layer_inputs)
        return layer.self_attn(hidden_states=hidden_states, attention_mask=None, position_ids=None,
                               past_key_values=None, use_cache=False, **layer_inputs)

    def _attention_part(self, layer, layer_idx, x, blocks, layer_inputs):
        """Remote ``_forward_attn_residual`` up to the MLP.

        Returns ``(prefix_sum after attention, block_residual after this layer, MLP input)``.
        """
        prefix = x
        h = (self._attn_res(prefix, blocks, layer.self_attention_res_proj, layer.self_attention_res_norm)
             if blocks.shape[1] > 0 else x)
        if layer_idx % self.attn_res_block_size == 0:
            blocks = torch.cat([blocks, prefix.reshape(-1, prefix.shape[-1]).unsqueeze(1)], dim=1)
            prefix = None
        attn_out = self._run_attention(layer, layer.input_layernorm(h), layer_inputs)
        prefix = attn_out if prefix is None else prefix + attn_out
        h = self._attn_res(prefix, blocks, layer.mlp_res_proj, layer.mlp_res_norm)
        return prefix, blocks, layer.post_attention_layernorm(h)

    def _act(self, gate, up):
        if self.situ is None:
            return F.silu(gate) * up
        beta, linear_beta = self.situ
        g, u = gate.float(), up.float()
        a = beta * torch.tanh(g / beta) * torch.sigmoid(g)
        if linear_beta is not None:
            u = linear_beta * torch.tanh(u / linear_beta)
        return (a * u).to(gate.dtype)

    def _dispatch(self, layer, hidden, with_weights=True):
        """Route MoE inputs ``hidden`` [B, T, H]; send each (token, expert) latent to its owner.

        Returns ``(send_idx, in_split, out_split, xin, win, eids, latent_shape)``; ``win``
        (fp32 gate weights) is None unless ``with_weights``.
        """
        moe = layer.block_sparse_moe
        pg = dist.group.WORLD
        topi, topw = moe.gate(hidden)
        latent = moe.routed_expert_down_proj(hidden.reshape(-1, hidden.shape[-1]))
        top_k = topi.shape[-1]

        tok_idx = torch.arange(latent.shape[0], device=self.device, dtype=torch.long).repeat_interleave(top_k)
        eid_flat = topi.reshape(-1).to(torch.long)
        owners = self._owner_lut.to(self.device)[eid_flat]
        perm = torch.argsort(owners, stable=True)
        owners = owners.index_select(0, perm)
        send_idx = tok_idx.index_select(0, perm)

        in_sizes = torch.bincount(owners, minlength=self.world_size).to(torch.long)
        all_sizes = [torch.empty_like(in_sizes) for _ in range(self.world_size)]
        dist.all_gather(all_sizes, in_sizes, group=pg)
        out_split = torch.stack(all_sizes)[:, self.rank].tolist()
        in_split = in_sizes.tolist()

        xin = AllToAllTokens.apply(latent.index_select(0, send_idx), out_split, in_split, pg)
        eids = AllToAllTokens.apply(eid_flat.index_select(0, perm).unsqueeze(1), out_split, in_split, pg).squeeze(1)
        win = None
        if with_weights:
            send_w = topw.reshape(-1).float().index_select(0, perm).unsqueeze(1)
            win = AllToAllTokens.apply(send_w, out_split, in_split, pg)
        return send_idx, in_split, out_split, xin, win, eids, latent.shape

    def _batched_expert_forward(self, xin, eids, compressed_weights=None, fake_act_quant=False):
        layer = self.get_layer_module(self.current_layer_idx)
        layer_key = self.get_current_layer()
        experts = layer.block_sparse_moe.experts

        unique_eids, inverse, counts = torch.unique(eids.to(torch.long), sorted=True,
                                                    return_inverse=True, return_counts=True)
        sort_idx = torch.argsort(inverse, stable=True)
        sorted_x = xin.index_select(0, sort_idx)
        out_buf = torch.empty_like(sorted_x)

        def _q(x):
            # Activation scale block aligned with the weight's dense scale group.
            return fake_quantize_activation_nvfp4(x, groupsize=self.groupsize) if fake_act_quant else x

        def _w(qw, proj):
            w = qw[proj]
            return w[0] if isinstance(w, tuple) else w

        offset = 0
        for i, eid_val in enumerate(unique_eids.tolist()):
            n = counts[i].item()
            inp_e = sorted_x[offset:offset + n]
            if compressed_weights is not None:
                qw = compressed_weights[f"{layer_key}.block_sparse_moe.experts.{eid_val}"]
                hidden = self._act(F.linear(_q(inp_e), _w(qw, "gate_proj")), F.linear(_q(inp_e), _w(qw, "up_proj")))
                out_e = F.linear(_q(hidden), _w(qw, "down_proj"))
            else:
                out_e = experts[eid_val](inp_e)
            out_buf[offset:offset + n] = out_e
            offset += n
        return out_buf.index_select(0, torch.argsort(sort_idx))

    def _moe_forward(self, layer, hidden, compressed_weights=None):
        """The MoE block on its input ``hidden`` [B, T, H] (remote ``KimiSparseMoeBlock``)."""
        moe = layer.block_sparse_moe
        send_idx, in_split, out_split, xin, win, eids, latent_shape = self._dispatch(layer, hidden)
        out_local = self._batched_expert_forward(xin, eids, compressed_weights)
        # Weight and sum in fp32, as the remote moe_infer does, then back to the model dtype.
        returned = AllToAllTokens.apply(out_local.float() * win, in_split, out_split, dist.group.WORLD)
        y = torch.zeros(latent_shape, device=self.device, dtype=torch.float32)
        y.index_add_(0, send_idx, returned)
        y = y.to(hidden.dtype)
        if moe.latent_moe_use_norm:
            y = moe.routed_expert_norm(y)
        y = moe.routed_expert_up_proj(y).view(hidden.shape)
        return y + moe.shared_experts(hidden)

    def _mlp(self, layer, layer_idx, hidden):
        if self._is_moe_layer(layer_idx):
            return self._moe_forward(layer, hidden)
        return layer.mlp(hidden)

    # ------------------------------------------------------------------ pipeline entry points

    @torch.no_grad()
    def get_layer_activations(self, data_all):
        idx = self.current_layer_idx
        layer = self.get_layer_module(idx)
        self._snapshot_block(data_all, idx)
        num_samples = data_all['input'].shape[0]
        for start in range(0, num_samples, self.batch_size):
            end = min(start + self.batch_size, num_samples)
            x = data_all['input'][start:end].to(self.device, non_blocking=True)
            layer_inputs = self._build_layer_inputs(x.shape[0])
            prefix, _, h = self._attention_part(layer, idx, x, self._blocks_for(data_all, start, end, idx),
                                                layer_inputs)
            data_all['input'][start:end] = (prefix + self._mlp(layer, idx, h)).detach().cpu()

    @torch.no_grad()
    def get_mlp_input_all(self, data_all):
        idx = self.current_layer_idx
        layer = self.get_layer_module(idx)
        self._snapshot_block(data_all, idx)
        residual = self._side_cache(data_all, "residual")
        num_samples = data_all['input'].shape[0]
        for start in range(0, num_samples, self.batch_size):
            end = min(start + self.batch_size, num_samples)
            x = data_all['input'][start:end].to(self.device, non_blocking=True)
            layer_inputs = self._build_layer_inputs(x.shape[0])
            prefix, _, h = self._attention_part(layer, idx, x, self._blocks_for(data_all, start, end, idx),
                                                layer_inputs)
            residual[start:end] = prefix.detach().cpu()
            data_all['input'][start:end] = h.detach().cpu()

    @torch.no_grad()
    def get_mlp_output_all(self, data_all):
        idx = self.current_layer_idx
        layer = self.get_layer_module(idx)
        residual = data_all["residual"]
        num_samples = data_all['input'].shape[0]
        for start in range(0, num_samples, self.batch_size):
            end = min(start + self.batch_size, num_samples)
            h = data_all['input'][start:end].to(self.device, non_blocking=True)
            out = residual[start:end].to(self.device, non_blocking=True) + self._mlp(layer, idx, h)
            data_all['input'][start:end] = out.detach().cpu()

    def get_mlp_input(self, batch):
        raise NotImplementedError(
            "Kimi-K3's MoE input depends on the cache's Attention Residual blocks; use get_mlp_input_all")

    @torch.no_grad()
    def get_mlp_output(self, mlp_input_batch):
        """MoE block output for its (normalised) input; the residual is added by the caller."""
        return self._moe_forward(self.get_layer_module(self.current_layer_idx), mlp_input_batch)

    def calculate_mse(self, mlp_input_batch, compressed_weights, self_attn=False, validation=False,
                      accumulation_steps=1, fake_act_quant=False):
        layer = self.get_layer_module(self.current_layer_idx)
        gate_p = getattr(self, "gate_weight_exponent", 0.0)
        with torch.no_grad():
            _, _, _, xin, win, eids, _ = self._dispatch(layer, mlp_input_batch, with_weights=gate_p > 0)
            out_fp = self._batched_expert_forward(xin, eids, compressed_weights=None)
        out_q = self._batched_expert_forward(xin, eids, compressed_weights=compressed_weights,
                                             fake_act_quant=fake_act_quant)

        total_mse = self._expert_recon_loss(out_q, out_fp, win)
        if not validation:
            (total_mse / accumulation_steps).backward()
        return total_mse.item()

    # ------------------------------------------------------------------ initialization

    def get_layer_initialization(self, trainer, gpt_all, config, logging):
        if logging is not None:
            logging = logging.logger
        idx = self.current_layer_idx
        layer = self.get_layer_module(idx)
        self.configure_quantization_from_config(config)
        self._snapshot_block(gpt_all, idx)
        layer_key = self.get_current_layer()
        experts = layer.block_sparse_moe.experts

        subset = {}
        for e in self._local_eids:
            for proj, attr in self._PROJ_TO_MODULE.items():
                subset[f"{layer_key}.block_sparse_moe.experts.{e}.{proj}"] = getattr(experts[e], attr)

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
                f"Unknown init_method={init_method!r}. Supported: 'gptq', 'obr', 'jsq', 'rtn', 'random'")

        gpts = {}
        for name in subset:
            gpts[name] = make_initializer(config, subset[name], name, self.device, self.dtype)
            if config.compression.quant_type == "nvfp4" or config.init.wbits < 16:
                gpts[name].quantizer = make_quantizer(config)

        def _add_batch(key, **kw):
            def _hook(_, inp, out):
                gpts[key].add_batch(inp[0].data, out.data, **kw)
            return _hook

        # gate (w1) and up (w3) share an input: up inherits gate's H after gate's
        # fasterquant. JSQ's range term needs up's own output, so it accumulates range only.
        handles = []
        for name, module in subset.items():
            if name.endswith(".up_proj"):
                if init_method == "jsq":
                    handles.append(module.register_forward_hook(_add_batch(name, h=False)))
                continue
            handles.append(module.register_forward_hook(_add_batch(name)))

        n_hessian = config.init.nsamples // self.world_size
        if logging is not None and self.rank == 0:
            logging.info(f"GPTQ Hessian accumulation: {n_hessian} samples, {len(self._local_eids)} local experts")
        calib_start = time.time()
        calib_report_interval = max(1, n_hessian // self.calib_report_divisor)
        try:
            with torch.no_grad():
                for j in range(n_hessian):
                    x = gpt_all['input'][j:j + 1].to(self.device, non_blocking=True)
                    _, _, h = self._attention_part(layer, idx, x, self._blocks_for(gpt_all, j, j + 1, idx),
                                                   self._build_layer_inputs(1))
                    self._moe_forward(layer, h)
                    if self.rank == 0 and (j + 1) % calib_report_interval == 0:
                        report_gptq_calib(j + 1, n_hessian, time.time() - calib_start)
        finally:
            for h in handles:
                h.remove()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # An expert that received no tokens is said so explicitly (see KimiK25Wrapper).
        for mh in gpts.values():
            if (getattr(mh, "nsamples", 0) == 0 and getattr(mh, "amax", None) is None
                    and hasattr(mh, "mark_no_tokens")):
                mh.mark_no_tokens()

        gptq_losses = []

        def _quantize(name, **kw):
            Q, scales = gpts[name].fasterquant(
                logging, percdamp=config.init.percdamp, blocksize=config.init.blocksize,
                groupsize=config.compression.groupsize, static_groups=config.init.static_groups,
                prunen=config.compression.prunen, prunem=config.compression.prunem, **kw)
            if hasattr(gpts[name], 'last_gptq_loss'):
                gptq_losses.append(gpts[name].last_gptq_loss)
            if scales is not None and refine_enabled:
                trainer.setup_layer_training(name, Q, scales,
                                             init_dense_weight=gpts[name].last_init_dense_weight,
                                             init_support_mask=gpts[name].last_init_support_mask)
            else:
                self.update_compressed_weights(name, (Q, scales) if scales is not None else Q)

        for name in gpts:
            if name.endswith(".up_proj"):
                continue
            _quantize(name)
            if name.endswith(".gate_proj"):
                up_name = f"{name[: -len('.gate_proj')]}.up_proj"
                gpts[up_name].H = gpts[name].H
                gpts[up_name].dead = gpts[name].dead
                _quantize(up_name, calculate_cholesky=False)
                gpts[up_name].free()
            gpts[name].free()

        if gptq_losses:
            trainer.gptq_avg_loss = sum(gptq_losses) / len(gptq_losses)
        del subset
        torch.cuda.empty_cache()
        dist.barrier()

    # ------------------------------------------------------------------ evaluation

    @torch.no_grad()
    def ppl_evaluation(self, read_from_disk=-1):
        dataset = get_dataset("open_thoughts", self.tokenizer)
        testloader = prepare_test_dataloader(dataset=dataset["test"], tokenizer=self.tokenizer,
                                             seqlen=self.model.seqlen, batch_size=4,
                                             world_size=self.world_size, rank=self.rank)
        pad_token_id = getattr(self.text_config, "pad_token_id", None)
        loss_fn = (torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)
                   if pad_token_id is not None else torch.nn.CrossEntropyLoss(reduction="none"))

        self.model.eval()
        self.move_embed_to(self.device)
        self.move_output_heads_to(self.device)
        text_model = self._text_model()

        ids_list = [batch["input_ids"].to("cpu") for batch in testloader]
        prefix_list = [None] * len(ids_list)
        blocks_list = [[] for _ in ids_list]

        idx_copy = self.current_layer_idx
        self._load_layer_for_eval(0, read_from_disk)
        ppl_start = time.time()
        for i in range(self.num_layers):
            if self.rank == 0:
                report_ppl_layer(i, self.num_layers, elapsed=time.time() - ppl_start)
            self.current_layer_idx = i
            layer = self.get_layer_module(i)
            for b, ids in enumerate(ids_list):
                x = (text_model.embed_tokens(ids.to(self.device)) if prefix_list[b] is None
                     else prefix_list[b].to(self.device))
                B, T, H = x.shape
                blocks = (torch.stack([blk.to(self.device) for blk in blocks_list[b]], dim=2).reshape(B * T, -1, H)
                          if blocks_list[b] else torch.zeros(B * T, 0, H, device=self.device, dtype=x.dtype))
                prefix, _, h = self._attention_part(layer, i, x, blocks, self._build_layer_inputs(B))
                if i % self.attn_res_block_size == 0:
                    blocks_list[b].append(x.cpu())
                prefix_list[b] = (prefix + self._mlp(layer, i, h)).cpu()
            self.offload_to_meta(f"{self.layer_prefix}.{i}")
            if i + 1 < self.num_layers:
                self._load_layer_for_eval(i + 1, read_from_disk)
            torch.cuda.empty_cache()

        nll_sum = torch.tensor(0.0, device=self.device)
        tok_cnt = torch.tensor(0.0, device=self.device)
        for b, ids in enumerate(ids_list):
            ids = ids.to(self.device)
            x = prefix_list[b].to(self.device)
            B, T, H = x.shape
            blocks = torch.stack([blk.to(self.device) for blk in blocks_list[b]], dim=2).reshape(B * T, -1, H)
            x = self._attn_res(x, blocks, text_model.output_attn_res_proj, text_model.output_attn_res_norm)
            logits = self.model.language_model.lm_head(text_model.norm(x))[:, :-1, :]
            labels = ids[:, 1:]
            nll = loss_fn(logits.permute(0, 2, 1), labels).float()
            mask = labels != loss_fn.ignore_index
            nll_sum += (nll * mask).sum()
            tok_cnt += mask.sum()

        self.move_embed_to("meta")
        self.move_output_heads_to("meta")
        self.current_layer_idx = idx_copy
        dist.all_reduce(nll_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(tok_cnt, op=dist.ReduceOp.SUM)
        ppl = math.exp((nll_sum / tok_cnt).item())
        gc.collect()
        torch.cuda.empty_cache()
        return ppl
