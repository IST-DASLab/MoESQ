import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
import math
import os, gc
from contextlib import ExitStack
from accelerate.utils import set_module_tensor_to_device
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file
from .base import BaseModelWrapper, token_clip_weights
from src.moe.placement import ExpertSharder
from src.moe.autograd_ops import AllToAllTokens
from src.compression.activation_quant import fake_quantize_activation_nvfp4
from src.compression.initialization.gptq import *
from src.compression.initialization import make_initializer
from src.evaluation.wiki_eval import *
from src.utils.progress_reporter import (
    report_gptq_calib, report_ppl_layer,
)


class Qwen3MoeWrapper(BaseModelWrapper):
    def __init__(self, model_name, tokenizer, batch_size, seqlen, device, dtype):
        super().__init__(model_name, tokenizer, batch_size, seqlen, device, dtype)
        self.layer_prefix = "model.layers"
        self.num_layers = len(self.model.model.layers)
        self.num_experts = self.model.config.num_experts
        self.decoder_sparse_step = getattr(self.model.config, 'decoder_sparse_step', 1)
        self.mlp_only_layers = getattr(self.model.config, 'mlp_only_layers', [])
        self.is_moe = True

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.sharder = ExpertSharder(num_experts=self.num_experts, world_size=self.world_size)
        self.groupsize = 32

        self._owner_lut = torch.tensor(
            [self.sharder.owner(e) for e in range(self.num_experts)],
            dtype=torch.long
        )

    def _is_moe_layer(self, layer_idx):
        if layer_idx in self.mlp_only_layers:
            return False
        return self.num_experts > 0 and (layer_idx + 1) % self.decoder_sparse_step == 0

    def _layer_prefixes(self, layer_name):
        layer_idx = int(layer_name.split('.')[-1])
        base = f"{self.layer_prefix}.{layer_idx}"
        if not self._is_moe_layer(layer_idx):
            non_mlp = [
                f"{base}.input_layernorm",
                f"{base}.self_attn",
                f"{base}.post_attention_layernorm"
            ]
            local_expert = [
                f"{base}.mlp"
            ]
        else:
            non_mlp = [
                f"{base}.input_layernorm",
                f"{base}.self_attn",
                f"{base}.mlp.gate",
                f"{base}.post_attention_layernorm"
            ]
            local_expert = [
                f"{base}.mlp.experts.{e}"
                for e in range(self.num_experts)
                if self.sharder.owner(e) == self.rank
            ]
        return {"non_mlp": non_mlp, "mlp": local_expert}

    # An MoE layer's router and experts are reached ONLY through the three methods
    # below. A wrapper for another module layout (transformers 5.x fused experts,
    # qwen3_moe_fused.py) overrides just these and inherits all routing, loss and
    # GPTQ logic unchanged.
    def _router_logits(self, layer, hidden):
        return layer.mlp.gate(hidden)

    def _router_config(self, layer):
        """(top_k, norm_topk_prob) of an MoE layer's router."""
        return layer.mlp.top_k, layer.mlp.norm_topk_prob

    def _expert_module(self, layer, eid):
        """Module computing expert `eid` as down_proj(silu(gate_proj(x)) * up_proj(x)),
        with gate_proj / up_proj / down_proj as its nn.Linear children."""
        return layer.mlp.experts[eid]

    def move_embed_to(self, device):
        names = self._names_from_ckpt(["model.embed_tokens"])
        if str(device).startswith("cuda"):
            self._set_tensors(names)
        else:
            self._offload_names_to_meta(names)

    def move_output_heads_to(self, device):
        names = []
        names += self._names_from_ckpt("model.norm")
        names += self._names_from_ckpt("lm_head")
        if str(device).startswith("cuda"):
            self._set_tensors(names)
        else:
            self._offload_names_to_meta(names)

    def get_mlp_input(self, batch):
        current_layer = self.get_layer_module(self.current_layer_idx)

        additional_layer_inputs = {"attention_mask": None}
        for k, v in self.kwargs.items():
            additional_layer_inputs[k] = v

        hidden_states = current_layer.input_layernorm(batch)
        hidden_states, _ = current_layer.self_attn(hidden_states, **additional_layer_inputs)
        return hidden_states + batch

    def get_layer_module(self, idx):
        return self.model.model.layers[idx]

    def update_compressed_weights(self, layer_name, compressed_weights):
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
                if ".experts" in layer_name:
                    layer.weight.data = compressed_weights.to(layer.weight.device)
                else:
                    layer.weight.data = compressed_weights.to(layer.weight.dtype).to(layer.weight.device)

    @torch.no_grad()
    def get_layer_activations(self, data_all):
        current_layer = self.get_layer_module(self.current_layer_idx)
        num_samples = data_all['input'].shape[0]
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, num_samples)
            x = data_all['input'][start_idx:end_idx].to(self.device, non_blocking=True)
            additional_layer_inputs = {"attention_mask": None}
            for k, v in self.kwargs.items():
                additional_layer_inputs[k] = v

            hidden_states = current_layer.input_layernorm(x)
            attn_out, _ = current_layer.self_attn(hidden_states, **additional_layer_inputs)
            mlp_input = x + attn_out
            if not self._is_moe_layer(self.current_layer_idx):
                hidden_states = current_layer.post_attention_layernorm(mlp_input)
                hidden_states = current_layer.mlp(hidden_states)
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]
                out = hidden_states + mlp_input
            else:
                out = self.run_expert_parallel(mlp_input)

            data_all['input'][start_idx:end_idx] = out.detach().cpu()

    @torch.no_grad()
    def get_mlp_output(self, mlp_input_batch):
        if not self._is_moe_layer(self.current_layer_idx):
            current_layer = self.get_layer_module(self.current_layer_idx)
            hidden_states = current_layer.post_attention_layernorm(mlp_input_batch)
            hidden_states = current_layer.mlp(hidden_states)
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            return hidden_states + mlp_input_batch
        return self.run_expert_parallel(mlp_input_batch)

    def _dispatch_tokens(self, mlp_input_batch):
        """Route tokens to expert-owning ranks via all-to-all."""
        layer = self.get_layer_module(self.current_layer_idx)
        device = self.device
        pg = dist.group.WORLD

        B, T, H = mlp_input_batch.shape
        hidden = layer.post_attention_layernorm(mlp_input_batch)
        x_flat = hidden.reshape(B * T, H)

        top_k, norm_topk_prob = self._router_config(layer)
        router_logits = self._router_logits(layer, hidden)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        topw, topi = torch.topk(routing_weights, top_k, dim=-1)
        if norm_topk_prob:
            topw = topw / topw.sum(dim=-1, keepdim=True)

        tok_idx_flat = torch.arange(B * T, device=device, dtype=torch.long).repeat_interleave(top_k)
        eid_flat = topi.reshape(-1).to(torch.long)
        w_flat = topw.reshape(-1).to(self.dtype)

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

    def _batched_expert_forward(self, xin, eids, compressed_weights=None, fake_act_quant=False):
        """Process all local experts in a single batched pass."""
        layer = self.get_layer_module(self.current_layer_idx)
        layer_key = self.get_current_layer()

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
                gate_out = F.linear(_q(inp_e), qw["gate_proj"][0] if isinstance(qw["gate_proj"], tuple) else qw["gate_proj"])
                up_out = F.linear(_q(inp_e), qw["up_proj"][0] if isinstance(qw["up_proj"], tuple) else qw["up_proj"])
                hidden = F.silu(gate_out) * up_out
                out_e = F.linear(_q(hidden), qw["down_proj"][0] if isinstance(qw["down_proj"], tuple) else qw["down_proj"])
            else:
                out_e = self._expert_module(layer, eid_val)(inp_e)

            out_buf[offset:offset + n] = out_e
            offset += n

        unsort_idx = torch.argsort(sort_idx)
        return out_buf.index_select(0, unsort_idx)

    def run_expert_parallel(self, mlp_input_batch, compressed_weights=None):
        pg = dist.group.WORLD

        x_flat, hidden, send_idx_flat, in_split_sizes, out_split_sizes, xin, win, eids, B, T, H = \
            self._dispatch_tokens(mlp_input_batch)

        out_local = self._batched_expert_forward(xin, eids, compressed_weights)
        xin = out_local * win

        returned = AllToAllTokens.apply(xin, in_split_sizes, out_split_sizes, pg)

        y_flat = x_flat.new_zeros(x_flat.shape)
        y_flat.index_add_(0, send_idx_flat, returned)
        y = y_flat.view(B, T, H)

        return y + mlp_input_batch

    @torch.no_grad()
    def moe_val_recon_metrics(self, mlp_input_batch, compressed_weights, fake_act_quant=False):
        # Exponent-invariant validation metrics (returned as local sum/count pairs;
        # the caller all-reduces SUM over ranks, so mean = sum/count is correct
        # regardless of how tokens are split across ranks):
        #   recon_unweighted = mean ||delta_e||^2 over (token,expert) rows
        #   block_error      = mean ||sum_e g_e delta_e||^2 over tokens
        # delta_e = f_hat_e(x) - f_e(x). Uses the same dispatch as the forward.
        pg = dist.group.WORLD
        x_flat, hidden, send_idx_flat, in_split_sizes, out_split_sizes, xin, win, eids, B, T, H = \
            self._dispatch_tokens(mlp_input_batch)

        out_fp = self._batched_expert_forward(xin, eids, compressed_weights=None)
        out_q = self._batched_expert_forward(xin, eids, compressed_weights=compressed_weights,
                                             fake_act_quant=fake_act_quant)
        delta = (out_q - out_fp).float()

        recon_se = delta.pow(2).sum()
        recon_n = torch.tensor(float(delta.numel()), device=self.device)

        weighted = (win.float() * delta).to(self.dtype)
        returned = AllToAllTokens.apply(weighted, in_split_sizes, out_split_sizes, pg).float()
        y_flat = x_flat.new_zeros((B * T, H), dtype=torch.float)
        y_flat.index_add_(0, send_idx_flat, returned)
        block_se = y_flat.pow(2).sum()
        block_n = torch.tensor(float(y_flat.numel()), device=self.device)

        # Concentration of the DENSE block output on its largest tokens: the share of
        # sum_t ||y_t||^2 carried by the top-8 tokens of this micro-batch on this rank.
        # ~1% on an ordinary layer; 90-100% on a layer that produces the attention-sink
        # massive activation (Qwen3-30B layers 1-3). Same all-to-all as `delta` above.
        weighted_fp = (win.float() * out_fp.float()).to(self.dtype)
        returned_fp = AllToAllTokens.apply(weighted_fp, in_split_sizes, out_split_sizes, pg).float()
        y_fp = x_flat.new_zeros((B * T, H), dtype=torch.float)
        y_fp.index_add_(0, send_idx_flat, returned_fp)
        n2 = y_fp.pow(2).sum(dim=-1)
        k = min(8, n2.numel())
        top = n2.topk(k)
        sink_top8_se = top.values.sum()
        sink_tot_se = n2.sum()

        # Split the block ERROR by whether the token is one of the sink tokens, which are
        # identified by the DENSE output energy (top.indices) and so do not move with the
        # compression. block_sink_se + block_ord_se == block_se by construction.
        # This is what separates "the fit is bad" from "the metric is sink-weighted":
        # a layer whose ordinary-token error matches its neighbours is not damaged, it is
        # only measured on ~8 tokens that carry ~50% of the block output.
        e2 = y_flat.pow(2).sum(dim=-1)
        block_sink_se = e2[top.indices].sum()
        block_sink_n = torch.tensor(float(k * y_flat.shape[1]), device=self.device)

        return (recon_se.detach(), recon_n, block_se.detach(), block_n,
                sink_top8_se.detach(), sink_tot_se.detach(),
                block_sink_se.detach(), block_sink_n)

    def _forward_single_expert(self, layer, expert_id, x_e, compressed_weights):
        expert = self._expert_module(layer, expert_id)

        class LinearWeightHook:
            def __init__(self, module, qweight):
                self.module = module
                self.qw = qweight
                self.saved = module.forward
            def __enter__(self):
                def new_forward(module_self, x):
                    return F.linear(x, self.qw, self.module.bias)
                self.module.forward = new_forward.__get__(self.module, torch.nn.Linear)
            def __exit__(self, a, b, c):
                self.module.forward = self.saved

        hooks = []
        try:
            if compressed_weights is not None:
                for name, module in expert.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        key = f"{self.get_current_layer()}.mlp.experts.{expert_id}"
                        hooks.append(LinearWeightHook(module, compressed_weights[key][name]))
            with ExitStack() as stack:
                for h in hooks:
                    stack.enter_context(h)
                out = expert(x_e)
            return out
        finally:
            hooks.clear()

    def calculate_mse(self, mlp_input_batch, compressed_weights, self_attn=False, validation=False, accumulation_steps=1, fake_act_quant=False):
        if not self._is_moe_layer(self.current_layer_idx):
            return super(Qwen3MoeWrapper, self).calculate_mse(
                mlp_input_batch, compressed_weights,
                self_attn=self_attn, validation=validation,
                accumulation_steps=accumulation_steps,
                fake_act_quant=fake_act_quant,
            )
        layer = self.get_layer_module(self.current_layer_idx)
        device = self.device
        pg = dist.group.WORLD

        gate_p = getattr(self, "gate_weight_exponent", 0.0)
        gate_weighted = gate_p > 0
        B, T, H = mlp_input_batch.shape
        with torch.no_grad():
            hidden = layer.post_attention_layernorm(mlp_input_batch)
            x_flat = hidden.reshape(B * T, H)

            top_k, norm_topk_prob = self._router_config(layer)
            router_logits = self._router_logits(layer, hidden)
            routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
            topw, topi = torch.topk(routing_weights, top_k, dim=-1)
            if gate_weighted and norm_topk_prob:
                topw = topw / topw.sum(dim=-1, keepdim=True)

            tok_idx_flat = torch.arange(B * T, device=device, dtype=torch.long).repeat_interleave(top_k)
            eid_flat = topi.reshape(-1).to(torch.long)
            if gate_weighted:
                w_flat = topw.reshape(-1).to(self.dtype)

            owner_lut = self._owner_lut.to(device)
            owners_flat = owner_lut[eid_flat]
            perm = torch.argsort(owners_flat, stable=True)
            owners_flat = owners_flat.index_select(0, perm)
            send_idx_flat = tok_idx_flat.index_select(0, perm)
            send_eid_flat = eid_flat.index_select(0, perm)
            if gate_weighted:
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
            eids = AllToAllTokens.apply(send_eid_flat.unsqueeze(1), out_split_sizes, in_split_sizes, pg).squeeze(1)
            if gate_weighted:
                w_local = AllToAllTokens.apply(send_w_flat.unsqueeze(1), out_split_sizes, in_split_sizes, pg)

        with torch.no_grad():
            out_fp = self._batched_expert_forward(xin, eids, compressed_weights=None)
        out_q = self._batched_expert_forward(xin, eids, compressed_weights=compressed_weights, fake_act_quant=fake_act_quant)

        # sink-aware row weights (None unless refine.token_weight_clip_k > 0): caps the
        # massive-activation rows so they stop dominating the layers that produce the sink
        clip = token_clip_weights(out_fp, getattr(self, "token_weight_clip_k", 0.0))
        if gate_weighted or clip is not None:
            # weight each (token,expert) row by g^p (p = gate_weight_exponent):
            # p=1 is a Jensen upper bound on the block-output error ||sum_e g_e d_e||^2,
            # p=2 its exact diagonal. Weighted mean keeps the loss scale comparable across p.
            if gate_weighted:
                wp = w_local.float().pow(gate_p)
            else:
                wp = torch.ones(out_fp.shape[0], 1, device=out_fp.device, dtype=torch.float)
            if clip is not None:
                wp = wp * clip
            diff2 = (out_q - out_fp).float().pow(2)
            total_mse = (wp * diff2).sum() / (wp.sum().clamp_min(1e-8) * diff2.shape[-1])
        else:
            total_mse = self.loss_fn(out_q, out_fp)
        if not validation:
            (total_mse / accumulation_steps).backward()

        return total_mse.item()

    def get_layer_initialization(self, trainer, gpt_all, config, logging):
        if not self._is_moe_layer(self.current_layer_idx):
            return super(Qwen3MoeWrapper, self).get_layer_initialization(
                trainer, gpt_all, config, logging
            )
        if logging is not None:
            logging = logging.logger
        layer_idx = self.current_layer_idx
        layer = self.get_layer_module(layer_idx)
        rank = self.rank
        self.configure_quantization_from_config(config)

        owned_experts = [e for e in range(self.num_experts) if self.sharder.owner(e) == rank]
        subset = {}

        for e in owned_experts:
            expert = self._expert_module(layer, e)
            base_prefix = f"{self.get_current_layer()}.mlp.experts.{e}"
            for name, module in expert.named_modules():
                if isinstance(module, torch.nn.Linear):
                    subset[f"{base_prefix}.{name}"] = module

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

        if init_method in ("gptq", "obr", "jsq"):
            gpts = {}
            for name in subset:
                gpts[name] = make_initializer(config, subset[name], name, self.device, self.dtype)
                if config.compression.quant_type == "nvfp4" or config.init.wbits < 16:
                    gpts[name].quantizer = make_quantizer(config)

            def _add_batch(full_key, **kw):
                def _hook(_, inp, out):
                    gpts[full_key].add_batch(inp[0].data, out.data, **kw)
                return _hook

            handles = []
            n_hessian = config.init.nsamples // self.world_size
            try:
                for full_name, module in subset.items():
                    # gate/up share an input, so up_proj normally inherits gate's H
                    # after gate's fasterquant instead of being hooked. JSQ's range
                    # term depends on this linear's OUTPUT (y = x W_up^T), so up_proj
                    # must accumulate its own statistics -- range only (h=False):
                    # its H is overwritten by gate's before its fasterquant anyway.
                    if "up_proj" in full_name:
                        if init_method != "jsq":
                            continue
                        handles.append(module.register_forward_hook(_add_batch(full_name, h=False)))
                        continue
                    handles.append(module.register_forward_hook(_add_batch(full_name)))

                calib_start = time.time()
                calib_report_interval = max(1, n_hessian // self.calib_report_divisor)
                with torch.no_grad():
                    for j in range(n_hessian):
                        x = gpt_all['input'][j].unsqueeze(0).to(self.device, non_blocking=True)

                        additional_layer_inputs = {"attention_mask": None}
                        for k, v in self.kwargs.items():
                            additional_layer_inputs[k] = v

                        hidden_states = layer.input_layernorm(x)
                        attn_out, _ = layer.self_attn(hidden_states, **additional_layer_inputs)
                        x = x + attn_out

                        _ = self.run_expert_parallel(x, compressed_weights=None)
                        if rank == 0 and (j + 1) % calib_report_interval == 0:
                            report_gptq_calib(j + 1, n_hessian, time.time() - calib_start)
            finally:
                for h in handles:
                    h.remove()

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

            dist.barrier()

    def _load_layer_for_eval(self, layer_idx, read_from_disk):
        layer_name = f"{self.layer_prefix}.{layer_idx}"
        if layer_idx <= read_from_disk and self._is_moe_layer(layer_idx):
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

        additional_layer_inputs = {"attention_mask": None}
        for k, v in self.kwargs.items():
            additional_layer_inputs[k] = v

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
                    x = self.model.model.embed_tokens(ids)
                else:
                    x = x_cpu.to(self.device, non_blocking=True)

                hidden = layer.input_layernorm(x)
                attn_out, _ = layer.self_attn(hidden, **additional_layer_inputs)
                x = x + attn_out
                if not self._is_moe_layer(i):
                    hidden_states = layer.post_attention_layernorm(x)
                    hidden_states = layer.mlp(hidden_states)
                    if isinstance(hidden_states, tuple):
                        hidden_states = hidden_states[0]
                    x = hidden_states + x
                else:
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

            x = self.model.model.norm(x)
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
                    if ".expert" in pfx:
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
                        new_name = local_name[:-len(".weight")]
                        base = f"{pfx}.{new_name}"
                        to_save[base + ".weight_packed"] = compressed["weight_packed"]
                        to_save[base + ".weight_scale"]  = scale
                        to_save[base + ".weight_shape"]  = compressed["weight_shape"]
                    else:
                        to_save[f"{pfx}.{local_name}"] = tensor.detach().cpu()

            safe = pfx.replace('.', '_')
            path = os.path.join(self.save_dir, f"{safe}.safetensors")
            safe_save_file(to_save, path)
