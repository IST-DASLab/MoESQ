import json
import math
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file as safe_save_file
from transformers import AutoModelForCausalLM

from .qwen35_moe import Qwen35MoeWrapper


class ShardedTableEmbedding(nn.Module):
    """``nn.Embedding`` lookup over row shards that stay memory-mapped on the host.

    Qwen3.8-Flash-Next's per-layer n-gram table is ~51B parameters (~95 GiB bf16), more
    than a GPU holds and never compressed. The checkpoint stores it as row shards that
    concatenate along dim 0; each shard is mapped read-only straight from its
    safetensors file, so ranks on one node share a single copy in the page cache.
    """

    _DTYPES = {"BF16": (np.uint16, torch.bfloat16), "F16": (np.float16, torch.float16),
               "F32": (np.float32, torch.float32)}

    def __init__(self, shard_files):
        """``shard_files``: ``[(safetensors_path, tensor_name), ...]`` in row order."""
        super().__init__()
        self._shards = [self._map(path, name) for path, name in shard_files]
        rows = torch.tensor([s.shape[0] for s in self._shards])
        self._starts = torch.cumsum(rows, 0) - rows
        self.num_embeddings = int(rows.sum())
        self.embedding_dim = self._shards[0].shape[1]
        # The HF caller asks for `.weight.device` to pick where to run the lookup; a meta
        # tensor tells it to leave the ids where they are.
        self.weight = torch.empty(0, device="meta")

    @classmethod
    def _map(cls, path, name):
        with open(path, "rb") as f:
            header_len = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(header_len))
        meta = header[name]
        np_dtype, torch_dtype = cls._DTYPES[meta["dtype"]]
        begin, _ = meta["data_offsets"]
        arr = np.memmap(path, dtype=np_dtype, mode="c", offset=8 + header_len + begin,
                        shape=tuple(meta["shape"]))
        t = torch.from_numpy(arr)
        return t.view(torch_dtype) if t.dtype != torch_dtype else t

    def forward(self, ids):
        flat = ids.reshape(-1).to("cpu", torch.long)
        out = torch.empty(flat.numel(), self.embedding_dim, dtype=self._shards[0].dtype)
        shard_of = torch.bucketize(flat, self._starts, right=True) - 1
        for s in torch.unique(shard_of).tolist():
            sel = shard_of == s
            out[sel] = self._shards[s].index_select(0, flat[sel] - self._starts[s])
        return out.view(*ids.shape, self.embedding_dim).to(ids.device)


def qsa_block_scores(indexer, hidden_states, position_embeddings, sample):
    """Indexer scores ``[S, nb]`` of one sample's queries against its complete key blocks.

    Blocks a query cannot see yet (``>= (s + 1) // r``) score ``-inf``. The formulas are
    the reference ``Qwen4ExpTextQSAIndexer.forward``'s, for every query at once.
    """
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    x = hidden_states[sample:sample + 1]
    _, S, _ = x.shape
    d, r = indexer.index_head_dim, indexer.compress_ratio
    full_cos, full_sin = (t[sample:sample + 1] for t in position_embeddings)
    q, token_k = torch.split(
        indexer.index_qk_proj(x), [indexer.index_n_heads * d, indexer.index_kv_heads * d], dim=-1)
    q = indexer.q_layernorm(q.reshape(1, S, -1, d))
    q = apply_rotary_pos_emb(q, cos=full_cos[:, -S:, :], sin=full_sin[:, -S:, :], unsqueeze_dim=2)[0]
    raw_keys = token_k.reshape(S, -1, d).squeeze(1)

    nb = S // r
    pooled = raw_keys[:nb * r].reshape(nb, r, d).float().mean(dim=1).to(raw_keys.dtype)
    pooled = indexer.k_layernorm(pooled)
    starts = torch.arange(nb, device=x.device) * r
    block_keys = apply_rotary_pos_emb(
        pooled.unsqueeze(1), cos=full_cos[0].index_select(0, starts),
        sin=full_sin[0].index_select(0, starts)).squeeze(1)
    scores = torch.relu(torch.einsum("shd,nd->snh", q.float(), block_keys.float())).sum(dim=-1)
    scores = scores / math.sqrt(d)
    n_complete = (torch.arange(S, device=x.device) + 1) // r
    visible = torch.arange(nb, device=x.device).view(1, nb) < n_complete.view(S, 1)
    return scores.masked_fill(~visible, float("-inf"))


def causal_qsa_selection(indexer, hidden_states, position_embeddings, attention_mask):
    """``Qwen4ExpTextQSAIndexer.forward`` for a plain causal mask and no cache, vectorized.

    The reference loops over every (batch, query) pair in Python, which dominates a
    calibration pass. With every earlier token visible, query ``s`` sees the complete
    key blocks ``[0, (s + 1) // r)`` plus a tail of up to ``r - 1`` tokens, and the pooled
    block keys do not depend on the query, so all queries are scored in one pass. Same
    formulas as the reference; only the reduction order of the scores can differ.

    Block scores are a sum of ReLUs, so exact ties (typically at 0) are common, and
    ``torch.topk`` leaves the choice among tied blocks unspecified. Here ties go to the
    lower block index (a stable sort), which the reference may or may not reproduce;
    away from a tied budget boundary the selections are identical.
    """
    B, S, _ = hidden_states.shape
    r = indexer.compress_ratio
    nb = S // r
    device = hidden_states.device
    pos = torch.arange(S, device=device)
    n_complete = (pos + 1) // r
    k_sel = n_complete.clamp(max=indexer.block_topk)
    token_block = pos // r
    in_block = token_block < nb
    tail = (pos.view(1, S) >= (r * n_complete).view(S, 1)) & (pos.view(1, S) <= pos.view(S, 1))

    selected = tail.unsqueeze(0).repeat(B, 1, 1)
    if nb > 0:
        rank = torch.arange(min(indexer.block_topk, nb), device=device).view(1, -1)
        for b in range(B):
            scores = qsa_block_scores(indexer, hidden_states, position_embeddings, b)
            top = scores.sort(dim=-1, descending=True, stable=True).indices[:, :rank.shape[1]]
            blocks = torch.zeros(S, nb + 1, dtype=torch.bool, device=device)
            blocks.scatter_(1, torch.where(rank < k_sel.view(S, 1), top, nb), True)
            selected[b][:, in_block] |= blocks[:, :nb].gather(
                1, token_block[in_block].view(1, -1).expand(S, -1))

    selected = selected.unsqueeze(1)
    if attention_mask.is_floating_point():
        min_dtype = torch.finfo(attention_mask.dtype).min
        return torch.where(selected, attention_mask.new_zeros(()), min_dtype)
    return selected


def _is_plain_causal(attention_mask):
    visible = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
    S = visible.shape[-1]
    if visible.shape[-2] != S:
        return False
    causal = torch.ones(S, S, dtype=torch.bool, device=visible.device).tril()
    return bool((visible[:, 0] == causal).all())


def _patch_qsa_indexer(indexer):
    reference = indexer.forward

    def forward(hidden_states, position_embeddings, attention_mask, past_key_values):
        if past_key_values is None and _is_plain_causal(attention_mask):
            return causal_qsa_selection(indexer, hidden_states, position_embeddings, attention_mask)
        return reference(hidden_states, position_embeddings, attention_mask, past_key_values)

    indexer.forward = forward


class Qwen4ExpWrapper(Qwen35MoeWrapper):
    """Expert-parallel Qwen3.8-Flash-Next (``qwen4_exp``) on the native transformers classes.

    The MoE block is Qwen3.5's (fused experts, softmax top-k router, sigmoid-gated
    shared expert); what changes is the residual stream around it:

    * **Gated-residual hyper-connections.** Layers pass ``hc_count`` streams, a
      ``[B, T, hc_count * hidden]`` state. Attention and the MoE block each read a
      learned mix of the streams (``*_hyper_connection``) and inject their output back
      into every stream. The activation caches therefore hold the full hyper state
      (``activation_hidden_size``) and ``mlp_input`` is the state after attention.
    * **Per-layer n-gram embedding (PLE).** One layer adds hashed n-gram features of the
      raw token ids before attention, so the token ids of every calibration sample are
      kept next to its activations (``data_all['input_ids']``). The n-gram table stays
      on the host (``ShardedTableEmbedding``).
    * **Qwen Sparse Attention** on the full-attention layers; its block indexer runs
      vectorized for the (unpadded, causal) calibration batches.

    There is no layernorm on the layer itself (the hyper-connections normalise) and no
    final norm: ``hyper_connection_mixer`` collapses the streams before ``lm_head``.
    """

    _NGRAM_SHARD_RE = re.compile(
        r"^(?P<emb>.*\.ple_embedding\.ngram_embedding)\.shard_(?P<idx>\d+)\.weight$")

    def __init__(self, model_name, tokenizer, batch_size, seqlen, device, dtype):
        super().__init__(model_name, tokenizer, batch_size, seqlen, device, dtype)
        text_cfg = self._text_model.config
        self.hc_count = text_cfg.hc_count
        self.activation_hidden_size = self.hc_count * self.hidden_size
        self._ple_layers = {i for i, layer in enumerate(self._get_layers()) if layer.ple is not None}
        for layer in self._get_layers():
            attn = getattr(layer, "self_attn", None)
            if attn is not None and hasattr(attn, "indexer"):
                _patch_qsa_indexer(attn.indexer)

    def _build_empty_model(self, cfg):
        # transformers ships no flash-attention path for QSA; the indexer mask needs sdpa.
        text_cfg = getattr(cfg, "text_config", None) or cfg
        text_cfg._attn_implementation = "sdpa"
        return AutoModelForCausalLM.from_config(text_cfg, attn_implementation="sdpa").eval()

    # ------------------------------------------------------------------ names / I/O

    def _layer_prefixes(self, layer_name):
        layer_idx = int(layer_name.split('.')[-1])
        prefixes = super()._layer_prefixes(layer_name)
        base = f"{self.layer_prefix}.{layer_idx}"
        attn_prefix = f"{base}.linear_attn" if self._is_linear_attention_layer(layer_idx) else f"{base}.self_attn"
        non_mlp = [
            f"{base}.attn_hyper_connection",
            attn_prefix,
            f"{base}.mlp.gate",
            f"{base}.mlp.shared_expert",
            f"{base}.mlp.shared_expert_gate",
            f"{base}.mlp_hyper_connection",
        ]
        if layer_idx in self._ple_layers:
            non_mlp.append(f"{base}.ple")
        return {"non_mlp": non_mlp, "mlp": prefixes["mlp"]}

    def _norm_prefix(self):
        return "model.hyper_connection_mixer"

    def _split_ngram(self, name_shard_pairs):
        table, rest = {}, []
        for n, s in name_shard_pairs:
            m = self._NGRAM_SHARD_RE.match(self._ckpt_to_model_name(n))
            if m is None:
                rest.append((n, s))
            else:
                table.setdefault(m.group("emb"), []).append((int(m.group("idx")), s, n))
        return table, rest

    def _attach_ngram_tables(self, tables):
        for emb_prefix, shards in tables.items():
            parent_name, attr = emb_prefix.rsplit(".", 1)
            parent = self._module_by_name(parent_name)
            if isinstance(getattr(parent, attr), ShardedTableEmbedding):
                continue
            shards.sort()
            if [i for i, _, _ in shards] != list(range(len(shards))):
                raise KeyError(f"{emb_prefix}: n-gram shards are not contiguous 0..{len(shards) - 1}")
            table = ShardedTableEmbedding([(path, name) for _, path, name in shards])
            expected = getattr(parent, attr)
            if (table.num_embeddings, table.embedding_dim) != tuple(expected.weight.shape):
                raise ValueError(
                    f"{emb_prefix}: shards hold {table.num_embeddings}x{table.embedding_dim}, "
                    f"the model expects {tuple(expected.weight.shape)}")
            setattr(parent, attr, table)
            # The hashing buffers are built at init, not stored; keep them with the ids.
            for key, buf in parent._buffers.items():
                if buf is not None and buf.device != torch.device(self.device):
                    parent._buffers[key] = buf.to(self.device)

    def _set_tensors(self, name_shard_pairs):
        tables, rest = self._split_ngram(name_shard_pairs)
        super()._set_tensors(rest)
        self._attach_ngram_tables(tables)

    def _offload_names_to_meta(self, name_shard_pairs):
        # The n-gram table is a host mmap that costs no device memory; keep it attached.
        _, rest = self._split_ngram(
            [p if isinstance(p, tuple) else (p, None) for p in name_shard_pairs])
        super()._offload_names_to_meta(rest)

    def save_prefixes_to_disc(self, prefixes, config=None):
        """Save non-expert prefixes, restricted to tensors the checkpoint itself has.

        The n-gram module also carries init-time hashing buffers
        (``ngram_heads_vocab_sizes`` / ``_offsets``) that are not checkpoint tensors;
        writing them would put unknown keys into the exported model.
        """
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        if not prefixes:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        ckpt_names = set(self._name_to_shard) if self._name_to_shard is not None else None
        for pfx in prefixes:
            sd = self._module_by_name(pfx).state_dict(keep_vars=True)
            to_save = {}
            for local_name, tensor in sd.items():
                name = f"{pfx}.{local_name}"
                if not (isinstance(tensor, torch.Tensor) and tensor.is_cuda):
                    continue
                if ckpt_names is not None and self._model_to_ckpt_name(name) not in ckpt_names:
                    continue
                to_save[name] = tensor.detach().cpu()
            safe_save_file(to_save, os.path.join(self.save_dir, f"{pfx.replace('.', '_')}.safetensors"))

    # ------------------------------------------------------------------ calibration inputs

    @torch.no_grad()
    def get_inputs(self, data_dict, data_loader):
        """Capture layer-0 inputs, plus every sample's token ids when a layer has PLE."""
        current_layer = self.get_layer_module(self.current_layer_idx)
        n, seqlen = data_dict['input'].shape[:2]
        if self._ple_layers and 'input_ids' not in data_dict:
            data_dict['input_ids'] = torch.zeros(n, seqlen, dtype=torch.int32)
        cache = {'index': 0}

        def store_input_hook(_, args, kwargs):
            start = cache['index'] * self.batch_size
            end = min(start + self.batch_size, n)
            data_dict['input'][start:end] = args[0] if isinstance(args, tuple) else args
            cache['index'] += 1
            for k, v in kwargs.items():
                if k == "attention_mask":
                    if v is not None:
                        self._attention_mask_1 = v[:1].detach().cpu()
                elif k == "conv_mask":
                    if v is not None:
                        raise NotImplementedError(
                            "padded calibration batches are not supported for qwen4_exp")
                elif k not in ("hidden_states", "past_key_values", "past_key_value", "ple_input_ids"):
                    self.kwargs[k] = v
            raise ValueError

        total_batches = len(data_loader)
        handle = current_layer.register_forward_pre_hook(store_input_hook, with_kwargs=True)
        for batch_idx, batch in enumerate(data_loader):
            if 'input_ids' in data_dict:
                start = batch_idx * self.batch_size
                data_dict['input_ids'][start:start + batch.shape[0]] = batch.to("cpu", torch.int32)
            try:
                self.model(batch.to(self.device))
            except ValueError:
                pass
            if self.rank == 0 and (batch_idx + 1) % max(1, total_batches // self.batch_report_divisor) == 0:
                print(f"  get_inputs: {batch_idx + 1}/{total_batches} batches", flush=True)
        handle.remove()

    def _batch_input_ids(self, data_all, start, end):
        ids = data_all.get('input_ids')
        if ids is None or self.current_layer_idx not in self._ple_layers:
            return None
        return ids[start:end]

    @torch.no_grad()
    def get_mlp_input_all(self, data_all):
        num_samples = data_all['input'].shape[0]
        for start in range(0, num_samples, self.batch_size):
            end = min(start + self.batch_size, num_samples)
            x = data_all['input'][start:end].to(self.device)
            data_all['input'][start:end] = self.get_mlp_input(
                x, self._batch_input_ids(data_all, start, end)).detach().cpu()

    # ------------------------------------------------------------------ residual stream

    def _attention_residual(self, layer, layer_idx, x, layer_inputs, input_ids=None):
        if layer.ple is not None:
            if input_ids is None:
                raise RuntimeError(
                    f"layer {layer_idx} has a per-layer n-gram embedding and needs the "
                    "batch's token ids")
            x = x + layer.ple(x, input_ids.to(x.device), None)
        mixed, hyper_input, inject = layer.attn_hyper_connection(x)
        attn_out = self._run_attention(layer, layer_idx, mixed, layer_inputs)
        return hyper_input + (attn_out.unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2)

    def _moe_hidden(self, layer, mlp_input):
        mixed, _, _ = layer.mlp_hyper_connection(mlp_input)
        return mixed

    def _moe_combine(self, layer, mlp_input, routed, hidden):
        B, T, H = routed.shape
        flat = hidden.reshape(-1, H)
        shared = F.sigmoid(layer.mlp.shared_expert_gate(flat)) * layer.mlp.shared_expert(flat)
        moe_out = routed + shared.view(B, T, H)
        _, hyper_input, inject = layer.mlp_hyper_connection(mlp_input)
        return hyper_input + (moe_out.unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2)

    def _embed_input(self, input_ids):
        return self._get_embed_tokens()(input_ids).repeat(1, 1, self.hc_count)

    def _output_hidden(self, x):
        return self._text_model.hyper_connection_mixer(x)

    def _build_layer_inputs(self, batch_size):
        inputs = super()._build_layer_inputs(batch_size)
        inputs.pop("ple_input_ids", None)
        inputs.pop("conv_mask", None)
        return inputs
