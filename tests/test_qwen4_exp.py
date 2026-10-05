"""Qwen4ExpWrapper (Qwen3.8-Flash-Next) on a tiny synthetic checkpoint (CPU, world size 1).

The checkpoint mimics the real one: multimodal names (`model.language_model.layers.N...`),
fused experts, the per-layer n-gram table split into row shards, and a hybrid stack of
Gated DeltaNet and Qwen Sparse Attention layers. What must hold:
  * layer by layer, the wrapper's propagation equals transformers' own forward, including
    the hyper-connection streams, the n-gram layer and the final mixer + lm_head (the
    reference gets the same QSA indexer, so this checks the wiring);
  * the vectorized QSA indexer selects what the reference loop selects, up to the choice
    among blocks tied at the budget boundary, which torch.topk leaves unspecified;
  * the memory-mapped n-gram table returns the checkpoint rows;
  * non-expert shards only hold checkpoint tensors.

Needs transformers >= 5.16 (qwen4_exp); skipped otherwise.
Run:  .venv/bin/python -m pytest tests/test_qwen4_exp.py -q
"""
import json
import os

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

qwen4 = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")

from src.models.qwen4_exp import (  # noqa: E402
    Qwen4ExpWrapper, ShardedTableEmbedding, _patch_qsa_indexer, causal_qsa_selection, qsa_block_scores,
)

E, TOPK, HID, INTER, HC, LAYERS, SEQ, BATCH = 8, 2, 64, 32, 4, 4, 32, 2
LAYER_TYPES = ["linear_attention", "linear_attention", "linear_attention", "qwen_sparse_attention"]


def _text_config():
    return {
        "model_type": "qwen4_exp_text", "vocab_size": 128, "hidden_size": HID,
        "num_hidden_layers": LAYERS, "layer_types": LAYER_TYPES, "full_attention_interval": 4,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
        "partial_rotary_factor": 0.25, "attention_bias": False, "attention_dropout": 0.0,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                            "mrope_section": [2, 1, 1], "mrope_interleaved": True},
        "linear_conv_kernel_dim": 4, "linear_key_head_dim": 16, "linear_value_head_dim": 16,
        "linear_num_key_heads": 2, "linear_num_value_heads": 4, "output_gate_type": "sigmoid",
        "mamba_ssm_dtype": "float32",
        "indexer_n_heads": 2, "indexer_kv_heads": 1, "indexer_head_dim": 16,
        "indexer_budget": 8, "indexer_compress_ratio": 4,
        "hc_count": HC, "hc_lowrank": 16,
        "ple_layer_ids": [2], "ple_embed_dim": HID, "ple_conv_kernel_size": 4, "ngram_size": 3,
        "heads_per_ngram": 2, "ngram_vocab_size_base": 50, "make_ngram_vocab_size_divisible_by": 8,
        "split_ngram_parts": 2,
        "num_experts": E, "num_experts_per_tok": TOPK, "moe_intermediate_size": INTER,
        "shared_expert_intermediate_size": INTER, "norm_topk_prob": True,
        "hidden_act": "silu", "rms_norm_eps": 1e-6, "max_position_embeddings": 256,
        "eos_token_id": 1, "bos_token_id": 1, "pad_token_id": None, "tie_word_embeddings": False,
        "mtp_num_hidden_layers": 0, "dtype": "float32",
    }


@pytest.fixture(scope="module")
def dist_world1():
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29613")
        dist.init_process_group("gloo", rank=0, world_size=1)
    yield


def _write_checkpoint(path):
    """Save a randomly initialised reference model under the real checkpoint's names."""
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump({"model_type": "qwen4_exp", "architectures": ["Qwen4ExpForConditionalGeneration"],
                   "tie_word_embeddings": False, "text_config": _text_config()}, f)
    cfg = qwen4.Qwen4ExpTextConfig(**{k: v for k, v in _text_config().items() if k != "model_type"})
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    ref = qwen4.Qwen4ExpForCausalLM(cfg).eval()
    with torch.no_grad():
        for name, p in ref.named_parameters():
            if "norm" in name:
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            elif name.endswith(("A_log", "dt_bias")):
                continue
            else:
                p.copy_(0.05 * torch.randn_like(p))
    tensors = {}
    for name, t in ref.state_dict().items():
        if name.endswith(("ngram_heads_vocab_sizes", "ngram_heads_offsets")):
            continue  # init-time buffers, absent from the real checkpoint
        ckpt = "model.language_model." + name[len("model."):] if name.startswith("model.") else name
        if name.endswith("ngram_embedding.weight"):
            for i, part in enumerate(t.chunk(cfg.split_ngram_parts, dim=0)):
                tensors[ckpt.replace(".weight", f".shard_{i}.weight")] = part.contiguous()
            continue
        tensors[ckpt] = t.contiguous()
    names = sorted(tensors)
    half = len(names) // 2
    weight_map = {}
    for idx, part in enumerate((names[:half], names[half:])):
        fname = f"model-0000{idx + 1}-of-00002.safetensors"
        save_file({n: tensors[n] for n in part}, os.path.join(path, fname))
        weight_map.update({n: fname for n in part})
    with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)
    return ref, tensors


@pytest.fixture(scope="module")
def setup(tmp_path_factory, dist_world1):
    path = str(tmp_path_factory.mktemp("qwen4_tiny"))
    ref, tensors = _write_checkpoint(path)
    for layer in ref.model.layers:
        if hasattr(layer, "self_attn"):
            _patch_qsa_indexer(layer.self_attn.indexer)
    w = Qwen4ExpWrapper(path, None, BATCH, SEQ, "cpu", "float32")
    w._set_tensors(w._names_from_ckpt([w._embed_prefix(), w._norm_prefix(), "lm_head"]))
    for i in range(LAYERS):
        w.move_layer_to_gpu(f"{w.layer_prefix}.{i}")
    ids = torch.randint(2, 128, (2 * BATCH, SEQ), generator=torch.Generator().manual_seed(1))
    ids[0, SEQ // 2] = 1  # an eos inside a sample exercises the n-gram segment reset
    return w, ref, tensors, ids, path


def _reference_layer_outputs(ref, ids):
    """Per-layer outputs and logits of the reference, run in the wrapper's batch size.

    Same batch shapes on both sides keep GEMM reduction orders equal, so near-tied
    top-k choices (router, QSA indexer) cannot flip between the two.
    """
    outs, logits = [[] for _ in ref.model.layers], []
    hooks = [layer.register_forward_hook(lambda m, i, o, li=li: outs[li].append(o.detach()))
             for li, layer in enumerate(ref.model.layers)]
    with torch.no_grad():
        for start in range(0, ids.shape[0], BATCH):
            logits.append(ref(input_ids=ids[start:start + BATCH], use_cache=False).logits)
    for h in hooks:
        h.remove()
    return [torch.cat(o) for o in outs], torch.cat(logits)


def test_propagation_matches_reference(setup):
    w, ref, _, ids, _ = setup
    ref_outs, ref_logits = _reference_layer_outputs(ref, ids)
    data = {"input": torch.zeros(ids.shape[0], SEQ, w.activation_hidden_size)}
    loader = [ids[i:i + BATCH] for i in range(0, ids.shape[0], BATCH)]
    w.current_layer_idx = 0
    w.get_inputs(data, loader)
    assert torch.equal(data["input_ids"], ids.to(torch.int32))
    for i in range(LAYERS):
        w.current_layer_idx = i
        w.get_layer_activations(data)
        torch.testing.assert_close(data["input"], ref_outs[i], rtol=1e-4, atol=1e-5)
    with torch.no_grad():
        logits = w.model.lm_head(w._output_hidden(data["input"]))
    torch.testing.assert_close(logits, ref_logits, rtol=1e-4, atol=1e-5)


def test_mlp_input_then_output_is_the_layer(setup):
    w, ref, _, ids, _ = setup
    ref_outs, _ = _reference_layer_outputs(ref, ids[:BATCH])
    li = 1  # the n-gram layer
    w.current_layer_idx = li
    x = ref_outs[li - 1]
    with torch.no_grad():
        mlp_in = w.get_mlp_input(x, ids[:BATCH])
        out = w.get_mlp_output(mlp_in)
    torch.testing.assert_close(out, ref_outs[li], rtol=1e-4, atol=1e-5)


def test_vectorized_qsa_indexer_matches_reference(setup):
    w, _, _, _, _ = setup
    indexer = w.get_layer_module(3).self_attn.indexer
    reference = type(indexer).forward.__get__(indexer)
    r, budget = indexer.compress_ratio, indexer.block_topk
    g = torch.Generator().manual_seed(3)
    hidden = torch.randn(BATCH, SEQ, HID, generator=g)
    pos = torch.arange(SEQ).view(1, 1, -1).expand(3, BATCH, -1)
    pe = w._text_model.rotary_emb(hidden, pos)
    mask = torch.ones(SEQ, SEQ, dtype=torch.bool).tril().expand(BATCH, 1, SEQ, SEQ)
    with torch.no_grad():
        ours = causal_qsa_selection(indexer, hidden, pe, mask)[:, 0]
        theirs = reference(hidden, pe, mask, None)[:, 0]
    assert ours.shape == theirs.shape
    n_tied = 0
    for b in range(BATCH):
        scores = qsa_block_scores(indexer, hidden, pe, b)
        for s in range(SEQ):
            nc = (s + 1) // r
            # the tail after the complete blocks is always kept
            assert ours[b, s, r * nc:s + 1].all() and theirs[b, s, r * nc:s + 1].all()
            assert not ours[b, s, s + 1:].any()
            if torch.equal(ours[b, s], theirs[b, s]):
                continue
            # otherwise both must be a valid top-k of the same scores, split on a tie
            ob = ours[b, s, :r * nc:r].nonzero().flatten()
            tb = theirs[b, s, :r * nc:r].nonzero().flatten()
            assert len(ob) == len(tb) == min(budget, nc)
            torch.testing.assert_close(scores[s, ob].sort().values, scores[s, tb].sort().values,
                                       rtol=1e-6, atol=1e-7)
            n_tied += 1
    assert n_tied < BATCH * SEQ // 4
    # the budget really binds: late queries see fewer tokens than the causal prefix
    assert ours[:, -1].sum(-1).max() < SEQ


def test_ngram_table_is_the_checkpoint(setup):
    w, ref, _, _, _ = setup
    emb = w.get_layer_module(1).ple.ple_embedding
    assert isinstance(emb.ngram_embedding, ShardedTableEmbedding)
    full = ref.model.layers[1].ple.ple_embedding.ngram_embedding.weight
    rows = torch.randint(0, full.shape[0], (5, 7))
    torch.testing.assert_close(emb.ngram_embedding(rows), full[rows], rtol=0, atol=0)
    assert emb.layer_multipliers.dtype == torch.long
    torch.testing.assert_close(emb.layer_multipliers, ref.model.layers[1].ple.ple_embedding.layer_multipliers)


def test_non_mlp_shards_hold_only_checkpoint_tensors(setup, tmp_path):
    w, _, tensors, _, _ = setup
    w.save_dir = str(tmp_path)
    w._set_tensors(w._names_from_ckpt(w._layer_prefixes(f"{w.layer_prefix}.1")["non_mlp"]))
    # save_prefixes_to_disc keeps device tensors only; on CPU mark them as such.
    prefixes = w._layer_prefixes(f"{w.layer_prefix}.1")["non_mlp"]
    orig = torch.Tensor.is_cuda
    try:
        torch.Tensor.is_cuda = property(lambda self: True)
        w.save_prefixes_to_disc(prefixes)
    finally:
        torch.Tensor.is_cuda = orig
    written = set()
    for pfx in prefixes:
        written |= set(load_file(os.path.join(str(tmp_path), f"{pfx.replace('.', '_')}.safetensors")))
    ckpt_names = {w._model_to_ckpt_name(n) for n in written}
    assert ckpt_names <= set(tensors)
    assert any(".ple." in n for n in written)
    assert not any("ngram" in n and "multipliers" not in n for n in written)


def test_calculate_mse_is_gate_weighted(setup):
    # the router sees the hyper-connection mix; p=2 weights each (token, expert) row by g^2
    w, _, _, _, _ = setup
    w.current_layer_idx = 3
    layer = w.get_layer_module(3)
    g = torch.Generator().manual_seed(4)
    x = torch.randn(1, 8, w.activation_hidden_size, generator=g)
    experts = layer.mlp.experts
    cw = {}
    for e in range(E):
        gu, down = experts.gate_up_proj[e], experts.down_proj[e]
        cw[f"{w.get_current_layer()}.mlp.experts.{e}"] = {
            "gate_proj": gu[:INTER] * (1 + 0.1 * torch.randn(INTER, HID, generator=g)),
            "up_proj": gu[INTER:] * (1 + 0.1 * torch.randn(INTER, HID, generator=g)),
            "down_proj": down * (1 + 0.1 * torch.randn(HID, INTER, generator=g))}
    silu, lin = torch.nn.functional.silu, torch.nn.functional.linear
    with torch.no_grad():
        hidden = w._moe_hidden(layer, x).reshape(-1, HID)
        _, weight, idx = layer.mlp.gate(hidden)
        num = den = 0.0
        for t in range(hidden.shape[0]):
            for k in range(TOPK):
                e = int(idx[t, k])
                gu, down = experts.gate_up_proj[e], experts.down_proj[e]
                qw = cw[f"{w.get_current_layer()}.mlp.experts.{e}"]
                fp = lin(silu(lin(hidden[t], gu[:INTER])) * lin(hidden[t], gu[INTER:]), down)
                q = lin(silu(lin(hidden[t], qw["gate_proj"])) * lin(hidden[t], qw["up_proj"]), qw["down_proj"])
                num += weight[t, k].float() ** 2 * (q - fp).pow(2).sum()
                den += weight[t, k].float() ** 2
    try:
        w.gate_weight_exponent = 2.0
        weighted = w.calculate_mse(x, cw, validation=True)
    finally:
        w.gate_weight_exponent = 0.0
    assert weighted == pytest.approx(float(num / (den * HID)), rel=1e-5)
