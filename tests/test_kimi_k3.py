"""Kimi-K3 pieces that run without a GPU or the checkpoint's remote code (CPU).

  * the MXFP4 decoder equals compressed-tensors' `mxfp4-pack-quantized` decompression;
  * the export ignore list leaves exactly the routed-expert linears for K3 and none of
    the extra Linears of Qwen3.8-Flash-Next.

The wrapper itself needs the remote code, fla-core and flash-attention on a GPU; it was
checked on real layers against the remote model (layers 0, 1, 3, expert-parallel).
Run:  .venv/bin/python -m pytest tests/test_kimi_k3.py -q
"""
import re

import pytest
import torch

from save_model import _build_ignore_list
from src.models.kimi_k3 import decode_mxfp4


def test_decode_mxfp4_matches_compressed_tensors():
    mx = pytest.importorskip("compressed_tensors.compressors.mxfp4.base")
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

    g = torch.Generator().manual_seed(0)
    packed = torch.randint(0, 256, (48, 96), dtype=torch.uint8, generator=g)
    scale = torch.randint(100, 140, (48, 6), dtype=torch.uint8, generator=g)
    scheme = QuantizationScheme(targets=["Linear"], weights=QuantizationArgs(
        num_bits=4, type="float", strategy="group", group_size=32, symmetric=True, scale_dtype=torch.uint8))
    ref = mx.MXFP4PackedCompressor.decompress({"weight_packed": packed, "weight_scale": scale}, scheme)["weight"]
    assert torch.equal(decode_mxfp4(packed, scale, torch.bfloat16), ref.to(torch.bfloat16))


def _unignored(config, module_names):
    ignore = _build_ignore_list(config)
    regexes = [re.compile(p[3:]) for p in ignore if p.startswith("re:")]
    literals = {p for p in ignore if not p.startswith("re:")}
    return {n for n in module_names if n not in literals and not any(r.match(n) for r in regexes)}


def test_kimi_k3_export_quantizes_only_routed_experts():
    config = {"model_type": "kimi_k3", "architectures": ["KimiK3ForConditionalGeneration"],
              "vision_config": {}, "text_config": {
                  "model_type": "kimi_linear", "num_experts": 896, "num_shared_experts": 2,
                  "first_k_dense_replace": 1, "routed_expert_hidden_size": 3584,
                  "attn_res_block_size": 12, "num_nextn_predict_layers": 0}}
    L = "language_model.model.layers"
    linears = [
        "language_model.lm_head", "mm_projector.proj.0",
        f"{L}.0.mlp.gate_proj", f"{L}.0.mlp.up_proj", f"{L}.0.mlp.down_proj",
        f"{L}.1.self_attn.q_proj", f"{L}.1.self_attn.f_b_proj", f"{L}.3.self_attn.kv_b_proj",
        f"{L}.1.self_attention_res_proj", f"{L}.1.mlp_res_proj", "language_model.model.output_attn_res_proj",
        f"{L}.1.block_sparse_moe.shared_experts.gate_proj", f"{L}.1.block_sparse_moe.routed_expert_down_proj",
        f"{L}.1.block_sparse_moe.routed_expert_up_proj",
        f"{L}.1.block_sparse_moe.experts.7.w1", f"{L}.1.block_sparse_moe.experts.7.w2",
        f"{L}.1.block_sparse_moe.experts.7.w3",
    ]
    assert _unignored(config, linears) == {f"{L}.1.block_sparse_moe.experts.7.w{i}" for i in (1, 2, 3)}


def test_qwen38_flash_next_export_ignores_its_extra_linears():
    config = {"model_type": "qwen4_exp", "vision_config": {}, "text_config": {
        "model_type": "qwen4_exp_text", "num_experts": 512, "shared_expert_intermediate_size": 640,
        "layer_types": ["linear_attention"], "hc_count": 4, "ple_layer_ids": [2],
        "mtp_num_hidden_layers": 1}}
    L = "model.language_model.layers"
    linears = [
        "lm_head", f"{L}.0.linear_attn.in_proj_qkv", f"{L}.3.self_attn.indexer.index_qk_proj",
        f"{L}.0.attn_hyper_connection.input_mix_weight_down", f"{L}.0.mlp_hyper_connection.block_inject_weight",
        "model.language_model.hyper_connection_mixer.input_mix_weight_up",
        f"{L}.1.ple.key_proj", f"{L}.1.ple.value_proj", f"{L}.0.mlp.gate",
        f"{L}.0.mlp.shared_expert.up_proj", f"{L}.0.mlp.shared_expert_gate", "mtp.fc_hidden",
        "model.visual.blocks.0.attn.qkv",
    ]
    assert _unignored(config, linears) == set()
