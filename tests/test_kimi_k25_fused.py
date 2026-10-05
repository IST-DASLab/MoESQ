"""KimiK25FusedWrapper on a tiny synthetic Kimi-K2.5 checkpoint (CPU, world size 1).

The checkpoint mimics the real one: checkpoint names (`language_model.model.layers.N...`),
per-expert INT4 pack-quantized experts, an fp32 router bias. What must hold:
  * every per-expert weight lands in its fused slice, gate in the first half of
    `gate_up_proj`, and the wrapper reads back exactly what compressed-tensors decodes;
  * the router helper reproduces the remote-code `MoEGate` (noaux_tc) selection and weights;
  * the expert-parallel MoE block equals transformers' own `DeepseekV3MoE.forward`;
  * shards and trainer keys stay in checkpoint names.

Needs transformers >= 5.14 (native Kimi_K25 classes); skipped otherwise.
Run:  .venv/bin/python -m pytest tests/test_kimi_k25_fused.py -q
"""
import json
import os

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from src.models.kimi_k25_fused import KimiK25FusedWrapper, native_kimi_k25_available

pytestmark = pytest.mark.skipif(not native_kimi_k25_available(),
                                reason="needs transformers with native Kimi_K25 classes")

E, TOPK, HID, INTER, LAYERS, GROUP = 8, 2, 64, 32, 3, 32
WEIGHT_ARGS = {"num_bits": 4, "type": "int", "symmetric": True, "strategy": "group",
               "group_size": GROUP, "dynamic": False, "observer": "minmax", "observer_kwargs": {},
               "actorder": None, "block_structure": None}


def _config():
    return {
        "architectures": ["KimiK25ForConditionalGeneration"],
        "model_type": "kimi_k25",
        "dtype": "float32",
        "tie_word_embeddings": False,
        "pad_token_id": 0,
        "vision_config": {"num_hidden_layers": 1, "hidden_size": 32, "intermediate_size": 64,
                          "num_attention_heads": 2, "patch_size": 2, "pos_emb_height": 4,
                          "pos_emb_width": 4},
        "projection_hidden_size": 32,
        "text_config": {
            "model_type": "kimi_k2", "vocab_size": 128, "hidden_size": HID,
            "intermediate_size": 96, "moe_intermediate_size": INTER,
            "num_hidden_layers": LAYERS, "first_k_dense_replace": 1,
            "num_attention_heads": 2, "num_key_value_heads": 2,
            "q_lora_rank": 32, "kv_lora_rank": 32, "qk_nope_head_dim": 16,
            "qk_rope_head_dim": 8, "v_head_dim": 16,
            "n_routed_experts": E, "n_shared_experts": 1, "num_experts_per_tok": TOPK,
            "n_group": 1, "topk_group": 1, "norm_topk_prob": True, "routed_scaling_factor": 2.5,
            "scoring_func": "sigmoid", "topk_method": "noaux_tc",
            "max_position_embeddings": 256, "rms_norm_eps": 1e-5, "hidden_act": "silu",
            "rope_theta": 50000.0,
            "rope_scaling": {"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 64,
                             "beta_fast": 32.0, "beta_slow": 1.0, "mscale": 1.0,
                             "mscale_all_dim": 1.0},
            "quantization_config": {
                "quant_method": "compressed-tensors", "format": "pack-quantized",
                "quantization_status": "compressed", "kv_cache_scheme": None,
                "ignore": ["re:.*self_attn.*", "re:.*shared_experts.*", "re:.*lm_head.*"],
                "config_groups": {"group_0": {"targets": ["Linear"], "input_activations": None,
                                              "output_activations": None, "weights": WEIGHT_ARGS}},
            },
        },
    }


@pytest.fixture(scope="module")
def dist_world1():
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29611")
        dist.init_process_group("gloo", rank=0, world_size=1)
    yield


def _write_checkpoint(path):
    """Checkpoint-named tensors for the tiny model; experts per-expert and INT4 packed."""
    from accelerate import init_empty_weights
    from src.compression.ct_compat import PackedQuantizationCompressor, QuantizationArgs
    from transformers import Kimi_K25Config, Kimi_K25ForConditionalGeneration

    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(_config(), f)
    with init_empty_weights():
        model = Kimi_K25ForConditionalGeneration._from_config(
            Kimi_K25Config.from_pretrained(path), attn_implementation="sdpa")
    g = torch.Generator().manual_seed(0)
    compressor, qargs = PackedQuantizationCompressor(), QuantizationArgs(**WEIGHT_ARGS)
    tensors, decoded = {}, {}
    named = list(model.named_parameters()) + list(model.named_buffers())
    for name, p in named:
        if not name.startswith(("model.language_model.", "lm_head.")) or name.endswith("inv_freq"):
            continue
        ckpt = KimiK25FusedWrapper._model_to_ckpt_name(name)
        if name.endswith(("mlp.experts.gate_up_proj", "mlp.experts.down_proj")):
            if name.endswith("down_proj"):
                projs = {"down_proj": (HID, INTER)}
            else:
                projs = {"gate_proj": (INTER, HID), "up_proj": (INTER, HID)}
            experts = ckpt.rsplit(".", 1)[0]
            for e in range(E):
                for proj, shape in projs.items():
                    w = torch.randn(shape, generator=g) * 0.05
                    scale = (w.reshape(shape[0], -1, GROUP).abs().amax(-1) / 7).to(torch.bfloat16)
                    packed = compressor.compress_weight(weight=w, scale=scale, quantization_args=qargs)
                    base = f"{experts}.{e}.{proj}"
                    tensors[f"{base}.weight_packed"] = packed["weight_packed"]
                    tensors[f"{base}.weight_scale"] = scale
                    tensors[f"{base}.weight_shape"] = packed["weight_shape"]
                    decoded[base] = compressor.decompress_weight(
                        {k: tensors[f"{base}.{k}"] for k in ("weight_packed", "weight_scale", "weight_shape")},
                        qargs).to(torch.float32)
            continue
        if name.endswith("e_score_correction_bias"):
            tensors[ckpt] = torch.randn(p.shape, generator=g) * 0.1
        elif "norm" in name:
            tensors[ckpt] = 1.0 + 0.1 * torch.randn(p.shape, generator=g)
        else:
            tensors[ckpt] = torch.randn(p.shape, generator=g) * 0.05
    save_file(tensors, os.path.join(path, "model.safetensors"))
    return decoded


@pytest.fixture(scope="module")
def wrapper(tmp_path_factory, dist_world1):
    path = str(tmp_path_factory.mktemp("kimi_tiny"))
    decoded = _write_checkpoint(path)
    w = KimiK25FusedWrapper(path, None, 2, 16, "cpu", "float32")
    for i in range(LAYERS):
        w.move_layer_to_gpu(f"{w.layer_prefix}.{i}")
    return w, decoded, path


def _remote_moe_gate(gate_weight, bias, cfg, hidden):
    """The checkpoint's remote-code MoEGate.forward (noaux_tc, sigmoid), transcribed."""
    n = hidden.shape[0]
    logits = torch.nn.functional.linear(hidden.float(), gate_weight.float())
    scores = logits.sigmoid()
    choice = scores + bias.unsqueeze(0)
    group_scores = choice.view(n, cfg.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=cfg.topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
    score_mask = group_mask.unsqueeze(-1).expand(n, cfg.n_group, E // cfg.n_group).reshape(n, -1)
    tmp = choice.masked_fill(~score_mask.bool(), 0.0)
    _, idx = torch.topk(tmp, k=cfg.num_experts_per_tok, dim=-1, sorted=False)
    w = scores.gather(1, idx)
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    return idx, w * cfg.routed_scaling_factor


def test_names_round_trip(wrapper):
    w, _, _ = wrapper
    for ckpt in ("language_model.model.layers.1.mlp.gate.weight", "language_model.lm_head.weight"):
        assert w._model_to_ckpt_name(w._ckpt_to_model_name(ckpt)) == ckpt
    assert w._ckpt_to_model_name("language_model.lm_head.weight") == "lm_head.weight"
    assert w.layer_prefix == "language_model.model.layers"


def test_fused_slices_hold_decoded_experts(wrapper):
    w, decoded, _ = wrapper
    for li in range(1, LAYERS):
        experts = w.get_layer_module(li).mlp.experts
        assert tuple(experts.gate_up_proj.shape) == (E, 2 * INTER, HID)
        for e in range(E):
            gate, up, down = w._expert_weights(experts, e)
            base = f"language_model.model.layers.{li}.mlp.experts.{e}"
            torch.testing.assert_close(gate, decoded[f"{base}.gate_proj"], rtol=0, atol=0)
            torch.testing.assert_close(up, decoded[f"{base}.up_proj"], rtol=0, atol=0)
            torch.testing.assert_close(down, decoded[f"{base}.down_proj"], rtol=0, atol=0)
            torch.testing.assert_close(experts.gate_up_proj[e, :INTER], decoded[f"{base}.gate_proj"],
                                       rtol=0, atol=0)


def test_router_matches_remote_moe_gate(wrapper):
    w, _, _ = wrapper
    layer = w.get_layer_module(1)
    hidden = torch.randn(4, 16, HID, generator=torch.Generator().manual_seed(1))
    idx, weight = w._route(layer, hidden)
    ref_idx, ref_w = _remote_moe_gate(layer.mlp.gate.weight, layer.mlp.gate.e_score_correction_bias,
                                      w.text_config, hidden.reshape(-1, HID))
    assert torch.equal(idx, ref_idx)
    torch.testing.assert_close(weight, ref_w, rtol=0, atol=0)


def test_expert_parallel_block_matches_native_moe(wrapper):
    w, _, _ = wrapper
    w.current_layer_idx = 2
    layer = w.get_layer_module(2)
    x = torch.randn(2, 16, HID, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        ours = w.run_expert_parallel(x)
        native = layer.mlp(layer.post_attention_layernorm(x)) + x
    torch.testing.assert_close(ours, native, rtol=1e-5, atol=1e-6)


def test_update_compressed_weights_writes_fused_slice(wrapper):
    w, _, _ = wrapper
    name = "language_model.model.layers.1.mlp.experts.3.up_proj"
    q = torch.full((INTER, HID), 0.25)
    w.update_compressed_weights(name, (q, torch.ones(INTER, HID // GROUP)))
    _, up, _ = w._expert_weights(w.get_layer_module(1).mlp.experts, 3)
    assert torch.equal(up, q) and torch.equal(w.temp_weights[f"{name}.Q"], q)
    w.temp_weights.clear()


def test_offload_and_reload(wrapper):
    w, decoded, _ = wrapper
    name = f"{w.layer_prefix}.1"
    w.offload_to_meta(name)
    experts = w.get_layer_module(1).mlp.experts
    assert experts.gate_up_proj.device.type == "meta"
    assert w.get_layer_module(1).mlp.gate.e_score_correction_bias.device.type == "meta"
    w.move_layer_to_gpu(name)
    gate, _, _ = w._expert_weights(experts, 5)
    torch.testing.assert_close(gate, decoded["language_model.model.layers.1.mlp.experts.5.gate_proj"],
                               rtol=0, atol=0)


def test_non_mlp_shard_keys_are_checkpoint_names(wrapper):
    # save_prefixes_to_disc writes `<prefix>.<state_dict key>` for each non-MLP prefix.
    w, _, path = wrapper
    ckpt = load_file(os.path.join(path, "model.safetensors"))
    for pfx in w._layer_prefixes(f"{w.layer_prefix}.1")["non_mlp"]:
        written = {f"{pfx}.{k}" for k in w._module(pfx).state_dict()}
        assert written == {k for k in ckpt if k.startswith(pfx + ".")}


def test_calculate_mse_is_gate_weighted(wrapper):
    # p=2: every (token, expert) row weighted by its router weight squared (refine.gate_weight_exponent)
    w, _, _ = wrapper
    w.current_layer_idx = 2
    layer = w.get_layer_module(2)
    g = torch.Generator().manual_seed(4)
    x = torch.randn(2, 8, HID, generator=g)
    experts = layer.mlp.experts
    cw = {}
    for e in range(E):
        gate, up, down = w._expert_weights(experts, e)
        cw[f"{w.get_current_layer()}.mlp.experts.{e}"] = {
            p: t * (1 + 0.1 * torch.randn(t.shape, generator=g))
            for p, t in (("gate_proj", gate), ("up_proj", up), ("down_proj", down))}

    def mlp(inp, gw, uw, dw):
        return torch.nn.functional.linear(
            torch.nn.functional.silu(torch.nn.functional.linear(inp, gw)) * torch.nn.functional.linear(inp, uw), dw)

    with torch.no_grad():
        hidden = layer.post_attention_layernorm(x).reshape(-1, HID)
        idx, weight = w._route(layer, hidden)
        num = den = plain = 0.0
        for t in range(hidden.shape[0]):
            for k in range(TOPK):
                e = int(idx[t, k])
                fp = mlp(hidden[t], *w._expert_weights(experts, e))
                q = mlp(hidden[t], *(cw[f"{w.get_current_layer()}.mlp.experts.{e}"][p]
                                     for p in ("gate_proj", "up_proj", "down_proj")))
                d2 = (q - fp).pow(2).sum()
                num += weight[t, k].float() ** 2 * d2
                den += weight[t, k].float() ** 2
                plain += d2
        rows = hidden.shape[0] * TOPK
    try:
        w.gate_weight_exponent = 2.0
        weighted = w.calculate_mse(x, cw, validation=True)
        w.gate_weight_exponent = 0.0
        unweighted = w.calculate_mse(x, cw, validation=True)
    finally:
        w.gate_weight_exponent = 0.0
    assert weighted == pytest.approx(float(num / (den * HID)), rel=1e-5)
    assert unweighted == pytest.approx(float(plain / (rows * HID)), rel=1e-5)
    assert weighted != pytest.approx(unweighted, rel=1e-3)
