"""Qwen35MoeWrapper on a tiny synthetic Qwen3.5-MoE checkpoint (CPU, world size 1).

Multimodal checkpoint names (`model.language_model.layers.N...`) with fused experts, and a
hybrid Gated DeltaNet / gated full-attention stack. Layer by layer, the wrapper's
expert-parallel propagation must equal transformers' own forward, and so must the logits.
The wrapper is built with sdpa here because flash-attention needs a GPU.

Run:  .venv/bin/python -m pytest tests/test_qwen35_moe.py -q
"""
import json
import os

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

qwen35 = pytest.importorskip("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig  # noqa: E402

from src.models.qwen35_moe import Qwen35MoeWrapper  # noqa: E402

E, TOPK, HID, INTER, LAYERS, SEQ, BATCH = 8, 2, 64, 32, 4, 24, 2


def _text_config():
    return {
        "model_type": "qwen3_5_moe_text", "vocab_size": 128, "hidden_size": HID,
        "num_hidden_layers": LAYERS,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                            "mrope_section": [2, 1, 1], "mrope_interleaved": True},
        "linear_conv_kernel_dim": 4, "linear_key_head_dim": 16, "linear_value_head_dim": 16,
        "linear_num_key_heads": 2, "linear_num_value_heads": 4,
        "num_experts": E, "num_experts_per_tok": TOPK, "moe_intermediate_size": INTER,
        "shared_expert_intermediate_size": INTER, "max_position_embeddings": 256,
        "rms_norm_eps": 1e-6, "eos_token_id": 1, "tie_word_embeddings": False, "dtype": "float32",
    }


class _SdpaWrapper(Qwen35MoeWrapper):
    def _build_empty_model(self, cfg):
        text_cfg = cfg.text_config
        text_cfg._attn_implementation = "sdpa"
        return AutoModelForCausalLM.from_config(text_cfg, attn_implementation="sdpa").eval()


@pytest.fixture(scope="module")
def dist_world1():
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29614")
        dist.init_process_group("gloo", rank=0, world_size=1)
    yield


@pytest.fixture(scope="module")
def setup(tmp_path_factory, dist_world1):
    path = str(tmp_path_factory.mktemp("qwen35_tiny"))
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump({"model_type": "qwen3_5_moe", "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                   "tie_word_embeddings": False, "text_config": _text_config()}, f)
    cfg = Qwen3_5MoeTextConfig(**{k: v for k, v in _text_config().items() if k != "model_type"})
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    ref = qwen35.Qwen3_5MoeForCausalLM(cfg).eval()
    with torch.no_grad():
        for name, p in ref.named_parameters():
            if "norm" in name:
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            elif not name.endswith(("A_log", "dt_bias")):
                p.copy_(0.05 * torch.randn_like(p))
    tensors = {("model.language_model." + n[len("model."):] if n.startswith("model.") else n): t.contiguous()
               for n, t in ref.state_dict().items()}
    save_file(tensors, os.path.join(path, "model.safetensors"))
    with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": {n: "model.safetensors" for n in tensors}}, f)

    w = _SdpaWrapper(path, None, BATCH, SEQ, "cpu", "float32")
    w._set_tensors(w._names_from_ckpt([w._embed_prefix(), w._norm_prefix(), "lm_head"]))
    for i in range(LAYERS):
        w.move_layer_to_gpu(f"{w.layer_prefix}.{i}")
    ids = torch.randint(2, 128, (2 * BATCH, SEQ), generator=torch.Generator().manual_seed(1))
    return w, ref, ids


def test_propagation_matches_reference(setup):
    w, ref, ids = setup
    outs, logits = [[] for _ in range(LAYERS)], []
    hooks = [layer.register_forward_hook(lambda m, i, o, li=li: outs[li].append(o.detach()))
             for li, layer in enumerate(ref.model.layers)]
    with torch.no_grad():
        for start in range(0, ids.shape[0], BATCH):
            logits.append(ref(input_ids=ids[start:start + BATCH], use_cache=False).logits)
    for h in hooks:
        h.remove()

    data = {"input": torch.zeros(ids.shape[0], SEQ, HID)}
    w.current_layer_idx = 0
    w.get_inputs(data, [ids[i:i + BATCH] for i in range(0, ids.shape[0], BATCH)])
    for i in range(LAYERS):
        w.current_layer_idx = i
        w.get_layer_activations(data)
        torch.testing.assert_close(data["input"], torch.cat(outs[i]), rtol=1e-4, atol=1e-5)
    with torch.no_grad():
        ours = w.model.lm_head(w._output_hidden(data["input"]))
    torch.testing.assert_close(ours, torch.cat(logits), rtol=1e-4, atol=1e-5)


def test_mlp_input_then_output_is_the_layer(setup):
    w, _, ids = setup
    w.current_layer_idx = 3
    x = torch.randn(BATCH, SEQ, HID, generator=torch.Generator().manual_seed(2))
    layer = w.get_layer_module(3)
    with torch.no_grad():
        ours = w.get_mlp_output(w.get_mlp_input(x))
        ref = layer(x, **w._build_layer_inputs(BATCH))
    torch.testing.assert_close(ours, ref[0] if isinstance(ref, tuple) else ref, rtol=1e-4, atol=1e-5)


def test_calculate_mse_is_gate_weighted(setup):
    # p=2: every (token, expert) row weighted by its router weight squared (refine.gate_weight_exponent)
    w, _, _ = setup
    w.current_layer_idx = 3
    layer = w.get_layer_module(3)
    g = torch.Generator().manual_seed(4)
    x = torch.randn(1, 8, HID, generator=g)
    experts = layer.mlp.experts
    cw = {}
    for e in range(E):
        gu, down = experts.gate_up_proj[e], experts.down_proj[e]
        cw[f"{w.get_current_layer()}.mlp.experts.{e}"] = {
            "gate_proj": gu[:INTER] * (1 + 0.1 * torch.randn(INTER, HID, generator=g)),
            "up_proj": gu[INTER:] * (1 + 0.1 * torch.randn(INTER, HID, generator=g)),
            "down_proj": down * (1 + 0.1 * torch.randn(HID, INTER, generator=g))}

    def mlp(inp, gw, uw, dw):
        return torch.nn.functional.linear(
            torch.nn.functional.silu(torch.nn.functional.linear(inp, gw)) * torch.nn.functional.linear(inp, uw), dw)

    with torch.no_grad():
        hidden = layer.post_attention_layernorm(x).reshape(-1, HID)
        _, weight, idx = layer.mlp.gate(hidden)
        num = den = plain = 0.0
        for t in range(hidden.shape[0]):
            for k in range(TOPK):
                e = int(idx[t, k])
                gu, down = experts.gate_up_proj[e], experts.down_proj[e]
                fp = mlp(hidden[t], gu[:INTER], gu[INTER:], down)
                qw = cw[f"{w.get_current_layer()}.mlp.experts.{e}"]
                d2 = (mlp(hidden[t], qw["gate_proj"], qw["up_proj"], qw["down_proj"]) - fp).pow(2).sum()
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
