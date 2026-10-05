"""save_model.py paired48 sparse storage (CPU only)."""
import json
import os
import tempfile

import torch

from save_model import inject_compression_config, to_paired48_sparse_storage
from src.compression import paired48_sparse


def _paired48_packed(rows, packed_k, seed=0):
    # Dense packed NVFP4 with paired-4:8 sparsity: 2 nonzero bytes out of every 4.
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(1, 256, (rows, packed_k), generator=g, dtype=torch.uint8)
    keep = torch.rand(rows, packed_k // 4, 4, generator=g).argsort(-1)[..., :2]
    mask = torch.zeros(rows, packed_k // 4, 4, dtype=torch.bool).scatter_(-1, keep, True)
    return w * mask.reshape(rows, packed_k)


def test_expert_weights_become_sparse_and_round_trip():
    pre = "model.layers.3.mlp.experts.7.gate_proj."
    w = _paired48_packed(16, 64)
    scale = torch.ones(16, 4, dtype=torch.float8_e4m3fn)
    attn = torch.randn(8, 8)
    out, n = to_paired48_sparse_storage({
        pre + "weight_packed": w,
        pre + "weight_scale": scale,
        "model.layers.3.self_attn.q_proj.weight": attn,
    })
    assert n == 1
    assert pre + "weight_packed" not in out
    values, mask = out[pre + paired48_sparse.PACKED_SUFFIX], out[pre + paired48_sparse.MASK_SUFFIX]
    assert values.shape == (16, 32) and mask.shape == (16, 8)
    rebuilt, _ = paired48_sparse.decompress_pair_bitmask(values, mask, strict=True)
    assert torch.equal(rebuilt, w)
    # Scales and non-expert tensors are kept untouched.
    assert out[pre + "weight_scale"] is scale
    assert out["model.layers.3.self_attn.q_proj.weight"] is attn


def test_non_expert_weight_packed_is_left_dense():
    key = "model.layers.3.mlp.shared_expert.down_proj.weight_packed"
    w = _paired48_packed(4, 16)
    out, n = to_paired48_sparse_storage({key: w})
    assert n == 0 and out[key] is w


def test_non_paired_weight_is_rejected():
    key = "model.layers.0.mlp.experts.0.up_proj.weight_packed"
    w = torch.full((4, 16), 7, dtype=torch.uint8)  # 4 nonzero bytes per chunk
    try:
        to_paired48_sparse_storage({key: w})
    except ValueError:
        return
    raise AssertionError("dense (non-paired) weight was accepted")


def test_config_marker():
    with tempfile.TemporaryDirectory() as d:
        for sparse in (False, True):
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump({"model_type": "qwen3_moe"}, f)
            inject_compression_config(d, 32, {"nvfp4-pack-quantized"},
                                      paired48_sparse_storage=sparse)
            with open(os.path.join(d, "config.json")) as f:
                qc = json.load(f)["quantization_config"]
            assert paired48_sparse.is_enabled(qc) == sparse


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print("ok ", fn.__name__)
    print(f"{len(fns)} passed"); sys.exit(0)
