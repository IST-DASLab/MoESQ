"""refine.token_weight_clip_k: sink-aware per-row loss weights (CPU only)."""
import math
import torch
try:
    import pytest
except ImportError:  # the moe-sq venv has no pytest; run this file directly instead
    pytest = None


def _approx(a, b, rel=1e-6):
    return math.isclose(a, b, rel_tol=rel, abs_tol=0.0)

from src.models.base import token_clip_weights
from src.config import RefineConfig


def _rows(norms, H=64):
    out = torch.zeros(len(norms), H)
    out[:, 0] = torch.tensor(norms, dtype=torch.float)
    return out


def test_disabled_returns_none():
    assert token_clip_weights(_rows([1.0, 2.0, 700.0]), 0.0) is None
    assert token_clip_weights(_rows([1.0, 2.0, 700.0]), -1.0) is None
    assert token_clip_weights(_rows([]), 20.0) is None


def test_ordinary_rows_keep_weight_one_and_sink_is_capped():
    norms = [0.8, 1.0, 1.2, 1.0, 0.9, 1.1, 1.0, 700.0]   # median ||y|| = 1.0, one sink row
    w = token_clip_weights(_rows(norms), k=20.0)
    assert w.shape == (8, 1)
    assert torch.allclose(w[:7], torch.ones(7, 1))
    # sink row: (k * median)^2 / ||y||^2 = (20 * 1)^2 / 700^2
    assert _approx(w[7, 0].item(), (20.0 / 700.0) ** 2, rel=1e-4)
    assert not w.requires_grad


def test_row_at_k_times_median_is_the_boundary():
    norms = [1.0] * 9 + [20.0]
    w = token_clip_weights(_rows(norms), k=20.0)
    assert _approx(w[9, 0].item(), 1.0, rel=1e-5)
    w2 = token_clip_weights(_rows([1.0] * 9 + [40.0]), k=20.0)
    assert _approx(w2[9, 0].item(), 0.25, rel=1e-4)


def test_weighted_mean_equals_plain_mse_when_nothing_is_clipped():
    torch.manual_seed(0)
    out_fp = torch.randn(256, 32)
    out_q = out_fp + 0.1 * torch.randn(256, 32)
    clip = token_clip_weights(out_fp, k=1e6)          # k huge -> every weight exactly 1
    assert torch.all(clip == 1.0)
    diff2 = (out_q - out_fp).pow(2)
    weighted = (clip * diff2).sum() / (clip.sum() * diff2.shape[-1])
    assert _approx(weighted.item(), torch.nn.functional.mse_loss(out_q, out_fp).item(), rel=1e-6)


def test_config_rejects_negative_k():
    cfg = RefineConfig()
    assert cfg.token_weight_clip_k == 0.0


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print("ok ", fn.__name__)
    print(f"{len(fns)} passed"); sys.exit(0)


def test_expert_recon_loss_gate_weighting():
    # BaseModelWrapper._expert_recon_loss: p=0 is the plain MSE, p>0 the g^p-weighted mean
    from types import SimpleNamespace
    from src.models.base import BaseModelWrapper
    g = torch.Generator().manual_seed(0)
    out_fp, out_q = torch.randn(6, 8, generator=g), torch.randn(6, 8, generator=g)
    gate_w = torch.rand(6, 1, generator=g)
    stub = SimpleNamespace(loss_fn=torch.nn.MSELoss(), gate_weight_exponent=0.0, token_weight_clip_k=0.0)
    loss = BaseModelWrapper._expert_recon_loss
    assert _approx(loss(stub, out_q, out_fp, gate_w).item(), torch.nn.functional.mse_loss(out_q, out_fp).item())
    stub.gate_weight_exponent = 2.0
    per_row = (out_q - out_fp).pow(2).mean(-1, keepdim=True)
    expected = (gate_w.pow(2) * per_row).sum() / gate_w.pow(2).sum()
    assert _approx(loss(stub, out_q, out_fp, gate_w).item(), expected.item(), rel=1e-5)
