import torch
import torch.nn as nn

from src.config import load_config
from src.compression.initialization import JSQ
from src.compression.initialization.gptq import make_quantizer, _validate_sparse_support

CFG = "configs/qwen3_30b/jsq.yaml"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _cfg(**over):
    c = load_config(CFG)
    for k, v in over.items():
        setattr(c.init, k, v)
    return c


def _make(cin, cout, cfg, seed=0, quant=False):
    g = torch.Generator(device="cpu").manual_seed(seed)
    W = torch.randn(cout, cin, generator=g) * 0.05
    lin = nn.Linear(cin, cout, bias=False).to(DEV)
    lin.weight.data = W.to(DEV)
    jsq = JSQ(lin, "t", cfg, DEV, torch.float32)
    if quant:
        jsq.quantizer = make_quantizer(cfg)
    return jsq, lin, g


def test_range_term_matches_naive():
    ## streaming amax/amin over ragged batches must equal the reference trick:
    ## zero input column i, recompute outputs, take per-output ranges -- for
    ## every (j, i) at once, over the concatenation of all batches
    cin, cout = 64, 8
    cfg = _cfg(jsq_edit_r=0.0)
    jsq, lin, g = _make(cin, cout, cfg, seed=1)

    batches = [torch.randn(1, L, cin, generator=g).to(DEV) for L in (96, 257, 31)]
    for X in batches:
        jsq.add_batch(X, None)

    Xall = torch.cat([b[0] for b in batches], 0).float()
    W = lin.weight.data.float()
    naive = torch.zeros(cout, cin, device=DEV)
    for i in range(cin):
        Wz = W.clone()
        Wz[:, i] = 0
        out = Xall @ Wz.t()
        naive[:, i] = out.max(dim=0).values - out.min(dim=0).values

    got = jsq.amax - jsq.amin
    rel = (got - naive).abs().max() / naive.abs().max()
    assert rel < 1e-4, f"range term rel err {rel:.3e}"
    assert jsq.a_tokens == Xall.shape[0]
    print(f"test_range_term_matches_naive: rel={rel:.2e} tokens={jsq.a_tokens} OK")


def test_editing_modes():
    cin, cout = 32, 4
    g = torch.Generator(device="cpu").manual_seed(2)
    X = (torch.randn(500, cin, generator=g) * torch.tensor([10.0] + [1.0] * (cin - 1))).to(DEV)

    jsq, _, _ = _make(cin, cout, _cfg(jsq_edit_r=0.0), seed=2)
    assert torch.equal(jsq._edit(X), X), "r=0 must be a no-op"

    r = 0.1
    jsq, _, _ = _make(cin, cout, _cfg(jsq_edit_r=r, jsq_edit_mode="range"), seed=2)
    E = jsq._edit(X)
    lo, hi = X.min(), X.max()
    span = hi - lo
    assert E.min() >= lo + r * span - 1e-5 and E.max() <= hi - r * span + 1e-5
    inside = (X > lo + r * span) & (X < hi - r * span)
    assert torch.equal(E[inside], X[inside]), "values inside the clamp window must be untouched"

    r = 0.01
    jsq, _, _ = _make(cin, cout, _cfg(jsq_edit_r=r, jsq_edit_mode="quantile"), seed=2)
    E = jsq._edit(X)
    k = max(1, int(r * X.numel()))
    th = X.abs().flatten().topk(k).values[-1]
    assert E.abs().max() <= th + 1e-6
    assert torch.equal(E[X.abs() < th], X[X.abs() < th])
    print("test_editing_modes: OK")


def test_lambda_zero_reduces_to_wanda():
    ## lam_n = 0 must select exactly the pair-summed Wanda support
    cin, cout, L = 256, 64, 512
    cfg = _cfg(jsq_lambda_norm=0.0, jsq_edit_r=0.0)
    jsq, lin, g = _make(cin, cout, cfg, seed=3, quant=True)
    jsq.add_batch(torch.randn(1, L, cin, generator=g).to(DEV), None)

    Q, S = jsq.fasterquant(None, prunen=4, prunem=8)

    W = lin.weight.data.float()
    lam = cfg.init.percdamp * torch.mean(torch.diag(jsq.H))
    wanda = W.abs() * torch.sqrt(torch.diag(jsq.H).clamp_min(0) + lam).unsqueeze(0)
    pair = wanda.reshape(cout, cin // 8, 4, 2).sum(dim=-1)
    idx = torch.topk(pair, 2, dim=-1, largest=True).indices
    pk = torch.zeros_like(pair, dtype=torch.bool)
    pk.scatter_(-1, idx, True)
    expected = pk.repeat_interleave(2, dim=-1).reshape(cout, cin)

    assert torch.equal(jsq.last_init_support_mask, expected), "lam_n=0 mask != pair-summed Wanda"
    print("test_lambda_zero_reduces_to_wanda: OK")


def test_full_path_mask_legality_and_scales():
    cin, cout, L = 256, 64, 512
    for mode, lam_n, r in [("range", 1.0, 5e-5), ("range", 1.0, 0.0),
                           ("quantile", 1.0, 0.01), ("range", 0.0, 0.0)]:
        cfg = _cfg(jsq_edit_mode=mode, jsq_lambda_norm=lam_n, jsq_edit_r=r)
        jsq, lin, g = _make(cin, cout, cfg, seed=4, quant=True)
        for Lb in (L, 128):
            jsq.add_batch(torch.randn(1, Lb, cin, generator=g).to(DEV), None)
        Q, S = jsq.fasterquant(None, prunen=4, prunem=8)
        M = jsq.last_init_support_mask
        _validate_sparse_support(M, 4, 8, "t")
        assert not (Q.ne(0) & ~M).any(), "nonzero outside the selected support"
        assert S.shape == (cout, cin // 32), f"scale shape {tuple(S.shape)}"
        assert torch.isfinite(S).all() and (S > 0).all(), "scales must be finite positive"
        assert jsq.last_gptq_loss is not None and jsq.last_gptq_loss > 0
        print(f"test_full_path {mode}/lam={lam_n}/r={r}: loss={jsq.last_gptq_loss:.4e} "
              f"a_tokens={jsq.a_tokens} OK")


def test_chunked_add_batch_matches():
    ## the fused-expert call path (qwen35): one call with every routed token,
    ## TF32 matmul rounds at ~1e-3, so different chunk splits accumulate different
    ## noise; disable it here to verify the chunking algebra itself is exact.
    ## chunk_tokens bounding the fp32 copy. H and both range accumulators must
    ## match the unchunked call -- clip bounds are fixed per call, so the edited
    ## stream is identical; running max/min are order-independent.
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    cin, cout, L = 64, 8, 333
    for r, mode in [(0.0, "range"), (5e-5, "range"), (0.01, "quantile")]:
        cfg = _cfg(jsq_edit_r=r, jsq_edit_mode=mode)
        a, lin_a, g = _make(cin, cout, cfg, seed=20)
        b, lin_b, _ = _make(cin, cout, cfg, seed=20)
        X = torch.randn(1, L, cin, generator=g).to(DEV)
        a.add_batch(X, None)
        b.add_batch(X, None, chunk_tokens=37)
        hdiff = (a.H - b.H).abs().max()
        assert hdiff <= 1e-5 * a.H.abs().max(), f"H diverged (r={r},{mode}): {hdiff:.3e}"
        assert torch.equal(a.amax, b.amax) and torch.equal(a.amin, b.amin), f"ranges diverged (r={r},{mode})"
        assert a.nsamples == b.nsamples and a.a_tokens == b.a_tokens
    torch.backends.cuda.matmul.allow_tf32 = tf32
    print("test_chunked_add_batch_matches: OK")


def test_h_false_accumulates_range_only():
    ## the up_proj path: range statistics without a wasted Hessian
    cin, cout, L = 64, 8, 128
    cfg = _cfg(jsq_edit_r=0.0)
    a, _, g = _make(cin, cout, cfg, seed=21)
    b, _, _ = _make(cin, cout, cfg, seed=21)
    X = torch.randn(1, L, cin, generator=g).to(DEV)
    a.add_batch(X, None)
    b.add_batch(X, None, h=False)
    assert b.H is None and b.nsamples == 0, "h=False must not touch H"
    assert torch.equal(a.amax, b.amax) and torch.equal(a.amin, b.amin)
    assert b.a_tokens == L
    print("test_h_false_accumulates_range_only: OK")


def test_bf16_range_dtype():
    cin, cout, L = 128, 16, 512
    f32, _, g = _make(cin, cout, _cfg(jsq_edit_r=0.0, jsq_range_dtype="float32"), seed=22)
    b16, _, _ = _make(cin, cout, _cfg(jsq_edit_r=0.0, jsq_range_dtype="bfloat16"), seed=22)
    X = torch.randn(1, L, cin, generator=g).to(DEV)
    f32.add_batch(X, None)
    b16.add_batch(X, None)
    assert b16.amax.dtype == torch.bfloat16
    ref = (f32.amax - f32.amin)
    got = (b16.amax - b16.amin).float()
    rel = ((got - ref).abs() / ref.abs().clamp_min(1e-6)).max()
    assert rel < 0.02, f"bf16 range accumulators off by {rel:.3f}"
    print(f"test_bf16_range_dtype: max rel dev {rel:.4f} OK")


def test_unhooked_linear_fails_loudly():
    ## a model whose dispatch never hooked this linear (the up_proj H-sharing
    ## pattern) must fail, not silently fall back to a Wanda-only mask
    cin, cout = 64, 8
    jsq, lin, g = _make(cin, cout, _cfg(), seed=5, quant=True)
    jsq.H = torch.eye(cin, device=DEV)  # H "inherited", ranges never accumulated
    try:
        jsq.fasterquant(None, prunen=4, prunem=8)
    except RuntimeError as e:
        assert "no range statistics" in str(e)
        print("test_unhooked_linear_fails_loudly: OK")
        return
    raise AssertionError("expected RuntimeError for missing range statistics")


def test_token_cap():
    cin, cout = 64, 8
    cfg = _cfg(jsq_token_cap=100, jsq_edit_r=0.0)
    jsq, lin, g = _make(cin, cout, cfg, seed=6)
    for Lb in (64, 64, 64):
        jsq.add_batch(torch.randn(1, Lb, cin, generator=g).to(DEV), None)
    assert jsq.a_tokens == 100, f"cap not honored: {jsq.a_tokens}"
    assert jsq.nsamples == 3, "H accumulation must not be capped"
    print("test_token_cap: OK")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_range_term_matches_naive()
    test_editing_modes()
    test_lambda_zero_reduces_to_wanda()
    test_chunked_add_batch_matches()
    test_h_false_accumulates_range_only()
    test_bf16_range_dtype()
    test_full_path_mask_legality_and_scales()
    test_unhooked_linear_fails_loudly()
    test_token_cap()
    print("\nALL JSQ TESTS PASSED")
