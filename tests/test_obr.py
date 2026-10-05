import copy
import torch
import torch.nn as nn

from src.config import load_config
from src.compression.initialization import OBR
from src.compression.initialization.gptq import make_quantizer, _validate_sparse_support

CFG = "configs/qwen3_30b/sgptq.yaml"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _cfg(**over):
    c = load_config(CFG)
    c.init.method = "obr"
    for k, v in over.items():
        setattr(c.init, k, v)
    return c


def _make(cin, cout, nsamples_len, cfg, seed=0, quant=False):
    g = torch.Generator(device="cpu").manual_seed(seed)
    W = torch.randn(cout, cin, generator=g) * 0.05
    lin = nn.Linear(cin, cout, bias=False).to(DEV)
    lin.weight.data = W.to(DEV)
    obr = OBR(lin, "t", cfg, DEV, torch.float32)
    if quant:
        obr.quantizer = make_quantizer(cfg)
    X = torch.randn(1, nsamples_len, cin, generator=g).to(DEV)   # (batch, L, Cin)
    obr.add_batch(X, None)
    return obr, lin, X


def _paired48_mask(rows, cols, seed):
    ## random legal paired-4:8 support: keep 2 of the 4 pairs in each block of 8
    g = torch.Generator(device="cpu").manual_seed(seed)
    nb = cols // 8
    s = torch.rand(rows, nb, 4, generator=g)
    idx = torch.topk(s, 2, dim=-1).indices
    pk = torch.zeros_like(s, dtype=torch.bool)
    pk.scatter_(-1, idx, True)
    return pk.repeat_interleave(2, dim=-1).reshape(rows, cols).to(DEV)


def test_step1_matches_lstsq_oracle():
    ## Step 1 is the exact masked least-squares reconstruction. Check it against
    ## an oracle built from the RAW activations via QR (lstsq) -- never forming or
    ## inverting H -- so this pins the sign, the H scaling, and the damping.
    cin, cout, L = 64, 8, 256
    cfg = _cfg(obr_solver="cholesky")
    obr, lin, X = _make(cin, cout, L, cfg, seed=1)

    lam = obr._prepare(cfg.init.percdamp)
    W = lin.weight.data.clone().float()
    M = _paired48_mask(cout, cin, seed=2)

    B1 = obr._cross((~M).float() * W, M.float(), lam)
    D1 = obr._solve(M.float(), B1, lam)
    Wbar = M.float() * (W + D1)

    ## objective is  dW (H + lam I) dW^T  with H = 2 X X^T (nsamples == 1), so it
    ## equals ||A dW^T||^2 for A = [sqrt(2) X ; sqrt(lam) I]
    Xd = X[0].double()                                    # (L, Cin)
    A = torch.cat([(2.0 ** 0.5) * Xd, (lam ** 0.5) * torch.eye(cin, dtype=torch.float64, device=DEV)], 0)
    Wd = W.double()

    for c in range(cout):
        keep = M[c].nonzero(as_tuple=True)[0]
        eviction = (~M[c]).nonzero(as_tuple=True)[0]
        rhs = A[:, eviction] @ Wd[c, eviction].unsqueeze(1)
        d = torch.linalg.lstsq(A[:, keep], rhs).solution.squeeze(1)
        want = Wd[c, keep] + d
        got = Wbar[c, keep].double()
        rel = (got - want).norm() / want.norm().clamp_min(1e-30)
        assert rel < 1e-4, f"row {c}: rel err {rel:.3e}"
    print("test_step1_matches_lstsq_oracle: OK")


def test_chunked_add_batch_matches():
    ## fused-expert call path: chunk_tokens bounds the fp32 copy, H is identical
    ## TF32 matmul rounds at ~1e-3, so different chunk splits accumulate different
    ## noise; disable it here to verify the chunking algebra itself is exact.
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    cin, cout, L = 64, 8, 333
    cfg = _cfg(obr_solver="cholesky")
    a, _, X = _make(cin, cout, L, cfg, seed=30)
    b = OBR(nn.Linear(cin, cout, bias=False).to(DEV), "t", cfg, DEV, torch.float32)
    b.layer.weight.data = a.layer.weight.data.clone()
    b.add_batch(X, None, chunk_tokens=37)
    hdiff = (a.H - b.H).abs().max()
    assert hdiff <= 1e-5 * a.H.abs().max(), f"chunked H diverged: {hdiff:.3e}"
    assert a.nsamples == b.nsamples
    torch.backends.cuda.matmul.allow_tf32 = tf32
    print("test_chunked_add_batch_matches: OK")


def test_step2_matches_lstsq_oracle():
    ## Step 2 is the quantization-error transfer: for a fixed error e on E2, the
    ## optimum of dW (H + lam I) dW^T with dW_E2 = -e and dW_R2 free is the least
    ## squares solution of min ||A dW_R2 - A_E2 e|| for the same A as step 1
    ## (normal equations: (H+lam I)_R2R2 d = H_R2E2 e; lam vanishes off-diagonal).
    ## Pins the sign, the H scaling, and the damping of the second compensation
    ## independently of the pruning step.
    cin, cout, L = 64, 8, 256
    cfg = _cfg(obr_solver="cholesky")
    obr, lin, X = _make(cin, cout, L, cfg, seed=8)
    lam = obr._prepare(cfg.init.percdamp)
    M = _paired48_mask(cout, cin, seed=9)

    ## arbitrary partition of the retain set, arbitrary "quantization error" on E2
    g = torch.Generator(device="cpu").manual_seed(10)
    coin = (torch.rand(cout, cin, generator=g) < 0.5).to(DEV)
    E2 = M & coin
    R2 = M & ~coin
    err = (torch.randn(cout, cin, generator=g) * 0.01).to(DEV) * E2

    B2 = obr._cross(err * E2, R2, lam)
    D2 = obr._solve(R2, B2, lam)

    Xd = X[0].double()
    A = torch.cat([(2.0 ** 0.5) * Xd, (lam ** 0.5) * torch.eye(cin, dtype=torch.float64, device=DEV)], 0)
    for c in range(cout):
        r2 = R2[c].nonzero(as_tuple=True)[0]
        e2 = E2[c].nonzero(as_tuple=True)[0]
        if r2.numel() == 0 or e2.numel() == 0:
            continue
        rhs = A[:, e2] @ err[c, e2].double().unsqueeze(1)
        want = torch.linalg.lstsq(A[:, r2], rhs).solution.squeeze(1)
        got = D2[c, r2].double()
        rel = (got - want).norm() / want.norm().clamp_min(1e-30)
        assert rel < 1e-4, f"row {c}: rel err {rel:.3e}"
    print("test_step2_matches_lstsq_oracle: OK")


def test_step2_composition_lowers_objective():
    ## the full step-2 composition as fasterquant runs it (grouped quantize ->
    ## error -> partition -> transfer): the deviation [-e on E2, +D2 on R2] must
    ## beat [-e on E2, 0] in the damped quadratic form, by optimality given e
    cin, cout, L = 256, 64, 512
    cfg = _cfg(obr_solver="cholesky")
    obr, lin, X = _make(cin, cout, L, cfg, seed=11, quant=True)
    lam = obr._prepare(cfg.init.percdamp)
    W = lin.weight.data.clone().float()
    M = _paired48_mask(cout, cin, seed=12)

    B1 = obr._cross((~M) * W, M, lam)
    Wbar = M * (W + obr._solve(M, B1, lam))

    Qtmp, _ = obr._grouped_quant(Wbar, M, 32, cin // 32, True)
    err = (Wbar - Qtmp) * M
    E2, R2 = obr._partition(M, err, lam, cfg.init.obr_alpha)
    B2 = obr._cross(err * E2, R2, lam)
    D2 = obr._solve(R2, B2, lam)

    def quad(D):
        return float((D.matmul(obr.H) * D).sum() + lam * (D * D).sum())

    without = quad(-err * E2)
    with_t = quad(-err * E2 + R2 * D2)
    assert with_t < without, f"transfer {with_t:.6e} !< no transfer {without:.6e}"
    print(f"test_step2_composition_lowers_objective: {without:.4e} -> {with_t:.4e} "
          f"({100 * (1 - with_t / without):.1f}% lower) OK")


def test_cg_agrees_with_cholesky():
    for cin, cout, L in [(128, 32, 512), (512, 128, 1024)]:
        chol = _make(cin, cout, L, _cfg(obr_solver="cholesky"), seed=3, quant=True)
        cg = _make(cin, cout, L, _cfg(obr_solver="cg"), seed=3, quant=True)
        Qc, Sc = chol[0].fasterquant(None, prunen=4, prunem=8)
        Qg, Sg = cg[0].fasterquant(None, prunen=4, prunem=8)
        rel = (Qg - Qc).norm() / Qc.norm()
        assert rel < 1e-4, f"({cin},{cout}) Q rel err {rel:.3e}"
        assert torch.allclose(Sc, Sg, rtol=1e-4, atol=0), "scales diverged"
        assert cg[0].last_cg_residual <= _cfg().init.obr_cg_tol
        print(f"test_cg_agrees_with_cholesky ({cin},{cout}): rel={rel:.2e} "
              f"iters={cg[0].last_cg_iters} resid={cg[0].last_cg_residual:.2e} OK")


def test_compensation_lowers_objective():
    ## same mask, with vs without step 1: compensation is the exact minimiser of
    ## the objective given the mask, so it can only go down
    cin, cout, L = 256, 64, 512
    cfg = _cfg(obr_solver="cg")
    obr, lin, X = _make(cin, cout, L, cfg, seed=4)
    lam = obr._prepare(cfg.init.percdamp)
    W = lin.weight.data.clone().float()
    M = _paired48_mask(cout, cin, seed=5).float()

    def obj(Q):
        d = W - Q
        return float((d.matmul(obr.H) * d).sum() + lam * (d * d).sum())

    plain = obj(M * W)
    B1 = obr._cross((1 - M) * W, M, lam)
    Wbar = M * (W + obr._solve(M, B1, lam))
    comp = obj(Wbar)
    assert comp < plain, f"compensated {comp:.6e} !< plain {plain:.6e}"
    print(f"test_compensation_lowers_objective: {plain:.4e} -> {comp:.4e} "
          f"({100 * (1 - comp / plain):.1f}% lower) OK")


def test_full_path_mask_legality_and_scales():
    cin, cout, L = 256, 64, 512
    for metric in ("wanda", "magnitude", "sparsegpt"):
        for part in ("column", "error"):
            cfg = _cfg(obr_mask_metric=metric, obr_partition=part, obr_solver="cg")
            obr, lin, X = _make(cin, cout, L, cfg, seed=6, quant=True)
            ## validate the selected support itself -- it cannot be recovered from
            ## Q's nonzeros, since a kept weight may round to FP4 0.0
            lam = obr._prepare(cfg.init.percdamp)
            W = obr.layer.weight.data.clone().float()
            M = obr._select_support(W, lam, 4, 8)
            _validate_sparse_support(M, 4, 8, "t")

            Q, S = obr.fasterquant(None, prunen=4, prunem=8)
            assert not (Q.ne(0) & ~M).any(), "nonzero outside the selected support"
            assert S.shape == (cout, cin // 32), f"scale shape {tuple(S.shape)}"
            assert torch.isfinite(S).all() and (S > 0).all(), "scales must be finite positive"
            print(f"test_full_path {metric}/{part}: loss={obr.last_gptq_loss:.4e} "
                  f"iters={obr.last_cg_iters} OK")


def test_scale_convention_flag():
    cin, cout, L = 256, 64, 512
    a, _, _ = _make(cin, cout, L, _cfg(obr_scale_on_masked=True), seed=7, quant=True)
    b, _, _ = _make(cin, cout, L, _cfg(obr_scale_on_masked=False), seed=7, quant=True)
    Qa, Sa = a.fasterquant(None, prunen=4, prunem=8)
    Qb, Sb = b.fasterquant(None, prunen=4, prunem=8)
    print(f"test_scale_convention_flag: masked loss={a.last_gptq_loss:.4e} "
          f"unmasked loss={b.last_gptq_loss:.4e} "
          f"(unmasked/masked = {b.last_gptq_loss / a.last_gptq_loss:.4f}) OK")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_step1_matches_lstsq_oracle()
    test_chunked_add_batch_matches()
    test_step2_matches_lstsq_oracle()
    test_step2_composition_lowers_objective()
    test_cg_agrees_with_cholesky()
    test_compensation_lowers_objective()
    test_full_path_mask_legality_and_scales()
    test_scale_convention_flag()
    print("\nALL OBR TESTS PASSED")
