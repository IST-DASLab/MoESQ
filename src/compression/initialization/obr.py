import logging as _logging
import math
import time
import torch
import torch.nn as nn
import transformers
import torch.distributed as dist

from .quant import NvFp4Quantizer, NVFP4_BLOCK_SIZE, quantize_nvfp4
from .gptq import _validate_sparse_support
from ..quant.nvfp4 import dense_scale_groupsize


class OBR:
    ## Optimal Brain Restoration (arXiv:2509.11177) as a one-shot joint
    ## sparse+quant baseline. Same implicit contract as GPTQ so the six model
    ## dispatch sites can construct either one.
    ##
    ## Two closed-form compensations, both of the form
    ##     dw_R = + H_RR^-1 H_RE e_E,        e_E = (value before) - (value after)
    ## The paper's Eq. 8/9 print a minus sign while also defining e_E1 = w_E1
    ## (a positive error rather than the signed perturbation dw_E = -w_E); the
    ## two conventions collide. Reference code (csguoh/OBR) uses plus.

    def __init__(self, layer, name, config, device, dtype):
        self.layer = layer
        self.name = name
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]
        self.quantizer = None
        self.H = None
        self.dead = None
        self.nsamples = 0
        self.config = config
        self.device = device
        self.dtype = dtype
        self.last_init_dense_weight = None
        self.last_init_support_mask = None
        self.last_gptq_loss = None
        self.last_cg_iters = None
        self.last_cg_residual = None
        self.last_diag_ratio = None

    def add_batch(self, inp, out, chunk_tokens=0):
        ## identical accumulator to GPTQ, including its chunk_tokens contract:
        ## fused-expert models (qwen35) pass every routed token in ONE call, so the
        ## fp32 working copy must be capped; H is the identical sum either way
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if self.H is None:
            self.H = torch.zeros((self.columns, self.columns), device=self.dev, dtype=torch.float32)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        alpha = 2.0 / self.nsamples
        num_tokens = inp.shape[1]
        step = chunk_tokens if chunk_tokens and chunk_tokens > 0 else num_tokens
        for start in range(0, num_tokens, step):
            block = inp[:, start:start + step].float()
            self.H.addmm_(block, block.t(), alpha=alpha)
            del block

    def sync_H(self, world_size):
        dist.all_reduce(self.H, op=dist.ReduceOp.SUM)
        self.H = self.H / world_size

    ## ---- damping without materializing H + lam*I -------------------------
    ## H_damped = H + lam*I, so (V @ H_damped) = (V @ H) + lam*V. Keeping the
    ## raw H intact is what lets up_proj reuse gate_proj's tensor (the six
    ## dispatch sites copy .H across) -- GPTQ's cholesky() overwrites it in
    ## bf16, OBR must not.

    def _prepare(self, percdamp):
        if self.H is None:
            _logging.getLogger(__name__).warning(
                "OBR %s: no calibration batches (H is None); using identity Hessian fallback", self.name
            )
            self.H = torch.eye(self.columns, device=self.dev, dtype=torch.float32)
        else:
            self.H = self.H.float()
        self.dead = torch.diag(self.H) == 0
        ## Repair dead input columns exactly as `GPTQ.cholesky` does.
        ##
        ## Two degenerate experts can occur under wide (384/512-way) routing:
        ##   (a) zero routed TOKENS -- add_batch never runs, H stays None, and the
        ##       `H is None` branch above substitutes the identity.
        ##   (b) identically-zero ACTIVATIONS -- add_batch does run, so H is a real
        ##       tensor of exact zeros. Then lam = percdamp*mean(diag(H)) is also 0,
        ##       Hs stays singular, and linalg.cholesky fails on "the leading minor of
        ##       order 1". Only the line below handles this case. (Tiny expert
        ##       activations, e.g. down_proj inputs ~1e-28, square to exactly 0 in fp32.)
        ##
        ## Effect for a FULLY dead expert: every diagonal entry becomes 1 while the
        ## off-diagonals are already 0, so H is exactly I -- not merely regularized.
        ## Under H = I the wanda score collapses to a constant times |W| (magnitude,
        ## not Wanda), H_RE = 0 kills the prune compensation, and quantization is RTN.
        ## GPTQ reaches the identical state by the identical repair, and both zero
        ## W's dead columns before scoring, so such an expert ships as zeros under OBR
        ## and SGPTQ alike. For a PARTIALLY dead expert H is not I and the repair only
        ## shifts lam.
        self.H[self.dead, self.dead] = 1
        hd = torch.diag(self.H)
        mean_d = torch.mean(hd)
        ## diag spread is what sets kappa(H_damped) ~ max(diag)/lam after Jacobi,
        ## hence CG's iteration count; log it so the Kimi budget is not a guess
        self.last_diag_ratio = float(hd.max() / mean_d.clamp_min(1e-30))
        return (percdamp * mean_d).item()

    def _apply(self, X, M, lam):
        ## M .* (H_damped (M .* X)) restricted to M's support
        MX = M * X
        return M * (MX.matmul(self.H) + lam * MX)

    def _cross(self, V, M_out, lam):
        ## M_out .* (H_damped V) where supp(V) is disjoint from M_out, so the
        ## lam*V term is annihilated by the outer mask and only H_RE survives
        return M_out * (V.matmul(self.H) + lam * V)

    ## ---- solvers ---------------------------------------------------------

    def _solve_cg(self, M, B, lam, max_iters, tol):
        ## H_{I_c I_c} x = B_c per row c, all rows batched: the action of a
        ## principal submatrix on a support-restricted vector is just
        ## mask-multiply-mask, so every row-solve shares one GEMM.
        eps = torch.finfo(torch.float32).tiny
        hdiag = torch.diag(self.H) + lam
        Pinv = M / hdiag.unsqueeze(0)

        X = torch.zeros_like(B)
        R = B.clone()
        Z = R * Pinv
        P = Z.clone()
        rz = (R * Z).sum(dim=1)

        bnorm = B.norm(dim=1)
        live = bnorm > 0
        if not torch.any(live):
            self.last_cg_iters = 0
            self.last_cg_residual = 0.0
            return X

        it = 0
        rel = torch.zeros_like(bnorm)
        for it in range(1, max_iters + 1):
            AP = self._apply(P, M, lam)
            pAp = (P * AP).sum(dim=1)
            alpha = torch.where(pAp > eps, rz / pAp.clamp_min(eps), torch.zeros_like(rz))
            X = X + alpha.unsqueeze(1) * P
            R = R - alpha.unsqueeze(1) * AP

            rel = torch.where(live, R.norm(dim=1) / bnorm.clamp_min(eps), torch.zeros_like(bnorm))
            if float(rel.max()) <= tol:
                break

            Z = R * Pinv
            rz_new = (R * Z).sum(dim=1)
            beta = torch.where(rz > eps, rz_new / rz.clamp_min(eps), torch.zeros_like(rz))
            P = Z + beta.unsqueeze(1) * P
            rz = rz_new

        self.last_cg_iters = it
        self.last_cg_residual = float(rel.max())
        return X

    def _solve_cholesky(self, M, B, lam):
        ## Exact per-row solve. Every row has a different mask, so H_RR is a
        ## different principal submatrix per row and there is no shared
        ## factorization -- but the factorizations can still be BATCHED, which a
        ## per-row Python loop cannot do (~108 GFLOP/s measured vs cuSOLVER's
        ## batched path). Rows are chunked to bound the gathered tensor.
        X = torch.zeros_like(B)
        rows = B.shape[0]
        counts = M.sum(dim=1)
        nkeep = int(counts.max())
        if nkeep == 0:
            self.last_cg_iters = 0
            self.last_cg_residual = 0.0
            return X
        if not bool((counts == nkeep).all()):
            ## ragged supports (only reachable without N:M) -- fall back per row
            for c in range(rows):
                idx = M[c].nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                Hs = self.H[idx][:, idx].clone()
                d = torch.arange(idx.numel(), device=B.device)
                Hs[d, d] += lam
                L = torch.linalg.cholesky(Hs)
                X[c, idx] = torch.cholesky_solve(B[c, idx].unsqueeze(1), L).squeeze(1)
            self.last_cg_iters = 0
            self.last_cg_residual = 0.0
            return X

        idx_all = M.nonzero(as_tuple=True)[1].reshape(rows, nkeep)
        chunk = max(1, int(2.0e9 // (nkeep * nkeep * 4)))
        d = torch.arange(nkeep, device=B.device)
        for c0 in range(0, rows, chunk):
            c1 = min(c0 + chunk, rows)
            ix = idx_all[c0:c1]                                      # (n, k)
            Hs = self.H[ix.unsqueeze(2), ix.unsqueeze(1)].clone()     # (n, k, k)
            Hs[:, d, d] += lam
            L = torch.linalg.cholesky(Hs)
            rhs = torch.gather(B[c0:c1], 1, ix).unsqueeze(2)          # (n, k, 1)
            sol = torch.cholesky_solve(rhs, L).squeeze(2)             # (n, k)
            X[c0:c1] = X[c0:c1].scatter(1, ix, sol)
            del Hs, L, rhs, sol
        self.last_cg_iters = 0
        self.last_cg_residual = 0.0
        return X

    def _solve(self, M, B, lam):
        if self.config.init.obr_solver == "cholesky":
            return self._solve_cholesky(M, B, lam)
        return self._solve_cg(
            M, B, lam,
            self.config.init.obr_cg_max_iters,
            self.config.init.obr_cg_tol,
        )

    ## ---- mask selection --------------------------------------------------

    def _keep_scores(self, W, lam):
        metric = self.config.init.obr_mask_metric
        if metric == "magnitude":
            return W.abs()
        if metric == "wanda":
            ## diag(H)_j = (2/N)||X_j||^2, so |W|*sqrt(diag H) is Wanda up to a
            ## per-linear constant that cancels inside a per-block top-k
            return W.abs() * torch.sqrt(torch.diag(self.H) + lam).unsqueeze(0)
        if metric == "sparseGPT" or metric == "sparsegpt":
            Hd = self.H.clone()
            d = torch.arange(self.columns, device=W.device)
            Hd[d, d] += lam
            L = torch.linalg.cholesky(Hd)
            Hi = torch.cholesky_inverse(L)
            Hi = torch.linalg.cholesky(Hi, upper=True)
            hdiag = torch.diag(Hi).clone()
            del Hd, L, Hi
            return (W ** 2) / (hdiag.unsqueeze(0) ** 2)
        raise ValueError(f"OBR {self.name}: unknown obr_mask_metric={metric!r}")

    def _select_support(self, W, lam, prunen, prunem):
        scores = self._keep_scores(W, lam)
        rows, cols = W.shape
        keep = torch.zeros_like(W, dtype=torch.bool)
        nblocks = cols // prunem

        if prunen == 4 and prunem == 8:
            ## paired 4:8 -- must emit one of the 6 PAIRED_4_8_PATTERNS, so score
            ## pairs and keep whole pairs (mirrors gptq.py:272-285)
            pair = scores.reshape(rows, nblocks, 4, 2).sum(dim=-1)
            idx = torch.topk(pair, 2, dim=-1, largest=True).indices
            pk = torch.zeros_like(pair, dtype=torch.bool)
            pk.scatter_(-1, idx, True)
            keep = pk.repeat_interleave(2, dim=-1).reshape(rows, cols)
        else:
            blk = scores.reshape(rows, nblocks, prunem)
            idx = torch.topk(blk, prunem - prunen, dim=-1, largest=True).indices
            bk = torch.zeros_like(blk, dtype=torch.bool)
            bk.scatter_(-1, idx, True)
            keep = bk.reshape(rows, cols)
        return keep

    ## ---- grouped NVFP4 ---------------------------------------------------

    def _grouped_quant(self, Wd, M, groupsize, n_groups, on_masked):
        rows, cols = Wd.shape
        scales = torch.zeros(rows, n_groups, device=Wd.device, dtype=Wd.dtype)
        Q = torch.zeros_like(Wd)
        for g in range(n_groups):
            c0 = g * groupsize
            c1 = min(c0 + groupsize, cols)
            ## scales come from the masked+compensated weight: compensation moves
            ## retained magnitudes, hence the per-group amax. on_masked=False
            ## reproduces the GPTQ baseline's unmasked-slice convention so the
            ## size of that confound can be measured.
            src = Wd[:, c0:c1] * M[:, c0:c1] if on_masked else Wd[:, c0:c1]
            self.quantizer.find_params(src, weight=True)
            scales[:, g] = self.quantizer.scale.view(rows)
            Q[:, c0:c1] = quantize_nvfp4(Wd[:, c0:c1], self.quantizer.scale)
        return Q, scales

    ## ---- step-2 partition ------------------------------------------------

    def _partition(self, M, err, lam, alpha):
        ## split the retain set R into E2 (quantization error left uncompensated)
        ## and R2 (compensated). |E2| = alpha*|R|.
        mode = self.config.init.obr_partition
        rows, cols = M.shape
        if mode == "column":
            ## faithful to the reference: E2 = retained columns with index below
            ## alpha*Cin. The paper justifies a positional split by the flat
            ## weight distribution its Hadamard rotation produces (their Fig. 6);
            ## we run no rotation, hence obr_partition: error.
            cut = int(round(alpha * cols))
            colidx = torch.arange(cols, device=M.device).unsqueeze(0)
            E2 = M & (colidx < cut)
        elif mode == "error":
            score = err.abs() * torch.sqrt(torch.diag(self.H) + lam).unsqueeze(0)
            score = torch.where(M, score, torch.full_like(score, float("-inf")))
            k = int(round(alpha * float(M[0].sum())))
            E2 = torch.zeros_like(M)
            if k > 0:
                idx = torch.topk(score, k, dim=1, largest=True).indices
                E2.scatter_(1, idx, True)
                E2 &= M
        else:
            raise ValueError(f"OBR {self.name}: unknown obr_partition={mode!r}")
        return E2, M & ~E2

    ## ---- main ------------------------------------------------------------

    def fasterquant(
        self, logging, blocksize=128, percdamp=.01, groupsize=-1, static_groups=False,
        calculate_cholesky=True, prunen=0, prunem=0
    ):
        ## calculate_cholesky is accepted for contract compatibility and ignored:
        ## damping is O(n) so up_proj just re-derives lam from the shared raw H.
        self.last_init_dense_weight = None
        self.last_init_support_mask = None

        W0 = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W0 = W0.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W0 = W0.t()
        W0 = W0.float()

        tick = time.time()
        lam = self._prepare(percdamp)
        W = W0.clone()
        W[:, self.dead] = 0

        is_nvfp4 = isinstance(self.quantizer, NvFp4Quantizer)
        is_sparse = (prunen != 0) and (prunem != 0)
        if is_sparse:
            if prunen >= prunem:
                raise ValueError(f"OBR {self.name}: expected prunen < prunem, got {prunen}:{prunem}.")
            if self.columns % prunem != 0:
                raise ValueError(
                    f"OBR {self.name}: columns={self.columns} is not divisible by prunem={prunem}."
                )
        if is_nvfp4:
            groupsize = dense_scale_groupsize(prunen, prunem, NVFP4_BLOCK_SIZE)
        n_groups = 1 if groupsize == -1 else (self.columns + groupsize - 1) // groupsize

        ## ---- mask -------------------------------------------------------
        if is_sparse:
            M = self._select_support(W, lam, prunen, prunem)
        else:
            M = torch.ones_like(W, dtype=torch.bool)

        ## ---- step 1: pruning compensation -------------------------------
        cg1 = (0, 0.0)
        if is_sparse:
            B1 = self._cross((~M) * W, M, lam)
            D1 = self._solve(M, B1, lam)
            cg1 = (self.last_cg_iters, self.last_cg_residual)
            Wbar = M * (W + D1)
            del B1, D1
        else:
            Wbar = W.clone()

        ## ---- step 2: quantization compensation --------------------------
        alpha = self.config.init.obr_alpha
        on_masked = self.config.init.obr_scale_on_masked
        cg2 = (0, 0.0)
        Wfinal = Wbar
        if self.quantizer is not None and alpha > 0:
            Qtmp, _ = self._grouped_quant(Wbar, M, groupsize, n_groups, on_masked)
            err = (Wbar - Qtmp) * M
            del Qtmp
            E2, R2 = self._partition(M, err, lam, alpha)
            B2 = self._cross(err * E2, R2, lam)
            D2 = self._solve(R2, B2, lam)
            cg2 = (self.last_cg_iters, self.last_cg_residual)
            Wfinal = Wbar + R2 * D2
            del B2, D2, err, E2, R2

        ## ---- step 3: quantize -------------------------------------------
        if self.quantizer is not None:
            Q, group_scales = self._grouped_quant(Wfinal, M, groupsize, n_groups, on_masked)
        else:
            Q = Wfinal
            group_scales = torch.zeros(self.rows, n_groups, device=W.device, dtype=W.dtype)
        Q = Q * M

        if is_sparse:
            _validate_sparse_support(M, prunen, prunem, self.name)
            pruned_nonzero = (~M) & Q.ne(0)
            if torch.any(pruned_nonzero):
                bad = pruned_nonzero.nonzero(as_tuple=False)[:8].tolist()
                raise RuntimeError(
                    f"OBR {self.name}: sparse output has nonzero pruned entries; "
                    f"bad [row, col] entries: {bad}"
                )

        ## same objective GPTQ reports (tr(dW H_damped dW^T) == ||dW L||_F^2 for
        ## L L^T = H_damped) so wandb gptq/avg_loss stays comparable across arms
        diff = W0 - Q
        obr_loss = (diff.matmul(self.H) * diff).sum() + lam * (diff * diff).sum()
        self.last_gptq_loss = obr_loss.item()

        if logging is not None:
            logging.info(
                f'Layer {self.name}: GPTQ Loss = {self.last_gptq_loss} '
                f'(OBR mask={self.config.init.obr_mask_metric} '
                f'part={self.config.init.obr_partition} '
                f'solver={self.config.init.obr_solver} '
                f'cg1={cg1[0]}/{cg1[1]:.2e} cg2={cg2[0]}/{cg2[1]:.2e} '
                f'dratio={self.last_diag_ratio:.3g} '
                f'{time.time() - tick:.1f}s)'
            )
        tol = self.config.init.obr_cg_tol
        if self.config.init.obr_solver == "cg" and max(cg1[1], cg2[1]) > tol:
            raise RuntimeError(
                f"OBR {self.name}: CG did not converge (residual "
                f"{max(cg1[1], cg2[1]):.3e} > tol {tol:.3e} after "
                f"{max(cg1[0], cg2[0])} iters). Raise init.obr_cg_max_iters."
            )

        return Q, group_scales

    def free(self):
        self.H = None
        self.dead = None
        self.last_init_dense_weight = None
        self.last_init_support_mask = None
        torch.cuda.empty_cache()
