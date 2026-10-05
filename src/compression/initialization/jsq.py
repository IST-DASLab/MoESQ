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


class JSQ:
    ## JSQ (Guo et al., ICML 2024; github.com/uanu2002/JSQ) as a one-shot joint
    ## sparse+quant baseline: SAR mask + RTN NVFP4, no weight reconstruction --
    ## faithful to the released artifact, which is RTN-based and has no GPTQ path.
    ##
    ## Adaptations vs the release:
    ##   * activation editing = the paper's Eq. 5 range clamp, applied to the
    ##     calibration statistics only, per add_batch call (the release clips a
    ##     count quantile of |entries|; NEITHER clips anything at inference)
    ##   * the range term A (Eq. 3) is exact over this linear's calibration
    ##     tokens, streamed; the release evaluates it on a sample-averaged
    ##     activation, which attenuates the outliers the metric exists to see
    ##   * S = I/mean(I) + lambda * A/mean(A) -- unit-free version of the paper's
    ##     lambda=2 operating point ("I and lambda*A in similar magnitude");
    ##     lambda=0 degenerates to pair-summed Wanda
    ##   * paired-4:8 by pair-summed S (serving format; all arms share this)
    ##   * no SmoothQuant migration, no annealing search (their Tab. 3 bounds
    ##     the search at 0.25 wiki2; migration is never ablated in the paper)
    ##
    ## The range term needs THIS linear's output, so up_proj cannot inherit
    ## gate_proj's statistics the way GPTQ/OBR share H -- the dispatch must hook
    ## up_proj too when init.method == "jsq" (wired in qwen3_moe only for now;
    ## fasterquant hard-fails if the hook never ran).

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
        self.a_tokens = 0
        ## Set by the model dispatch for an expert it owns but routed no TOKENS to.
        ## Distinguishes that from a dispatch that never wired jsq at all -- the range
        ## guard below must stay fatal for the second and must not be for the first.
        ##
        ## This is condition (a) of obr.py's two: no add_batch, so no H and no range.
        ## Condition (b), an expert fed identically-zero activations, does NOT come
        ## here -- add_batch ran, so amax/amin exist (both 0) and a_tokens > 0. It is
        ## already safe: arange is 0 and arange.mean().clamp_min(eps) keeps the
        ## normalization finite, so the SAR term contributes 0 instead of NaN.
        self.no_tokens = False
        self.amax = None
        self.amin = None
        self.config = config
        self.device = device
        self.dtype = dtype
        self.last_init_dense_weight = None
        self.last_init_support_mask = None
        self.last_gptq_loss = None

    def _edit_bounds(self, T):
        ## editing touches only what gets ACCUMULATED (H, ranges); the model's
        ## forward and the weights being compressed see unclipped activations.
        ## Bounds come from the WHOLE call (the paper's F is the layer's activation
        ## tensor), computed on the input dtype so a fused-expert call never
        ## materializes an fp32 copy just to find two scalars. Shape-agnostic.
        r = self.config.init.jsq_edit_r
        if r <= 0:
            return None
        if self.config.init.jsq_edit_mode == "range":
            lo, hi = T.min().float(), T.max().float()
            span = hi - lo
            if span <= 0:
                return None
            return (lo + r * span, hi - r * span)
        ## "quantile": the release's clip_matrix abs path (clip_h fraction of entries)
        k = max(1, int(r * T.numel()))
        th = T.abs().flatten().topk(k).values[-1].float()
        return (-th, th)

    def _edit(self, X):
        b = self._edit_bounds(X)
        return X if b is None else X.clamp(b[0], b[1])

    def add_batch(self, inp, out, chunk_tokens=0, h=True):
        ## `out` is accepted for call-site symmetry and unused: Y is recomputed from
        ## the CLIPPED input so the range term sees the same edited stream as H.
        ## chunk_tokens caps the fp32 working copy under GPTQ's exact contract --
        ## fused-expert models (qwen35) pass every routed token in one call. Clip
        ## bounds are fixed per call before chunking, so the edited stream is
        ## identical either way.
        ## h=False accumulates ONLY the range statistics: up_proj shares gate's
        ## input, its H is overwritten by gate's after gate's fasterquant in every
        ## dispatch, and a wasted H is 205 MB per Kimi expert.
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if h:
            if self.H is None:
                self.H = torch.zeros((self.columns, self.columns), device=self.dev, dtype=torch.float32)
            self.H *= self.nsamples / (self.nsamples + tmp)
            self.nsamples += tmp
            alpha = 2.0 / self.nsamples
        bounds = self._edit_bounds(inp)
        num_tokens = inp.shape[1]
        step = chunk_tokens if chunk_tokens and chunk_tokens > 0 else num_tokens
        for start in range(0, num_tokens, step):
            X = inp[:, start:start + step].t().float()
            if bounds is not None:
                X.clamp_(bounds[0], bounds[1])
            if h:
                Xh = X.t()
                self.H.addmm_(Xh, Xh.t(), alpha=alpha)
                del Xh
            self._accumulate_range(X)
            del X

    def _accumulate_range(self, X):
        ## running per-(row, col) max/min of y_row(t) - x_col(t) * W[row, col]:
        ## the exact post-removal output range of Eq. 3, since output j couples
        ## to input i only through W[j, i]
        cap = self.config.init.jsq_token_cap
        if cap > 0:
            left = cap - self.a_tokens
            if left <= 0:
                return
            X = X[:left]  # first-N-tokens cap; biased toward early sequences, disclosed
        if self.amax is None:
            ## bfloat16 accumulators halve the footprint at Kimi scale (fp32 is
            ## ~34 GB/rank at ws=4); the running max loses only bf16 rounding
            rdt = torch.bfloat16 if self.config.init.jsq_range_dtype == "bfloat16" else torch.float32
            self.amax = torch.full((self.rows, self.columns), float("-inf"), device=self.dev, dtype=rdt)
            self.amin = torch.full((self.rows, self.columns), float("inf"), device=self.dev, dtype=rdt)
        W = self.layer.weight.data.float()
        cchunk = 128
        tchunk = max(1, int(1.6e8 // (self.rows * cchunk)))
        for t0 in range(0, X.shape[0], tchunk):
            Xt = X[t0:t0 + tchunk]
            Y = Xt.matmul(W.t())
            for c0 in range(0, self.columns, cchunk):
                c1 = min(c0 + cchunk, self.columns)
                cand = Y.unsqueeze(2) - Xt[:, c0:c1].unsqueeze(1) * W[:, c0:c1].unsqueeze(0)
                self.amax[:, c0:c1] = torch.maximum(self.amax[:, c0:c1], cand.amax(dim=0).to(self.amax.dtype))
                self.amin[:, c0:c1] = torch.minimum(self.amin[:, c0:c1], cand.amin(dim=0).to(self.amin.dtype))
        self.a_tokens += X.shape[0]

    def sync_H(self, world_size):
        dist.all_reduce(self.H, op=dist.ReduceOp.SUM)
        self.H = self.H / world_size
        if self.amax is not None:
            dist.all_reduce(self.amax, op=dist.ReduceOp.MAX)
            dist.all_reduce(self.amin, op=dist.ReduceOp.MIN)

    def _select_support(self, scores, prunen, prunem):
        rows, cols = scores.shape
        nblocks = cols // prunem
        if prunen == 4 and prunem == 8:
            ## paired 4:8 -- must emit one of the 6 PAIRED_4_8_PATTERNS, so score
            ## pairs and keep whole pairs (mirrors gptq.py / obr.py)
            pair = scores.reshape(rows, nblocks, 4, 2).sum(dim=-1)
            idx = torch.topk(pair, 2, dim=-1, largest=True).indices
            pk = torch.zeros_like(pair, dtype=torch.bool)
            pk.scatter_(-1, idx, True)
            return pk.repeat_interleave(2, dim=-1).reshape(rows, cols)
        blk = scores.reshape(rows, nblocks, prunem)
        idx = torch.topk(blk, prunem - prunen, dim=-1, largest=True).indices
        bk = torch.zeros_like(blk, dtype=torch.bool)
        bk.scatter_(-1, idx, True)
        return bk.reshape(rows, cols)

    def _grouped_quant(self, Wd, M, groupsize, n_groups):
        ## RTN per dense group, scales found on the MASKED slice (same
        ## convention as obr_scale_on_masked: true; the GPTQ control's
        ## unmasked-slice quirk is a disclosed confound, obr plan §4.2b)
        rows, cols = Wd.shape
        scales = torch.zeros(rows, n_groups, device=Wd.device, dtype=Wd.dtype)
        Q = torch.zeros_like(Wd)
        for g in range(n_groups):
            c0 = g * groupsize
            c1 = min(c0 + groupsize, cols)
            self.quantizer.find_params(Wd[:, c0:c1] * M[:, c0:c1], weight=True)
            scales[:, g] = self.quantizer.scale.view(rows)
            Q[:, c0:c1] = quantize_nvfp4(Wd[:, c0:c1], self.quantizer.scale)
        return Q, scales

    def fasterquant(
        self, logging, blocksize=128, percdamp=.01, groupsize=-1, static_groups=False,
        calculate_cholesky=True, prunen=0, prunem=0
    ):
        self.last_init_dense_weight = None
        self.last_init_support_mask = None

        W0 = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W0 = W0.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W0 = W0.t()
        W0 = W0.float()

        tick = time.time()
        if self.H is None:
            _logging.getLogger(__name__).warning(
                "JSQ %s: no calibration batches (H is None); using identity Hessian fallback", self.name
            )
            self.H = torch.eye(self.columns, device=self.dev, dtype=torch.float32)
        else:
            self.H = self.H.float()
        self.dead = torch.diag(self.H) == 0
        lam = (percdamp * torch.mean(torch.diag(self.H))).item()
        W = W0.clone()
        W[:, self.dead] = 0

        is_nvfp4 = isinstance(self.quantizer, NvFp4Quantizer)
        is_sparse = (prunen != 0) and (prunem != 0)
        if is_sparse:
            if prunen >= prunem:
                raise ValueError(f"JSQ {self.name}: expected prunen < prunem, got {prunen}:{prunem}.")
            if self.columns % prunem != 0:
                raise ValueError(
                    f"JSQ {self.name}: columns={self.columns} is not divisible by prunem={prunem}."
                )
        if is_nvfp4:
            groupsize = dense_scale_groupsize(prunen, prunem, NVFP4_BLOCK_SIZE)
        n_groups = 1 if groupsize == -1 else (self.columns + groupsize - 1) // groupsize

        ## ---- SAR mask -----------------------------------------------------
        if is_sparse:
            lam_n = self.config.init.jsq_lambda_norm
            if lam_n > 0 and (self.amax is None or self.a_tokens == 0):
                if self.no_tokens:
                    ## Zero routed tokens: there is no output range to measure, so the
                    ## SAR term is undefined rather than missing. Degrade this expert to
                    ## lam_n=0 on the identity Hessian the H branch already substituted.
                    ## NOT "pair-summed Wanda": with H = I, diag(H) is uniform, so
                    ## |W|*sqrt(diag(H)+lam) collapses to a constant times |W| and the
                    ## score is pair-summed MAGNITUDE. lam_n=0 is genuine Wanda only on an
                    ## expert with a real H, where diag(H) varies. Same score SGPTQ/GPTQ
                    ## reaches through its own dead-column repair, so the arms stay
                    ## comparable -- and since W's dead columns are zeroed before scoring,
                    ## a fully dead expert ships as zeros under either method regardless
                    ## of which tie the mask breaks.
                    ## An expert that never fires on the calibration set cannot be scored
                    ## by any activation-derived metric; this is not an approximation of
                    ## one. Kimi-K2.5 hits this at layer 54 (expert 156).
                    _logging.getLogger(__name__).warning(
                        "JSQ %s: expert received zero routed tokens; SAR range term "
                        "disabled for it (lam_n 0, identity H)", self.name
                    )
                    lam_n = 0.0
                else:
                    raise RuntimeError(
                        f"JSQ {self.name}: no range statistics accumulated and the dispatch "
                        "did not mark this expert as unrouted -- this model's dispatch never "
                        "hooked this linear (up_proj H-sharing skips the hook). A dispatch "
                        "that wires jsq must call mark_no_tokens() on the handles of experts "
                        "it routed nothing to."
                    )
            eps = torch.finfo(torch.float32).tiny
            wanda = W.abs() * torch.sqrt(torch.diag(self.H).clamp_min(0) + lam).unsqueeze(0)
            S = wanda / wanda.mean().clamp_min(eps)
            if lam_n > 0:
                arange = (self.amax - self.amin).float().clamp_min(0)
                arange = torch.where(torch.isfinite(arange), arange, torch.zeros_like(arange))
                S = S + lam_n * arange / arange.mean().clamp_min(eps)
            ## a dead input never changes the output, so Eq. 3 assigns it the FULL
            ## output range (high salience) -- keeping a no-op weight. The paper
            ## never sees dead columns; zero them out of the ranking instead.
            S = S.masked_fill(self.dead.unsqueeze(0), 0.0)
            M = self._select_support(S, prunen, prunem)
        else:
            M = torch.ones_like(W, dtype=torch.bool)
        ## side channel for tests/debugging only -- refine+jsq is rejected by the
        ## validator, so nothing downstream consumes this
        self.last_init_support_mask = M

        ## ---- prune by zeroing + RTN quantize (no reconstruction) ----------
        if self.quantizer is not None:
            Q, group_scales = self._grouped_quant(W, M, groupsize, n_groups)
        else:
            Q = W.clone()
            group_scales = torch.zeros(self.rows, n_groups, device=W.device, dtype=W.dtype)
        Q = Q * M

        if is_sparse:
            _validate_sparse_support(M, prunen, prunem, self.name)
            pruned_nonzero = (~M) & Q.ne(0)
            if torch.any(pruned_nonzero):
                bad = pruned_nonzero.nonzero(as_tuple=False)[:8].tolist()
                raise RuntimeError(
                    f"JSQ {self.name}: sparse output has nonzero pruned entries; "
                    f"bad [row, col] entries: {bad}"
                )

        ## same objective GPTQ/OBR report, so wandb gptq/avg_loss stays comparable
        diff = W0 - Q
        jsq_loss = (diff.matmul(self.H) * diff).sum() + lam * (diff * diff).sum()
        self.last_gptq_loss = jsq_loss.item()

        if logging is not None:
            logging.info(
                f'Layer {self.name}: GPTQ Loss = {self.last_gptq_loss} '
                f'(JSQ lam_n={self.config.init.jsq_lambda_norm} '
                f'edit={self.config.init.jsq_edit_mode}/{self.config.init.jsq_edit_r} '
                f'a_tokens={self.a_tokens} '
                f'{time.time() - tick:.1f}s)'
            )

        return Q, group_scales

    def mark_no_tokens(self):
        self.no_tokens = True

    def free(self):
        self.H = None
        self.dead = None
        self.amax = None
        self.amin = None
        self.no_tokens = False
        self.last_init_dense_weight = None
        self.last_init_support_mask = None
        torch.cuda.empty_cache()
