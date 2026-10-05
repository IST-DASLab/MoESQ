import logging
import math
import time
import torch
import torch.nn as nn
import transformers
import torch.distributed as dist

from .quant import (
    Quantizer, NvFp4Quantizer, NVFP4_BLOCK_SIZE,
    quantize, quantize_nvfp4, FP4_CODEBOOK, FP4_MAX,
)
from ..quant.nvfp4 import dense_scale_groupsize


DEBUG = False


def make_quantizer(config):
    if config.compression.quant_type == "nvfp4":
        q = NvFp4Quantizer()
        q.configure(mse=True)
    else:
        q = Quantizer()
        q.configure(
            config.init.wbits, perchannel=True, sym=config.init.sym,
            mse=True, trits=config.init.trits,
        )
    return q


def _effective_groupsize(quantizer, config):
    if isinstance(quantizer, NvFp4Quantizer):
        return dense_scale_groupsize(
            config.compression.prunen, config.compression.prunem, NVFP4_BLOCK_SIZE
        )
    return config.compression.groupsize


def _validate_sparse_support(support_mask, prunen, prunem, name):
    if support_mask.shape[1] % prunem != 0:
        raise ValueError(
            f"GPTQ {name}: columns={support_mask.shape[1]} is not divisible by "
            f"prunem={prunem}."
        )

    expected_nnz = prunem - prunen
    block_mask = support_mask.reshape(support_mask.shape[0], -1, prunem)
    block_nnz = block_mask.sum(dim=-1)
    valid = block_nnz == expected_nnz

    if prunen == 4 and prunem == 8:
        pair_view = block_mask.reshape(support_mask.shape[0], -1, 4, 2)
        pair_nnz = pair_view.sum(dim=-1)
        valid_pairs = ((pair_nnz == 0) | (pair_nnz == 2)).all(dim=-1)
        kept_pairs = (pair_nnz == 2).sum(dim=-1) == 2
        valid = valid & valid_pairs & kept_pairs

    if not torch.all(valid):
        bad = (~valid).nonzero(as_tuple=False)[:8].tolist()
        raise RuntimeError(
            f"GPTQ {name}: internal support mask is not valid {expected_nnz}:{prunem} "
            f"sparsity; bad [row, block] entries: {bad}"
        )


def rtn_quantize(layer, config, device, dtype):
    W = layer.weight.data.clone().float()
    rows, columns = W.shape
    quantizer = make_quantizer(config)
    groupsize = _effective_groupsize(quantizer, config)

    n_groups = (columns + groupsize - 1) // groupsize if groupsize > 0 else 1
    group_scales = torch.zeros(rows, n_groups, device=W.device, dtype=W.dtype)
    Q = torch.zeros_like(W)

    for g in range(n_groups):
        c_start = g * groupsize
        c_end = min(c_start + groupsize, columns)
        W_group = W[:, c_start:c_end]

        quantizer.find_params(W_group, weight=True)
        group_scales[:, g] = quantizer.scale.view(rows)
        Q[:, c_start:c_end] = quantizer.quantize(W_group)

    return Q.to(device), group_scales.to(device)


def random_quantize(layer, config, device, dtype):
    rows, columns = layer.weight.shape
    quantizer = make_quantizer(config)
    groupsize = _effective_groupsize(quantizer, config)

    W = torch.empty(rows, columns, device='cpu', dtype=torch.float32)
    nn.init.kaiming_uniform_(W, a=math.sqrt(5))

    n_groups = (columns + groupsize - 1) // groupsize if groupsize > 0 else 1
    group_scales = torch.zeros(rows, n_groups, device=W.device, dtype=W.dtype)
    Q = torch.zeros_like(W)

    for g in range(n_groups):
        c_start = g * groupsize
        c_end = min(c_start + groupsize, columns)
        W_group = W[:, c_start:c_end]

        quantizer.find_params(W_group, weight=True)
        group_scales[:, g] = quantizer.scale.view(rows)
        Q[:, c_start:c_end] = quantizer.quantize(W_group)

    return Q.to(device), group_scales.to(device)


class GPTQ:

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

    def add_batch(self, inp, out, chunk_tokens=0):
        ## `out` is accepted for call-site symmetry with the forward hooks and is unused.
        ##
        ## chunk_tokens caps the fp32 working copy. Hook-driven models call this once per
        ## calibration micro-batch, so inp is small and chunking is a no-op (the default).
        ## Fused-expert models (qwen35) cannot hook per-expert nn.Linears, so they call
        ## this ONCE with every token routed to an expert -- under routing skew that is
        ## millions of tokens and inp.float() alone is tens of GiB. Chunk below the
        ## nsamples bookkeeping so H is the identical sum either way: splitting the call
        ## instead would advance nsamples per chunk and silently re-weight H.
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
        alpha = 2.0 / self.nsamples          # sqrt(2/n) applied to both factors == 2/n on the product
        num_tokens = inp.shape[1]
        step = chunk_tokens if chunk_tokens and chunk_tokens > 0 else num_tokens
        for start in range(0, num_tokens, step):
            block = inp[:, start:start + step].float()
            self.H.addmm_(block, block.t(), alpha=alpha)   # in-place: no [C, C] temporary
            del block

    def cholesky(self, percdamp=.01):
        if self.H is None:
            logging.getLogger(__name__).warning(
                "GPTQ %s: no calibration batches (H is None); using identity Hessian fallback", self.name
            )
            self.H = torch.eye(self.columns, device=self.dev, dtype=torch.float32)
        else:
            self.H = self.H.float()
        self.dead = torch.diag(self.H) == 0
        self.H[self.dead, self.dead] = 1

        damp = percdamp * torch.mean(torch.diag(self.H))
        diag = torch.arange(self.columns, device=self.dev)
        self.H[diag, diag] += damp
        self.H = torch.linalg.cholesky(self.H)
        self.H = self.H.to(self.dtype)

    def sync_H(self, world_size):
        dist.all_reduce(self.H, op=dist.ReduceOp.SUM)
        self.H = self.H / world_size

    def fasterquant(
        self, logging, blocksize=128, percdamp=.01, groupsize=-1, static_groups=False, calculate_cholesky=True, prunen=0, prunem=0
    ):
        self.last_init_dense_weight = None
        self.last_init_support_mask = None

        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        if calculate_cholesky:
            self.cholesky(percdamp)
        W[:, self.dead] = 0

        tick = time.time()

        if self.quantizer and not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        rows = W.shape[0]
        is_nvfp4 = isinstance(self.quantizer, NvFp4Quantizer)
        is_sparse = (prunen != 0) and (prunem != 0) ## n:m sparsity only.
        track_dense_support = is_sparse
        if is_sparse:
            if prunen >= prunem:
                raise ValueError(f"GPTQ {self.name}: expected prunen < prunem, got {prunen}:{prunem}.")
            if self.columns % prunem != 0:
                raise ValueError(
                    f"GPTQ {self.name}: columns={self.columns} is not divisible by prunem={prunem}."
                )
            if blocksize % prunem != 0:
                raise ValueError(
                    f"GPTQ {self.name}: blocksize={blocksize} must be divisible by prunem={prunem} "
                    "for structured sparsity masks."
                )
        if is_nvfp4:
            groupsize = dense_scale_groupsize(prunen, prunem, NVFP4_BLOCK_SIZE)
        if groupsize == -1:
            n_groups = 1
        else:
            n_groups = (self.columns + groupsize - 1) // groupsize
        group_scales = torch.zeros(rows, n_groups, device=W.device, dtype=W.dtype)
        group_zeros  = torch.zeros(rows, n_groups, device=W.device, dtype=W.dtype)

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
                groups.append(quantizer)
            for g, q in enumerate(groups):
                group_scales[:, g] = q.scale.view(rows)
                group_zeros[:, g]  = q.zero.view(rows)

        if True:
            total_loss = torch.zeros((), device=self.dev)
            Q = torch.zeros_like(W)
            Q_dense = torch.zeros_like(W) if track_dense_support else None
            support_mask = torch.zeros_like(W, dtype=torch.bool) if track_dense_support else None

            H = torch.cholesky_inverse(self.H.float())
            H = torch.linalg.cholesky(H, upper=True)
            Hinv = H

            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1

                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Q1_dense = torch.zeros_like(W1) if track_dense_support else None
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[i1:i2, i1:i2]
                hdiag1 = torch.diag(Hinv1)

                if prunen > 0:
                    mask1 = torch.zeros_like(W1, dtype=torch.bool)
                else:
                    mask1 = None
                support1 = torch.zeros_like(W1, dtype=torch.bool) if track_dense_support else None

                for i in range(count):
                    w = W1[:, i]
                    d = hdiag1[i]

                    if self.quantizer and groupsize != -1:
                        if not static_groups:
                            if (i1 + i) % groupsize == 0:
                                self.quantizer.find_params(
                                    W[:, (i1 + i):(i1 + i + groupsize)],
                                    weight=True
                                )
                                g = (i1 + i) // groupsize
                                group_scales[:, g] = self.quantizer.scale.view(rows)
                                group_zeros[:, g] = self.quantizer.zero.view(rows)
                        else:
                            idx = i1 + i
                            self.quantizer = groups[idx // groupsize]

                    if prunen > 0:
                        if (i1 + i) % prunem == 0:
                            if prunen == 4 and prunem == 8: ## for 4:8, we assume paired 4:8 (supports nvfp4). note we don't currently support unpaired 4:8.
                                elem_count = min(prunem, count - i)
                                pair_count = elem_count // 2
                                if pair_count > 0:
                                    elem_count = pair_count * 2
                                    block_scores = (W1[:, i:(i + elem_count)] ** 2) / (
                                        hdiag1[i:(i + elem_count)].reshape((1, -1)) ** 2
                                    )
                                    pair_scores = block_scores.reshape(block_scores.shape[0], pair_count, 2).sum(dim=-1)
                                    prune_pairs = min(prunen // 2, pair_count)
                                    idx = torch.topk(pair_scores, prune_pairs, dim=1, largest=False).indices
                                    pair_mask = torch.zeros_like(pair_scores, dtype=torch.bool)
                                    pair_mask.scatter_(1, idx, True)
                                    mask1[:, i:(i + elem_count)] = pair_mask.repeat_interleave(2, dim=1)
                            else: ## N:M sparsity. 2:4.
                                m = min(prunem, count - i)
                                n = min(prunen, m)
                                block_scores = (W1[:, i:(i + m)] ** 2) / (
                                    hdiag1[i:(i + m)].reshape((1, -1)) ** 2
                                )
                                idx = torch.topk(block_scores, n, dim=1, largest=False).indices
                                mask1.scatter_(1, i + idx, True)

                    if track_dense_support: ## save dense weights before masking so GSQ can relearn the support pattern
                        if self.quantizer:
                            q_dense = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                        else:
                            q_dense = w.clone()
                        q = q_dense.clone()
                        if mask1 is not None:
                            q[mask1[:, i]] = 0
                    else:
                        q = w.clone()
                        if mask1 is not None:
                            q[mask1[:, i]] = 0
                        if self.quantizer:
                            q = self.quantizer.quantize(q.unsqueeze(1)).flatten()

                    Q1[:, i] = q
                    if track_dense_support:
                        Q1_dense[:, i] = q_dense
                        support1[:, i] = ~mask1[:, i]
                    Losses1[:, i] = (w - q) ** 2 / (d ** 2)

                    err1 = (w - q) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1

                W[:, i1:i2] = Q1
                if track_dense_support:
                    Q_dense[:, i1:i2] = Q1_dense
                    support_mask[:, i1:i2] = support1
                total_loss += Losses1.sum() / 2
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
            
            Q = W.clone()
            if track_dense_support:
                _validate_sparse_support(support_mask, prunen, prunem, self.name)
                if self.quantizer is None:
                    pruned_nonzero = (~support_mask) & Q.ne(0)
                    if torch.any(pruned_nonzero):
                        bad = pruned_nonzero.nonzero(as_tuple=False)[:8].tolist()
                        raise RuntimeError(
                            f"GPTQ {self.name}: sparse output has nonzero pruned entries; "
                            f"bad [row, col] entries: {bad}"
                        )
            W = self.layer.weight.data.clone()
            gptq_loss = torch.linalg.norm((W - Q.to(self.dtype)) @ self.H, ord='fro')**2
            self.last_gptq_loss = gptq_loss.item()
            if logging is not None:
                logging.info(f'Layer {self.name}: GPTQ Loss = {self.last_gptq_loss}')

        if track_dense_support:
            self.last_init_dense_weight = Q_dense
            self.last_init_support_mask = support_mask

        return Q, group_scales

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        self.last_init_dense_weight = None
        self.last_init_support_mask = None
        torch.cuda.empty_cache()
