import torch

from ..base import BaseQuant


NVFP4_GROUPSIZE = 16  # FP8 scale block on the *compressed* weight (kernel/format constant)
FP4_E2M1_MAX = 6.0


def dense_scale_groupsize(prunen, prunem, base=NVFP4_GROUPSIZE):
    # The sparse NVFP4 kernel applies one FP8 scale per `base` (=16) elements of the
    # *compressed* weight. With N:M sparsity the compressed weight keeps prunen/prunem of
    # the columns, so one compressed block of `base` spans base*prunem//prunen *dense*
    # columns. Pruned positions are zero and don't change the per-block amax, so a
    # per-(base*prunem//prunen) dense scale equals the per-base compressed scale the kernel
    # expects (e.g. 32 dense for both 2:4 and paired-4:8 at 50% sparsity).
    if not prunen or not prunem:
        return base
    if (base * prunem) % prunen != 0:
        raise ValueError(
            f"dense scale groupsize {base}*{prunem}/{prunen} is not integral."
        )
    return base * prunem // prunen
SCALE_EPS = 1e-12
FP8_E4M3_MIN_POSITIVE = float(torch.finfo(torch.float8_e4m3fn).tiny) / 8.0
FP8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
FP4_E2M1_VALUES = (-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5,
                   0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def fp4_project(x):
    with torch.no_grad():
        sign = torch.sign(x)
        ax = x.abs()
        q = torch.empty_like(ax)
        q[ax <= 0.25] = 0.0
        q[(ax > 0.25) & (ax < 0.75)] = 0.5
        q[(ax >= 0.75) & (ax <= 1.25)] = 1.0
        q[(ax > 1.25) & (ax < 1.75)] = 1.5
        q[(ax >= 1.75) & (ax <= 2.5)] = 2.0
        q[(ax > 2.5) & (ax < 3.5)] = 3.0
        q[(ax >= 3.5) & (ax <= 5.0)] = 4.0
        q[ax > 5.0] = 6.0
        return q * sign


def fp4_ste(x):
    projected = fp4_project(x.float())
    return x + (projected.to(x.dtype) - x).detach()


def fp8_project(x):
    return x.to(torch.float8_e4m3fn).to(torch.float32)


class NVFP4Quant(BaseQuant):
    """NVFP4 fake-quant with dynamic, detached scales.

    Used by the main path (paired-4:8 + NVFP4 + learn-weights). Each forward,
    per-group scales are recomputed from the current dense weight, detached,
    and FP8-projected — matching vLLM's dynamic-local runtime path. FP4 values
    are projected through a straight-through estimator on the forward.
    """

    def __init__(self, weight_shape, groupsize, device, dtype,
                 scale_grad_p=None, coupling_stats=None):
        super().__init__()
        # groupsize is the *dense* scale group (= NVFP4_GROUPSIZE * prunem/prunen). It is a
        # multiple of NVFP4_GROUPSIZE so that, once the weight is 2:N compressed, each block
        # maps onto an exact NVFP4_GROUPSIZE compressed block as the kernel requires.
        if groupsize % NVFP4_GROUPSIZE != 0:
            raise ValueError(
                f"NVFP4Quant dense groupsize must be a multiple of {NVFP4_GROUPSIZE}, got {groupsize}."
            )
        rows, cols = weight_shape
        if cols % groupsize != 0:
            raise ValueError(
                f"NVFP4Quant requires weight width divisible by groupsize={groupsize}, got {cols}."
            )
        self.weight_shape = tuple(weight_shape)
        self.groupsize = groupsize
        self.device = device
        self.dtype = dtype
        self.register_buffer("idx", torch.arange(cols, device=device) // groupsize)
        self.scale_grad_p = scale_grad_p
        self._coupling_stats = coupling_stats
        self.register_buffer("scale_eps", torch.tensor(SCALE_EPS, device=device, dtype=torch.float32))
        self.register_buffer("scale_min", torch.tensor(FP8_E4M3_MIN_POSITIVE, device=device, dtype=torch.float32))
        self.register_buffer("scale_max", torch.tensor(FP8_E4M3_MAX, device=device, dtype=torch.float32))

    def fake_quantize(self, effective_weight):
        if self.scale_grad_p is None:
            scale_per_col = self._training_scale_per_col(effective_weight)
        else:
            scale_per_col = self._surrogate_scale_per_col(effective_weight)
        r = effective_weight / scale_per_col
        values = fp4_ste(r)
        out = values * scale_per_col
        # fake_quantize also runs under no_grad (validation, hard-weight builds), where the
        # output has no graph to hook onto.
        if self._coupling_stats is not None and out.requires_grad:
            self._capture_coupling(out, values, r)
        return out

    def _surrogate_scale_per_col(self, effective_weight):
        """Exact amax in the forward, a smooth p-norm derivative in the backward.

        s = max|w|/6 makes ds/dw a ONE-HOT on the argmax element, so the coupling term
        C * ds/dw dumps the whole group's scale gradient on a single weight -- which is why
        it was detached. But that term IS the
        support/value coupling the method claims, so dropping it removes the mechanism from
        the gradient rather than removing a nuisance. A p-norm surrogate spreads the same
        derivative over the largest few elements instead of exactly one. The forward value
        is still the exact amax scale, so nothing about the export changes.
        """
        hard = self._training_scale_per_col(effective_weight)
        blocks = effective_weight.float().reshape(self.weight_shape[0], -1, self.groupsize)
        p = float(self.scale_grad_p)
        soft = blocks.abs().clamp_min(1e-12).pow(p).sum(dim=-1).pow(1.0 / p) / FP4_E2M1_MAX
        soft = torch.clamp(soft, min=self.scale_eps)[:, self.idx]
        return soft + (hard - soft).detach()

    def _capture_coupling(self, out, values, r):
        """Measure C = <g, q-r>, the scale-coupling term the detached path throws away.

        C is also a direct read on the STE residual e = q - r, so one hook answers both
        'how big is the dropped coupling' and 'how biased is the straight-through'.
        Reported against the kept term so the numbers are ratios, not raw scales.
        """
        e = (values - r).detach().float()
        rows, gs = self.weight_shape[0], self.groupsize
        acc = self._coupling_stats

        def _hook(grad, e=e, acc=acc, rows=rows, gs=gs):
            g = grad.detach().float()
            gb = g.reshape(rows, -1, gs)
            eb = e.reshape(rows, -1, gs)
            C = (gb * eb).sum(dim=-1)                    # per group
            # the dropped term lands on the amax element as C * ds/dw = C / FP4_E2M1_MAX
            dropped = C.abs() / FP4_E2M1_MAX
            kept = gb.abs().amax(dim=-1)                 # |g| on that same element, approx
            acc['groups'] += C.numel()
            acc['dropped'] += dropped.sum().item()
            acc['kept'] += kept.sum().item()
            acc['ratio'] += (dropped / kept.clamp_min(1e-30)).sum().item()
            acc['ste_resid'] += eb.abs().mean(dim=-1).sum().item()
            acc['r_mag'] += (r.detach().float().reshape(rows, -1, gs)
                             .abs().mean(dim=-1)).sum().item()
            return grad

        out.register_hook(_hook)

    def quantize_hard(self, effective_weight):
        scales = self._compute_scales(effective_weight)
        hard_scales, global_scale = self._project_hard_scales(scales)
        scale_per_col = hard_scales[:, self.idx] / global_scale
        values = fp4_ste(effective_weight / scale_per_col)
        return values * scale_per_col, hard_scales, global_scale

    def effective_scales(self, effective_weight):
        # The per-group scale in the dense domain: hard_scales carry the global
        # factor, and it is hard_scales/global_scale that multiplies the FP4 codes.
        # Exposed so a caller can compare shipped scales against the initializer's
        # without reaching into the projection internals.
        with torch.no_grad():
            hard_scales, global_scale = self._project_hard_scales(
                self._compute_scales(effective_weight))
            return hard_scales / global_scale

    def _training_scale_per_col(self, effective_weight):
        # Scales are observer metadata only: recompute from current weights,
        # but do not flow gradients through the amax/FP8 projection path.
        with torch.no_grad():
            scales = self._compute_scales(effective_weight)
            hard_scales, global_scale = self._project_hard_scales(scales)
            return hard_scales[:, self.idx] / global_scale

    def _compute_scales(self, effective_weight):
        if effective_weight.shape != self.weight_shape:
            raise ValueError(
                f"NVFP4Quant effective weights must match {self.weight_shape}, "
                f"got {tuple(effective_weight.shape)}."
            )
        if not torch.isfinite(effective_weight).all():
            raise ValueError("NVFP4Quant effective weights must be finite.")

        blocks = effective_weight.float().reshape(self.weight_shape[0], -1, self.groupsize)
        scales = blocks.abs().amax(dim=-1) / FP4_E2M1_MAX
        if not torch.isfinite(scales).all():
            raise ValueError("NVFP4Quant recomputed scales must be finite.")
        return torch.clamp(scales, min=self.scale_eps)

    def _project_hard_scales(self, scales):
        global_scale = self._compute_global_scale(scales)
        scaled = torch.clamp(scales * global_scale, min=self.scale_min, max=self.scale_max)
        projected = fp8_project(scaled)
        if not torch.isfinite(projected).all() or torch.any(projected <= 0):
            raise ValueError("NVFP4Quant hard scales must be finite and positive.")
        return projected, global_scale

    def _compute_global_scale(self, scales):
        if not torch.isfinite(scales).all() or torch.any(scales < 0):
            raise ValueError("NVFP4Quant scales must be finite and non-negative.")
        max_scale = scales.amax().to(torch.float32)
        if max_scale <= 0:
            return torch.ones(1, device=self.device, dtype=torch.float32)
        return (self.scale_max / max_scale).reshape(1).to(torch.float32)


